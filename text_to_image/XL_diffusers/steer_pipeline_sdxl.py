from typing import Any, Dict, List, Optional, Union

import torch

from diffusers.image_processor import PipelineImageInput

try:
    from .rewards import get_reward_function
except ImportError:
    from rewards import get_reward_function

from .pipeline_sdxl import (
    XLA_AVAILABLE,
    OriginalStableDiffusionXL,
    get_scheduler_sigmas_for_timesteps,
    latent_to_decode,
    rescale_noise_cfg,
)

if XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm


class IncrementStableDiffusionXL(OriginalStableDiffusionXL):
    @torch.no_grad()
    def __call__(
        self,
        *args,
        t_loops: int = 1,
        iterative_num_particles: Optional[int] = None,
        iterative_source_timestep_idx: int = 0,
        iterative_reward_fn: str = "ImageReward",
        iterative_reward_temperature: float = 1.0,
        iterative_source_noise_scale: float = 1.0,
        iterative_alpha_coefficient: float = 1.0,
        iterative_metric_to_chase: str = "overall_score",
        **kwargs,
    ):
        self._incremental_cfg = {
            "t_loops": t_loops,
            "iterative_num_particles": iterative_num_particles,
            "iterative_source_timestep_idx": iterative_source_timestep_idx,
            "iterative_reward_fn": iterative_reward_fn,
            "iterative_reward_temperature": iterative_reward_temperature,
            "iterative_source_noise_scale": iterative_source_noise_scale,
            "iterative_alpha_coefficient": iterative_alpha_coefficient,
            "iterative_metric_to_chase": iterative_metric_to_chase,
        }
        try:
            return super().__call__(*args, **kwargs)
        finally:
            self._incremental_cfg = None

    def _sample_latents(
        self,
        *,
        prompt: Union[str, List[str], None],
        timesteps: torch.Tensor,
        num_inference_steps: int,
        num_warmup_steps: int,
        latents: torch.Tensor,
        extra_step_kwargs: Dict[str, Any],
        add_text_embeds: torch.Tensor,
        add_time_ids: torch.Tensor,
        ip_adapter_image: Optional[PipelineImageInput],
        ip_adapter_image_embeds: Optional[List[torch.Tensor]],
        image_embeds: Optional[List[torch.Tensor]],
        prompt_embeds: torch.Tensor,
        timestep_cond: Optional[torch.Tensor],
        callback_on_step_end: Optional[Any],
        callback_on_step_end_tensor_inputs: List[str],
        negative_prompt_embeds: Optional[torch.Tensor],
        negative_pooled_prompt_embeds: Optional[torch.Tensor],
        negative_add_time_ids: Optional[torch.Tensor],
        callback: Optional[Any],
        callback_steps: Optional[int],
    ) -> torch.Tensor:
        cfg = self._incremental_cfg or {}
        t_loops = cfg.get("t_loops", 1)
        iterative_num_particles = cfg.get("iterative_num_particles", None)
        iterative_source_timestep_idx = cfg.get("iterative_source_timestep_idx", 0)
        iterative_reward_fn = cfg.get("iterative_reward_fn", "ImageReward")
        iterative_reward_temperature = cfg.get("iterative_reward_temperature", 1.0)
        iterative_source_noise_scale = cfg.get("iterative_source_noise_scale", 1.0)
        iterative_alpha_coefficient = cfg.get("iterative_alpha_coefficient", 1.0)
        iterative_metric_to_chase = cfg.get("iterative_metric_to_chase", "overall_score")

        if t_loops < 1:
            raise ValueError("`t_loops` must be >= 1 when `steer_scheme == 'incremental'`.")
        if not (0.0 <= iterative_alpha_coefficient <= 1.0):
            raise ValueError("`iterative_alpha_coefficient` must be in [0.0, 1.0].")

        # Align source latent mixing with scheduler noise level (sigma) instead of raw timestep index.
        # source = alpha * mean + (1 - alpha) * eps -> target noise scale is proportional to (1 - alpha).
        step_sigmas = get_scheduler_sigmas_for_timesteps(self.scheduler, timesteps, latents.device)
        target_sigma = (1.0 - iterative_alpha_coefficient) * float(step_sigmas[0])
        alpha_source_timestep_idx = int(torch.argmin(torch.abs(step_sigmas - target_sigma)).item())
        iterative_source_timestep_idx = alpha_source_timestep_idx

        steps_per_full_loop = len(timesteps)
        steps_per_restart_loop = len(timesteps) - iterative_source_timestep_idx
        progress_total = steps_per_full_loop + (t_loops - 1) * steps_per_restart_loop

        n_particles = iterative_num_particles or latents.shape[0]
        if n_particles != latents.shape[0]:
            raise ValueError(
                "`iterative_num_particles` must match the effective batch size for this call. "
                f"Got iterative_num_particles={n_particles}, expected {latents.shape[0]}."
            )

        if isinstance(prompt, str):
            reward_prompts = [prompt] * n_particles
        elif isinstance(prompt, list):
            if len(prompt) == n_particles:
                reward_prompts = prompt
            elif len(prompt) == 1:
                reward_prompts = prompt * n_particles
            else:
                reward_prompts = [prompt[0]] * n_particles
        else:
            reward_prompts = [""] * n_particles

        source_latent = None
        with self.progress_bar(total=progress_total) as progress_bar:
            for _ in range(t_loops):
                if source_latent is not None:
                    base_latents = source_latent.expand(n_particles, -1, -1, -1)
                    latents = base_latents + iterative_source_noise_scale * torch.randn_like(base_latents)

                captured_latents = None
                loop_start_idx = 0 if source_latent is None else iterative_source_timestep_idx
                loop_timesteps = timesteps[loop_start_idx:]
                for i, t in enumerate(loop_timesteps, start=loop_start_idx):
                    if self.interrupt:
                        continue

                    latent_model_input = (
                        torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    )
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    added_cond_kwargs = {
                        "text_embeds": add_text_embeds,
                        "time_ids": add_time_ids,
                    }
                    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                        added_cond_kwargs["image_embeds"] = image_embeds

                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                    if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                        noise_pred = rescale_noise_cfg(
                            noise_pred,
                            noise_pred_text,
                            guidance_rescale=self.guidance_rescale,
                        )

                    latents_dtype = latents.dtype
                    latents = self.scheduler.step(
                        noise_pred,
                        t,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )[0]

                    if i == iterative_source_timestep_idx:
                        captured_latents = latents.detach().clone()

                    if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop(
                            "negative_prompt_embeds", negative_prompt_embeds
                        )
                        add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                        negative_pooled_prompt_embeds = callback_outputs.pop(
                            "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                        )
                        add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                        negative_add_time_ids = callback_outputs.pop(
                            "negative_add_time_ids", negative_add_time_ids
                        )

                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                    ):
                        progress_bar.update()
                        if (
                            callback is not None
                            and callback_steps is not None
                            and i % callback_steps == 0
                        ):
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)

                    if XLA_AVAILABLE:
                        xm.mark_step()

                if captured_latents is None:
                    captured_latents = latents.detach().clone()

                reward_decoded = latent_to_decode(model=self, output_type="pil", latents=latents)
                reward_images = self.image_processor.postprocess(reward_decoded, output_type="pil")
                rewards = get_reward_function(
                    iterative_reward_fn,
                    images=reward_images,
                    prompts=reward_prompts,
                    metric_to_chase=iterative_metric_to_chase,
                )
                rewards = torch.as_tensor(rewards, device=latents.device, dtype=latents.dtype)
                weights = torch.softmax(iterative_reward_temperature * rewards, dim=0)
                weighted_source = torch.sum(
                    captured_latents * weights.view(-1, 1, 1, 1), dim=0, keepdim=True
                )
                eps = torch.randn_like(weighted_source)
                source_latent = (
                    iterative_alpha_coefficient * weighted_source + (1.0 - iterative_alpha_coefficient) * eps
                )

        return latents
