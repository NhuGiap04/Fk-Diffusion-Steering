from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import math
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
        iterative_metric_to_chase = cfg.get("iterative_metric_to_chase", "overall_score")

        if t_loops < 1:
            raise ValueError("`t_loops` must be >= 1 when `steer_scheme == 'incremental'`.")
        if not (0.0 <= iterative_source_noise_scale <= 1.0):
            raise ValueError("`iterative_source_noise_scale` must be in [0.0, 1.0].")

        # Align source latent mixing with scheduler noise level (sigma) instead of raw timestep index.
        # source = source_scale * mean + (1 - source_scale) * eps -> target noise scale is proportional to (1 - source_scale).
        step_sigmas = get_scheduler_sigmas_for_timesteps(self.scheduler, timesteps, latents.device)
        target_sigma = (1.0 - iterative_source_noise_scale) * float(step_sigmas[0])
        iterative_source_timestep_idx = int(torch.argmin(torch.abs(step_sigmas - target_sigma)).item())

        print("Iterative source timestep index:", iterative_source_timestep_idx)


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
                captured_latents = None
                loop_start_idx = 0 if source_latent is None else iterative_source_timestep_idx
                loop_timesteps = timesteps[loop_start_idx:]
                for i, t in enumerate(loop_timesteps, start=loop_start_idx):
                    if self.interrupt:
                        continue

                    if source_latent is not None:
                        base_latents = source_latent.expand(n_particles, -1, -1, -1)
                        latents = base_latents + (1.0 - iterative_source_noise_scale) * torch.randn_like(base_latents)

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
                source_latent = (
                    iterative_source_noise_scale * weighted_source
                )

        return latents


class TemperedDiverseRejuvenatedStableDiffusionXL(OriginalStableDiffusionXL):
    @torch.no_grad()
    def __call__(
        self,
        *args,
        tdr_num_islands: int = 3,
        tdr_target_lambda: float = 2.0,
        tdr_island_lambdas: Optional[List[float]] = None,
        tdr_particles_per_island: Optional[List[int]] = None,
        tdr_resample_frequency: int = 5,
        tdr_resampling_t_start: int = 0,
        tdr_resampling_t_end: Optional[int] = None,
        tdr_ess_threshold_ratio: float = 0.5,
        tdr_rollout_count: int = 2,
        tdr_rollout_steps: int = 2,
        tdr_kappa_start: float = 0.30,
        tdr_kappa_end: float = 0.05,
        tdr_rho_start: float = 0.15,
        tdr_rho_end: float = 0.30,
        tdr_crowding_bandwidth: float = 0.6,
        tdr_rejuvenation_steps: int = 1,
        tdr_rejuvenation_noise_scale: float = 0.08,
        tdr_promotion_every: int = 2,
        tdr_promotion_count: int = 1,
        tdr_immigrant_fraction: float = 0.15,
        tdr_immigrant_islands: Optional[List[int]] = None,
        tdr_immigrant_step_indices: Optional[List[int]] = None,
        tdr_immigrant_coarse_steps: int = 4,
        tdr_final_select_m: int = 4,
        tdr_final_diversity_beta: float = 0.2,
        tdr_reward_fn: str = "ImageReward",
        tdr_metric_to_chase: str = "overall_score",
        **kwargs,
    ):
        self._tdr_fk_cfg = {
            "num_islands": tdr_num_islands,
            "target_lambda": tdr_target_lambda,
            "island_lambdas": tdr_island_lambdas,
            "particles_per_island": tdr_particles_per_island,
            "resample_frequency": tdr_resample_frequency,
            "resampling_t_start": tdr_resampling_t_start,
            "resampling_t_end": tdr_resampling_t_end,
            "ess_threshold_ratio": tdr_ess_threshold_ratio,
            "rollout_count": tdr_rollout_count,
            "rollout_steps": tdr_rollout_steps,
            "kappa_start": tdr_kappa_start,
            "kappa_end": tdr_kappa_end,
            "rho_start": tdr_rho_start,
            "rho_end": tdr_rho_end,
            "crowding_bandwidth": tdr_crowding_bandwidth,
            "rejuvenation_steps": tdr_rejuvenation_steps,
            "rejuvenation_noise_scale": tdr_rejuvenation_noise_scale,
            "promotion_every": tdr_promotion_every,
            "promotion_count": tdr_promotion_count,
            "immigrant_fraction": tdr_immigrant_fraction,
            "immigrant_islands": tdr_immigrant_islands,
            "immigrant_step_indices": tdr_immigrant_step_indices,
            "immigrant_coarse_steps": tdr_immigrant_coarse_steps,
            "final_select_m": tdr_final_select_m,
            "final_diversity_beta": tdr_final_diversity_beta,
            "reward_fn": tdr_reward_fn,
            "metric_to_chase": tdr_metric_to_chase,
        }
        try:
            return super().__call__(*args, **kwargs)
        finally:
            self._tdr_fk_cfg = None

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
        cfg = self._tdr_fk_cfg or {}

        total_particles = latents.shape[0]
        num_islands = int(cfg.get("num_islands", 3))
        if num_islands < 2:
            raise ValueError("`tdr_num_islands` must be >= 2.")
        if total_particles < num_islands:
            raise ValueError(
                "Total particle count must be >= number of islands. "
                f"Got particles={total_particles}, islands={num_islands}."
            )

        island_lambdas = cfg.get("island_lambdas", None)
        target_lambda = float(cfg.get("target_lambda", 2.0))
        if island_lambdas is None:
            island_lambdas_t = torch.linspace(
                0.4 * target_lambda,
                target_lambda,
                steps=num_islands,
                device=latents.device,
                dtype=latents.dtype,
            )
        else:
            if len(island_lambdas) != num_islands:
                raise ValueError("`tdr_island_lambdas` length must equal `tdr_num_islands`.")
            island_lambdas_t = torch.tensor(
                island_lambdas,
                device=latents.device,
                dtype=latents.dtype,
            )
        if not torch.all(island_lambdas_t[1:] > island_lambdas_t[:-1]):
            raise ValueError("`tdr_island_lambdas` must be strictly increasing.")

        particles_per_island = cfg.get("particles_per_island", None)
        if particles_per_island is None:
            base = total_particles // num_islands
            particles_per_island = [base] * num_islands
            particles_per_island[-1] += total_particles - sum(particles_per_island)
        if len(particles_per_island) != num_islands:
            raise ValueError("`tdr_particles_per_island` length must equal number of islands.")
        if sum(particles_per_island) != total_particles:
            raise ValueError(
                "`tdr_particles_per_island` must sum to batch particle count. "
                f"Expected {total_particles}, got {sum(particles_per_island)}."
            )

        resample_frequency = int(cfg.get("resample_frequency", 5))
        if resample_frequency < 1:
            raise ValueError("`tdr_resample_frequency` must be >= 1.")
        resampling_t_start = int(cfg.get("resampling_t_start", 0))
        resampling_t_end = cfg.get("resampling_t_end", None)
        if resampling_t_end is None:
            resampling_t_end = len(timesteps) - 1
        resampling_t_end = int(resampling_t_end)

        ess_threshold_ratio = float(cfg.get("ess_threshold_ratio", 0.5))
        rollout_count = int(cfg.get("rollout_count", 2))
        rollout_steps = int(cfg.get("rollout_steps", 2))
        kappa_start = float(cfg.get("kappa_start", 0.30))
        kappa_end = float(cfg.get("kappa_end", 0.05))
        rho_start = float(cfg.get("rho_start", 0.15))
        rho_end = float(cfg.get("rho_end", 0.30))
        crowding_bandwidth = float(cfg.get("crowding_bandwidth", 0.6))
        rejuvenation_steps = int(cfg.get("rejuvenation_steps", 1))
        rejuvenation_noise_scale = float(cfg.get("rejuvenation_noise_scale", 0.08))
        promotion_every = int(cfg.get("promotion_every", 2))
        promotion_count = int(cfg.get("promotion_count", 1))
        immigrant_fraction = float(cfg.get("immigrant_fraction", 0.15))
        immigrant_islands = cfg.get("immigrant_islands", None)
        immigrant_step_indices = cfg.get("immigrant_step_indices", None)
        immigrant_coarse_steps = int(cfg.get("immigrant_coarse_steps", 4))
        final_select_m = int(cfg.get("final_select_m", 4))
        final_diversity_beta = float(cfg.get("final_diversity_beta", 0.2))
        reward_fn = cfg.get("reward_fn", "ImageReward")
        metric_to_chase = cfg.get("metric_to_chase", "overall_score")

        if immigrant_islands is None:
            immigrant_islands = [0]
        immigrant_islands = [int(x) for x in immigrant_islands if 0 <= int(x) < num_islands]
        if not immigrant_islands:
            immigrant_islands = [0]

        if immigrant_step_indices is None:
            last_idx = len(timesteps) - 1
            immigrant_step_indices = sorted(
                {
                    max(1, int(round(0.5 * last_idx))),
                    max(1, int(round(0.75 * last_idx))),
                }
            )
        immigrant_step_indices = set(int(x) for x in immigrant_step_indices)

        def _prompt_list(input_prompt: Union[str, List[str], None], n: int) -> List[str]:
            if isinstance(input_prompt, str):
                return [input_prompt] * n
            if isinstance(input_prompt, list):
                if len(input_prompt) == n:
                    return list(input_prompt)
                if len(input_prompt) == 1:
                    return input_prompt * n
                return [input_prompt[0]] * n
            return [""] * n

        reward_prompts = _prompt_list(prompt, total_particles)

        island_offsets: List[int] = []
        cursor = 0
        for size in particles_per_island:
            island_offsets.append(cursor)
            cursor += size

        island_latents: List[torch.Tensor] = []
        island_prompt_lists: List[List[str]] = []
        island_prev_rewards: List[torch.Tensor] = []

        for m, size in enumerate(particles_per_island):
            start = island_offsets[m]
            end = start + size
            chunk = latents[start:end].clone()
            if m > 0:
                chunk = chunk + 0.01 * m * torch.randn_like(chunk)
            island_latents.append(chunk)
            island_prompt_lists.append(reward_prompts[start:end])
            island_prev_rewards.append(torch.zeros(size, device=latents.device, dtype=latents.dtype))

        def _cfg_slice(tensor: Optional[torch.Tensor], idx: torch.Tensor) -> Optional[torch.Tensor]:
            if tensor is None:
                return None
            if not self.do_classifier_free_guidance:
                return tensor[idx]
            half = tensor.shape[0] // 2
            return torch.cat([tensor[:half][idx], tensor[half:][idx]], dim=0)

        def _ip_adapter_slice(
            embeds: Optional[List[torch.Tensor]], idx: torch.Tensor
        ) -> Optional[List[torch.Tensor]]:
            if embeds is None:
                return None
            sliced = []
            for item in embeds:
                if not self.do_classifier_free_guidance:
                    sliced.append(item[idx])
                else:
                    half = item.shape[0] // 2
                    sliced.append(torch.cat([item[:half][idx], item[half:][idx]], dim=0))
            return sliced

        def _forward_one_step(
            *,
            local_latents: torch.Tensor,
            t: torch.Tensor,
            idx: torch.Tensor,
            do_return_x0: bool = True,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            local_prompt_embeds = _cfg_slice(prompt_embeds, idx)
            local_add_text_embeds = _cfg_slice(add_text_embeds, idx)
            local_add_time_ids = _cfg_slice(add_time_ids, idx)
            local_timestep_cond = None
            if timestep_cond is not None:
                local_timestep_cond = timestep_cond[idx]
            local_image_embeds = _ip_adapter_slice(image_embeds, idx)

            latent_model_input = (
                torch.cat([local_latents] * 2)
                if self.do_classifier_free_guidance
                else local_latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            added_cond_kwargs = {
                "text_embeds": local_add_text_embeds,
                "time_ids": local_add_time_ids,
            }
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                added_cond_kwargs["image_embeds"] = local_image_embeds

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=local_prompt_embeds,
                timestep_cond=local_timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                if self.guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self.guidance_rescale,
                    )

            step_dict = self.scheduler.step(
                noise_pred,
                t,
                local_latents,
                **extra_step_kwargs,
                return_dict=True,
            )
            if do_return_x0:
                return step_dict.prev_sample, step_dict.pred_original_sample
            return step_dict.prev_sample, None

        def _compute_rewards(
            x0_preds: torch.Tensor,
            prompt_list: Sequence[str],
        ) -> torch.Tensor:
            decoded = latent_to_decode(model=self, output_type="pil", latents=x0_preds)
            images = self.image_processor.postprocess(decoded, output_type="pil")
            scores = get_reward_function(
                reward_fn,
                images=images,
                prompts=list(prompt_list),
                metric_to_chase=metric_to_chase,
            )
            return torch.as_tensor(scores, device=x0_preds.device, dtype=x0_preds.dtype)

        def _crowding_scores(latent_batch: torch.Tensor, bandwidth: float) -> torch.Tensor:
            if latent_batch.shape[0] <= 1:
                return torch.zeros(latent_batch.shape[0], device=latent_batch.device, dtype=latent_batch.dtype)
            # cdist on CUDA does not support float16; use float32 distances for stability.
            z = latent_batch.flatten(start_dim=1).float()
            z = z / (torch.norm(z, dim=1, keepdim=True) + 1e-6)
            d2 = torch.cdist(z, z, p=2.0).pow(2)
            logits = -d2 / max(bandwidth * bandwidth, 1e-5)
            return torch.logsumexp(logits, dim=1).to(dtype=latent_batch.dtype)

        def _linear_schedule(step_idx: int, total_steps: int, start: float, end: float) -> float:
            if total_steps <= 1:
                return end
            frac = float(step_idx) / float(total_steps - 1)
            return (1.0 - frac) * start + frac * end

        def _ess(weights: torch.Tensor) -> torch.Tensor:
            total = torch.clamp(weights.sum(), min=1e-12)
            norm_w = weights / total
            return 1.0 / torch.clamp(norm_w.pow(2).sum(), min=1e-12)

        def _lookahead_stats(
            *,
            base_latents: torch.Tensor,
            island_idx: int,
            step_idx: int,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            if rollout_count < 1 or rollout_steps < 1:
                mu = island_prev_rewards[island_idx]
                sigma = torch.zeros_like(mu)
                return mu, sigma

            size = base_latents.shape[0]
            reward_matrix = torch.zeros(
                (rollout_count, size),
                device=base_latents.device,
                dtype=base_latents.dtype,
            )
            island_start = island_offsets[island_idx]
            island_end = island_start + size
            idx = torch.arange(island_start, island_end, device=base_latents.device)

            for k in range(rollout_count):
                rollout_latents = base_latents + 0.02 * torch.randn_like(base_latents)
                max_j = min(len(timesteps) - 1, step_idx + rollout_steps)
                rollout_x0 = None
                for j in range(step_idx, max_j + 1):
                    rollout_latents, rollout_x0 = _forward_one_step(
                        local_latents=rollout_latents,
                        t=timesteps[j],
                        idx=idx,
                        do_return_x0=True,
                    )
                if rollout_x0 is None:
                    rollout_x0 = rollout_latents
                reward_matrix[k] = _compute_rewards(rollout_x0, island_prompt_lists[island_idx])

            return reward_matrix.mean(dim=0), reward_matrix.std(dim=0, unbiased=False)

        def _rejuvenate(
            *,
            latent_batch: torch.Tensor,
            island_idx: int,
            t: torch.Tensor,
        ) -> torch.Tensor:
            if rejuvenation_steps < 1 or rejuvenation_noise_scale <= 0.0:
                return latent_batch
            size = latent_batch.shape[0]
            island_start = island_offsets[island_idx]
            island_end = island_start + size
            idx = torch.arange(island_start, island_end, device=latent_batch.device)
            out = latent_batch
            for _ in range(rejuvenation_steps):
                out = out + rejuvenation_noise_scale * torch.randn_like(out)
                out, _ = _forward_one_step(local_latents=out, t=t, idx=idx, do_return_x0=False)
            return out

        def _coarse_denoise_to_step(
            *,
            step_idx: int,
            latent_batch: torch.Tensor,
            island_idx: int,
        ) -> torch.Tensor:
            if step_idx <= 0:
                return latent_batch
            size = latent_batch.shape[0]
            island_start = island_offsets[island_idx]
            island_end = island_start + size
            idx = torch.arange(island_start, island_end, device=latent_batch.device)

            n_coarse = max(1, immigrant_coarse_steps)
            coarse_points = torch.linspace(
                0,
                step_idx,
                steps=min(n_coarse, step_idx + 1),
                device=latent_batch.device,
            ).round().long().tolist()
            coarse_points = sorted(set(int(x) for x in coarse_points))

            out = latent_batch
            for c in coarse_points:
                out, _ = _forward_one_step(
                    local_latents=out,
                    t=timesteps[c],
                    idx=idx,
                    do_return_x0=False,
                )
            return out

        def _select_diverse_subset(
            *,
            latent_batch: torch.Tensor,
            rewards: torch.Tensor,
            subset_size: int,
            beta: float,
        ) -> torch.Tensor:
            n = latent_batch.shape[0]
            subset_size = max(1, min(subset_size, n))
            if subset_size == n:
                return torch.arange(n, device=latent_batch.device)

            # cdist on CUDA does not support float16; use float32 distances for stability.
            z = latent_batch.flatten(start_dim=1).float()
            z = z / (torch.norm(z, dim=1, keepdim=True) + 1e-6)
            d2 = torch.cdist(z, z, p=2.0).pow(2)
            kernel = torch.exp(-d2 / 0.5)
            selected: List[int] = []
            candidates = set(range(n))
            first_idx = int(torch.argmax(rewards).item())
            selected.append(first_idx)
            candidates.remove(first_idx)

            while len(selected) < subset_size and candidates:
                best_idx = None
                best_score = -1e30
                for c in list(candidates):
                    idxs = selected + [c]
                    k_sub = kernel[idxs][:, idxs] + 1e-4 * torch.eye(
                        len(idxs),
                        device=kernel.device,
                        dtype=kernel.dtype,
                    )
                    sign, logdet = torch.linalg.slogdet(k_sub)
                    if sign <= 0:
                        logdet_val = -1e6
                    else:
                        logdet_val = float(logdet.item())
                    score = float(rewards[idxs].sum().item()) + beta * logdet_val
                    if score > best_score:
                        best_score = score
                        best_idx = c
                if best_idx is None:
                    break
                selected.append(int(best_idx))
                candidates.remove(int(best_idx))

            return torch.tensor(selected, device=latent_batch.device, dtype=torch.long)

        hottest_latents = island_latents[-1]
        resample_events = 0

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                island_x0_preds: List[torch.Tensor] = []
                for m in range(num_islands):
                    size = particles_per_island[m]
                    start = island_offsets[m]
                    end = start + size
                    idx = torch.arange(start, end, device=latents.device)
                    island_latents[m], x0_preds = _forward_one_step(
                        local_latents=island_latents[m],
                        t=t,
                        idx=idx,
                        do_return_x0=True,
                    )
                    island_x0_preds.append(x0_preds)

                hottest_latents = island_latents[-1]

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and callback_steps is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, hottest_latents)

                if XLA_AVAILABLE:
                    xm.mark_step()

                should_consider_resample = (
                    resampling_t_start <= i <= resampling_t_end
                    and (i % resample_frequency == 0)
                ) or (i == len(timesteps) - 1)
                if not should_consider_resample:
                    continue

                kappa_t = _linear_schedule(i, len(timesteps), kappa_start, kappa_end)
                rho_t = _linear_schedule(i, len(timesteps), rho_start, rho_end)

                any_resampled = False
                for m in range(num_islands):
                    rewards_now = _compute_rewards(
                        island_x0_preds[m],
                        island_prompt_lists[m],
                    )
                    reward_delta = rewards_now - island_prev_rewards[m]
                    fk_weights = torch.exp(island_lambdas_t[m] * reward_delta)
                    fk_weights = torch.clamp(fk_weights, min=1e-10, max=1e10)
                    fk_weights[torch.isnan(fk_weights)] = 1e-10

                    island_ess = _ess(fk_weights)
                    needs_resample = (
                        island_ess < ess_threshold_ratio * particles_per_island[m]
                    ) or (i == len(timesteps) - 1)

                    if needs_resample:
                        mu, sigma = _lookahead_stats(
                            base_latents=island_latents[m],
                            island_idx=m,
                            step_idx=i,
                        )
                        crowding = _crowding_scores(island_latents[m], crowding_bandwidth)
                        aux_scores = island_lambdas_t[m] * mu + kappa_t * sigma - rho_t * crowding
                        probs = torch.softmax(aux_scores, dim=0)
                        if torch.any(torch.isnan(probs)) or float(probs.sum().item()) <= 0:
                            probs = fk_weights / torch.clamp(fk_weights.sum(), min=1e-12)

                        ancestors = torch.multinomial(
                            probs,
                            num_samples=particles_per_island[m],
                            replacement=True,
                        )
                        island_latents[m] = island_latents[m][ancestors]
                        island_prev_rewards[m] = rewards_now[ancestors]
                        island_prompt_lists[m] = [island_prompt_lists[m][int(a.item())] for a in ancestors]
                        island_latents[m] = _rejuvenate(
                            latent_batch=island_latents[m],
                            island_idx=m,
                            t=t,
                        )
                        any_resampled = True
                    else:
                        island_prev_rewards[m] = rewards_now

                if any_resampled:
                    resample_events += 1

                if any_resampled and promotion_every > 0 and (resample_events % promotion_every == 0):
                    for m in range(num_islands - 1):
                        cooler_rewards = island_prev_rewards[m]
                        hotter_rewards = island_prev_rewards[m + 1]
                        k = max(0, min(promotion_count, cooler_rewards.shape[0], hotter_rewards.shape[0]))
                        if k == 0:
                            continue
                        promote_idx = torch.topk(cooler_rewards, k=k, largest=True).indices
                        replace_idx = torch.topk(hotter_rewards, k=k, largest=False).indices
                        island_latents[m + 1][replace_idx] = island_latents[m][promote_idx]
                        island_prev_rewards[m + 1][replace_idx] = island_prev_rewards[m][promote_idx]
                        for r_i, p_i in zip(replace_idx.tolist(), promote_idx.tolist()):
                            island_prompt_lists[m + 1][r_i] = island_prompt_lists[m][p_i]

                if i in immigrant_step_indices and immigrant_fraction > 0.0:
                    for m in immigrant_islands:
                        island_size = particles_per_island[m]
                        immigrant_count = int(math.ceil(immigrant_fraction * island_size))
                        immigrant_count = max(0, min(immigrant_count, island_size))
                        if immigrant_count == 0:
                            continue

                        replace_idx = torch.topk(
                            island_prev_rewards[m],
                            k=immigrant_count,
                            largest=False,
                        ).indices
                        fresh = torch.randn_like(island_latents[m][replace_idx])
                        fresh = _coarse_denoise_to_step(
                            step_idx=i,
                            latent_batch=fresh,
                            island_idx=m,
                        )
                        island_latents[m][replace_idx] = fresh
                        island_prev_rewards[m][replace_idx] = island_prev_rewards[m].mean()

                hottest_latents = island_latents[-1]

        final_rewards = _compute_rewards(hottest_latents, island_prompt_lists[-1])
        selected_idx = _select_diverse_subset(
            latent_batch=hottest_latents,
            rewards=final_rewards,
            subset_size=final_select_m,
            beta=final_diversity_beta,
        )
        return hottest_latents[selected_idx]
