from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from diffusers.image_processor import PipelineImageInput

from .pipeline_sdxl import BaseSDXL, get_scheduler_sigmas_for_timesteps
from .rewards import get_reward_function


def score_log_prob_reward(
    x_t: torch.Tensor,
    support_x0: torch.Tensor,
    sigma_t: Union[float, torch.Tensor],
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Approximate ``grad_{x_t} log p(x_t | good)`` from supported clean samples.

    Matches steer.py DDPM-form score approximation:
        q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) I)

    Here we receive ``sigma_t`` from the SDXL scheduler and recover
    ``alpha_bar_t`` using:
        sigma_t^2 = (1 - alpha_bar_t) / alpha_bar_t
        alpha_bar_t = 1 / (1 + sigma_t^2)
    """
    if x_t.ndim != 2 or support_x0.ndim != 2:
        raise ValueError("x_t and support_x0 must have shape [B, D]")
    if x_t.shape[1] != support_x0.shape[1]:
        raise ValueError("x_t and support_x0 must have matching feature dimension")
    if support_x0.shape[0] == 0:
        raise ValueError("support_x0 is empty; cannot approximate score")

    device = x_t.device
    dtype = x_t.dtype
    support_x0 = support_x0.to(device=device, dtype=dtype)

    sigma_t_tensor: torch.Tensor = torch.as_tensor(sigma_t, device=device, dtype=dtype)
    alpha_bar_t = 1.0 / torch.clamp(1.0 + sigma_t_tensor * sigma_t_tensor, min=eps)
    sqrt_alpha_bar_t = torch.sqrt(torch.clamp(alpha_bar_t, min=eps))
    var_t = torch.clamp(1.0 - alpha_bar_t, min=eps)

    means = sqrt_alpha_bar_t * support_x0
    delta = x_t[:, None, :] - means[None, :, :]
    sq_dist = (delta**2).sum(dim=-1)

    log_w = -0.5 * sq_dist / var_t
    w = torch.softmax(log_w, dim=1)

    component_scores = -delta / var_t
    return (w[:, :, None] * component_scores).sum(dim=1)


def stein_variational_vector_field(
    x: torch.Tensor,
    score: torch.Tensor,
    bandwidth: Optional[float] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute the empirical SVGD vector field for particle updates."""
    if x.ndim != 2 or score.ndim != 2:
        raise ValueError("x and score must have shape [N, D]")
    if x.shape != score.shape:
        raise ValueError("x and score must have the same shape")

    n, _ = x.shape
    if n == 0:
        return torch.empty_like(x)

    diff = x[:, None, :] - x[None, :, :]
    sq_dist = (diff**2).sum(dim=-1)

    if bandwidth is None:
        off_diag = sq_dist[~torch.eye(n, dtype=torch.bool, device=x.device)]
        if off_diag.numel() == 0:
            h = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        else:
            h = torch.median(off_diag).clamp_min(eps)
    else:
        h = torch.as_tensor(bandwidth, device=x.device, dtype=x.dtype).clamp_min(eps)

    k_mat = torch.exp(-sq_dist / h)
    attractive = k_mat.transpose(0, 1) @ score
    repulsive = (-2.0 / h) * (k_mat[:, :, None] * diff).sum(dim=0)
    return (attractive + repulsive) / float(n)


@torch.no_grad()
def stein_step(
    x_t: torch.Tensor,
    accepted_x0: torch.Tensor,
    sigma_t: Union[float, torch.Tensor],
    rejected_x0: Optional[torch.Tensor] = None,
    step_size: float = 0.05,
    rejected_penalty: float = 0.0,
    bandwidth: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One SVGD update step on latent particles at the current scheduler sigma."""
    if x_t.ndim < 2 or accepted_x0.ndim < 2:
        raise ValueError("x_t and accepted_x0 must have at least 2 dims [B, ...]")
    if x_t.shape[1:] != accepted_x0.shape[1:]:
        raise ValueError(
            "x_t and accepted_x0 must have matching non-batch dimensions "
            f"(got {x_t.shape[1:]} vs {accepted_x0.shape[1:]})"
        )
    if rejected_x0 is not None and rejected_x0.ndim < 2:
        raise ValueError("rejected_x0 must have at least 2 dims [B, ...]")
    if rejected_x0 is not None and x_t.shape[1:] != rejected_x0.shape[1:]:
        raise ValueError(
            "x_t and rejected_x0 must have matching non-batch dimensions "
            f"(got {x_t.shape[1:]} vs {rejected_x0.shape[1:]})"
        )

    x_shape = x_t.shape
    out_dtype = x_t.dtype
    x_flat = x_t.reshape(x_t.shape[0], -1).to(dtype=torch.float32)
    accepted_flat = accepted_x0.reshape(accepted_x0.shape[0], -1).to(dtype=torch.float32)

    accepted_score = score_log_prob_reward(
        x_t=x_flat,
        support_x0=accepted_flat,
        sigma_t=sigma_t,
    )
    accepted_phi = stein_variational_vector_field(x=x_flat, score=accepted_score, bandwidth=bandwidth)

    phi = accepted_phi
    if rejected_x0 is not None and rejected_x0.shape[0] > 0 and rejected_penalty > 0.0:
        rejected_flat = rejected_x0.reshape(rejected_x0.shape[0], -1).to(dtype=torch.float32)
        rejected_score = score_log_prob_reward(
            x_t=x_flat,
            support_x0=rejected_flat,
            sigma_t=sigma_t,
        )
        rejected_phi = stein_variational_vector_field(x=x_flat, score=rejected_score, bandwidth=bandwidth)
        phi = accepted_phi - rejected_penalty * rejected_phi

    x_next = x_flat + step_size * phi
    return x_next.to(dtype=out_dtype).reshape(x_shape), phi.to(dtype=out_dtype).reshape(x_shape)


@torch.no_grad()
def steer_sample(
    model: BaseSDXL,
    prompt: Union[str, List[str]],
    *,
    accepted_x0: Optional[torch.Tensor] = None,
    rejected_x0: Optional[torch.Tensor] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    denoising_end: Optional[float] = None,
    guidance_scale: float = 5.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: int = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    output_type: str = "pil",
    return_dict: bool = True,
    steer_start_timestep: Optional[int] = None,
    steer_end_timestep: int = 0,
    stein_step_size: float = 0.05,
    stein_bandwidth: Optional[float] = None,
    stein_rejected_penalty: float = 0.0,
    callback_on_step_end: Optional[Any] = None,
    callback_on_step_end_tensor_inputs: Optional[List[str]] = None,
    **kwargs: Any,
) -> Tuple[Any, List[torch.Tensor]]:
    """
    Prompt-conditioned SDXL sampling with optional Stein updates in the denoising loop.

    This follows `BaseSDXL` sampling flow and injects Stein steps through
    `callback_on_step_end`, rather than using a toy DDPM `p_sample` API.

    Returns:
        Tuple `(pipeline_output, latent_trajectory)`.
        `latent_trajectory` stores one latent tensor per denoising step (CPU).
    """
    if callback_on_step_end_tensor_inputs is None:
        callback_on_step_end_tensor_inputs = ["latents"]
    elif "latents" not in callback_on_step_end_tensor_inputs:
        callback_on_step_end_tensor_inputs = [
            *callback_on_step_end_tensor_inputs,
            "latents",
        ]

    # When accepted particles are provided and start timestep is omitted,
    # steer through the full denoising horizon.
    if accepted_x0 is not None and accepted_x0.shape[0] > 0 and steer_start_timestep is None:
        steer_start_timestep = int(model.scheduler.config.num_train_timesteps - 1)

    use_stein = (
        accepted_x0 is not None
        and accepted_x0.shape[0] > 0
        and steer_start_timestep is not None
    )

    if use_stein:
        assert steer_start_timestep is not None
        t_hi = max(steer_start_timestep, steer_end_timestep)
        t_lo = min(steer_start_timestep, steer_end_timestep)

    latent_trajectory: List[torch.Tensor] = []

    def _combined_step_callback(pipe, step_idx: int, t, callback_kwargs: Dict[str, Any]):
        callback_updates: Dict[str, Any] = {}
        current_latents = callback_kwargs["latents"]

        if use_stein:
            assert accepted_x0 is not None
            timestep_value = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            if t_lo <= timestep_value <= t_hi:
                sigma_t = get_scheduler_sigmas_for_timesteps(
                    scheduler=pipe.scheduler,
                    timesteps=[timestep_value],
                    device=current_latents.device,
                )[0]
                accepted_particles = accepted_x0.to(
                    device=current_latents.device,
                    dtype=current_latents.dtype,
                )
                rejected_particles: Optional[torch.Tensor] = None
                if rejected_x0 is not None and rejected_x0.shape[0] > 0:
                    rejected_particles = rejected_x0.to(
                        device=current_latents.device,
                        dtype=current_latents.dtype,
                    )
                current_latents, _ = stein_step(
                    x_t=current_latents,
                    accepted_x0=accepted_particles,
                    rejected_x0=rejected_particles,
                    sigma_t=sigma_t,
                    step_size=stein_step_size,
                    rejected_penalty=stein_rejected_penalty,
                    bandwidth=stein_bandwidth,
                )

        if callback_on_step_end is not None:
            user_kwargs = dict(callback_kwargs)
            user_kwargs["latents"] = current_latents
            user_updates = callback_on_step_end(pipe, step_idx, t, user_kwargs)
            if user_updates is not None:
                callback_updates.update(user_updates)
                current_latents = callback_updates.pop("latents", current_latents)

        latent_trajectory.append(current_latents.detach().cpu())
        callback_updates["latents"] = current_latents
        return callback_updates

    model_call_kwargs: Dict[str, Any] = {}
    if timesteps is not None:
        model_call_kwargs["timesteps"] = timesteps
    if sigmas is not None:
        model_call_kwargs["sigmas"] = sigmas

    result = model(
        prompt=prompt,
        prompt_2=prompt_2,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        denoising_end=denoising_end,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        num_images_per_prompt=num_images_per_prompt,
        eta=eta,
        generator=generator,
        latents=latents,
        ip_adapter_image=ip_adapter_image,
        ip_adapter_image_embeds=ip_adapter_image_embeds,
        output_type=output_type,
        return_dict=return_dict,
        callback_on_step_end=_combined_step_callback,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        **model_call_kwargs,
        **kwargs,
    )

    return result, latent_trajectory


def _resolve_prompt_and_particles(
    prompt: Union[str, List[str]],
    num_particles: int,
) -> Tuple[Union[str, List[str]], int, List[str]]:
    """Normalize prompt inputs for generation and reward evaluation."""
    if num_particles <= 0:
        raise ValueError("num_particles must be > 0")

    if isinstance(prompt, str):
        return prompt, num_particles, [prompt] * num_particles

    if not isinstance(prompt, list) or len(prompt) == 0:
        raise ValueError("prompt must be a non-empty string or list of strings")

    if len(prompt) == num_particles:
        return prompt, 1, prompt

    if len(prompt) == 1:
        p = prompt[0]
        return p, num_particles, [p] * num_particles

    raise ValueError(
        "When prompt is a list, its length must be 1 or num_particles "
        f"(got len(prompt)={len(prompt)}, num_particles={num_particles})"
    )


@torch.no_grad()
def split_samples(
    model: BaseSDXL,
    prompt: Union[str, List[str]],
    *,
    num_particles: int,
    threshold: float,
    guidance_reward_fn: str = "ImageReward",
    metric_to_chase: Optional[str] = None,
    steer_start_timestep: Optional[int] = None,
    steer_end_timestep: int = 0,
    stein_step_size: float = 0.05,
    stein_bandwidth: Optional[float] = None,
    stein_rejected_penalty: float = 0.0,
    accepted_x0: Optional[torch.Tensor] = None,
    rejected_x0: Optional[torch.Tensor] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Sample particles with optional Stein steering, then split by reward threshold.

    Returns a dictionary containing rewards, accepted/rejected particles, and outputs.
    """
    generation_prompt, num_images_per_prompt, reward_prompts = _resolve_prompt_and_particles(
        prompt=prompt,
        num_particles=num_particles,
    )

    result, latent_trajectory = steer_sample(
        model=model,
        prompt=generation_prompt,
        accepted_x0=accepted_x0,
        rejected_x0=rejected_x0,
        steer_start_timestep=steer_start_timestep,
        steer_end_timestep=steer_end_timestep,
        stein_step_size=stein_step_size,
        stein_bandwidth=stein_bandwidth,
        stein_rejected_penalty=stein_rejected_penalty,
        num_images_per_prompt=num_images_per_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        output_type="pil",
        return_dict=True,
        **kwargs,
    )

    images = result.images
    rewards = torch.as_tensor(
        get_reward_function(
            reward_name=guidance_reward_fn,
            images=images,
            prompts=reward_prompts,
            metric_to_chase=metric_to_chase or "overall_score",
        ),
        dtype=torch.float32,
    )

    final_latents = latent_trajectory[-1]
    accept_mask = rewards >= threshold
    reject_mask = ~accept_mask

    accepted_latents = final_latents[accept_mask]
    rejected_latents = final_latents[reject_mask]

    accepted_images = [img for i, img in enumerate(images) if bool(accept_mask[i].item())]
    rejected_images = [img for i, img in enumerate(images) if bool(reject_mask[i].item())]

    return {
        "result": result,
        "latent_trajectory": latent_trajectory,
        "rewards": rewards,
        "accepted_x0": accepted_latents,
        "rejected_x0": rejected_latents,
        "accepted_images": accepted_images,
        "rejected_images": rejected_images,
        "threshold": float(threshold),
    }


@torch.no_grad()
def iterative_sample_with_stein(
    model: BaseSDXL,
    prompt: Union[str, List[str]],
    *,
    num_loops: int,
    num_particles: int,
    steer_start_timestep: int,
    steer_end_timestep: int = 0,
    base_threshold: float = 0.0,
    stein_step_size: float = 0.05,
    stein_bandwidth: Optional[float] = None,
    stein_rejected_penalty: float = 0.0,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    guidance_reward_fn: str = "ImageReward",
    metric_to_chase: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Iteratively sample and update accepted pool, mirroring steer.py loop logic.

    Loop behavior:
    1) First loop samples without Stein steering.
    2) Later loops steer using an accumulated pool of accepted particles.
    3) Base threshold tracks the best mean reward seen so far.
    4) Whenever base threshold increases, pool samples with reward below it are pruned.
    """
    if num_loops <= 0:
        raise ValueError("num_loops must be > 0")

    all_results: List[Any] = []
    all_trajectories: List[List[torch.Tensor]] = []
    all_rewards: List[torch.Tensor] = []
    all_thresholds: List[float] = []
    all_accepted: List[torch.Tensor] = []
    all_rejected: List[torch.Tensor] = []

    best_mean_reward = float(base_threshold)
    running_base_threshold = float(base_threshold)
    accepted_pool: Optional[torch.Tensor] = None
    accepted_pool_rewards: Optional[torch.Tensor] = None
    rejected_pool: Optional[torch.Tensor] = None

    for loop_idx in range(num_loops):
        use_stein = (
            loop_idx > 0
            and accepted_pool is not None
            and accepted_pool.shape[0] > 0
        )

        split = split_samples(
            model=model,
            prompt=prompt,
            num_particles=num_particles,
            threshold=running_base_threshold,
            guidance_reward_fn=guidance_reward_fn,
            metric_to_chase=metric_to_chase,
            steer_start_timestep=steer_start_timestep,
            steer_end_timestep=steer_end_timestep,
            stein_step_size=stein_step_size,
            stein_bandwidth=stein_bandwidth,
            stein_rejected_penalty=stein_rejected_penalty,
            accepted_x0=accepted_pool if use_stein else None,
            rejected_x0=rejected_pool if use_stein else None,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs,
        )

        rewards = split["rewards"]
        best_reward = float(rewards.max().item())
        mean_reward = float(rewards.mean().item())
        std_reward = float(rewards.std().item())

        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward

        # Base threshold is the best mean reward observed so far.
        running_base_threshold = best_mean_reward + std_reward / 2.0
        threshold = running_base_threshold

        final_latents = split["latent_trajectory"][-1]
        accept_mask = rewards >= threshold
        reject_mask = ~accept_mask

        accepted_latents = final_latents[accept_mask]
        rejected_latents = final_latents[reject_mask]
        accepted_rewards = rewards[accept_mask].to(dtype=torch.float32).cpu()

        if accepted_latents.shape[0] > 0:
            if accepted_pool is None:
                accepted_pool = accepted_latents
                accepted_pool_rewards = accepted_rewards
            else:
                accepted_pool = torch.cat([accepted_pool, accepted_latents], dim=0)
                assert accepted_pool_rewards is not None
                accepted_pool_rewards = torch.cat([accepted_pool_rewards, accepted_rewards], dim=0)

        if rejected_latents.shape[0] > 0:
            if rejected_pool is None:
                rejected_pool = rejected_latents
            else:
                rejected_pool = torch.cat([rejected_pool, rejected_latents], dim=0)

        # Keep pool and pool rewards aligned, and move pruned entries to rejected pool.
        if accepted_pool is not None:
            assert accepted_pool_rewards is not None
            keep_mask = accepted_pool_rewards >= threshold
            pruned_to_rejected = accepted_pool[~keep_mask]
            pruned_accepted_pool = accepted_pool[keep_mask]
            pruned_accepted_rewards = accepted_pool_rewards[keep_mask]

            if pruned_to_rejected.shape[0] > 0:
                if rejected_pool is None:
                    rejected_pool = pruned_to_rejected
                else:
                    rejected_pool = torch.cat([rejected_pool, pruned_to_rejected], dim=0)

            if pruned_accepted_pool.shape[0] == 0:
                accepted_pool = None
                accepted_pool_rewards = None
            else:
                accepted_pool = pruned_accepted_pool
                accepted_pool_rewards = pruned_accepted_rewards

        all_results.append(split["result"])
        all_trajectories.append(split["latent_trajectory"])
        all_rewards.append(rewards)
        all_thresholds.append(float(threshold))
        all_accepted.append(accepted_latents)
        all_rejected.append(rejected_latents)

        print(
            f"loop={loop_idx + 1}/{num_loops} "
            f"use_stein={use_stein} "
            f"best_reward={best_reward:.4f} "
            f"mean_reward={mean_reward:.4f} "
            f"std_reward={std_reward:.4f} "
            f"threshold={threshold:.4f} "
            f"accepted={accepted_latents.shape[0]} "
            f"rejected={rejected_latents.shape[0]} "
            f"accepted_pool={0 if accepted_pool is None else accepted_pool.shape[0]} "
            f"rejected_pool={0 if rejected_pool is None else rejected_pool.shape[0]}"
        )

    return {
        "results": all_results,
        "trajectories": all_trajectories,
        "rewards": all_rewards,
        "thresholds": all_thresholds,
        "accepted": all_accepted,
        "rejected": all_rejected,
        "accepted_pool": accepted_pool,
        "accepted_pool_rewards": accepted_pool_rewards,
        "rejected_pool": rejected_pool,
        "best_mean_reward": best_mean_reward,
        "best_reward": best_reward,
        "std_reward": std_reward,
    }


