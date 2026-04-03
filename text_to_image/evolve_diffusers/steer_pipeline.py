import inspect
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from diffusers.image_processor import PipelineImageInput
from PIL import Image, ImageDraw, ImageFont

from .pipeline_sdxl import BaseSDXL, get_scheduler_sigmas_for_timesteps, latent_to_decode
from .rewards import get_reward_function


def stein_variational_vector_field(
    x_t: torch.Tensor,
    support_x0: torch.Tensor,
    sigma_t: Union[float, torch.Tensor],
    temperature: float = 1.0,
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

    temperature_tensor = torch.as_tensor(temperature, device=device, dtype=dtype)
    if torch.any(temperature_tensor <= 0):
        raise ValueError("temperature must be > 0")

    sigma_t_tensor: torch.Tensor = torch.as_tensor(sigma_t, device=device, dtype=dtype)
    alpha_bar_t = 1.0 / torch.clamp(1.0 + sigma_t_tensor * sigma_t_tensor, min=eps)
    sqrt_alpha_bar_t = torch.sqrt(torch.clamp(alpha_bar_t, min=eps))
    var_t = torch.clamp(1.0 - alpha_bar_t, min=eps)

    means = sqrt_alpha_bar_t * support_x0
    delta = x_t[:, None, :] - means[None, :, :]
    sq_dist = (delta**2).sum(dim=-1)

    log_w = -0.5 * sq_dist / (temperature_tensor * var_t)
    w = torch.softmax(log_w, dim=1)

    component_scores = -delta / var_t
    return (w[:, :, None] * component_scores).sum(dim=1)


@torch.no_grad()
def stein_step(
    x_t: torch.Tensor,
    accepted_x0: torch.Tensor,
    sigma_t: Union[float, torch.Tensor],
    step_size: float = 0.05,
    step_index: int = 0,
    weight_temperature: float = 1.0,
    langevin_lambda: float = 1.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One Langevin corrector step using accepted-only Stein score estimates."""

    if x_t.ndim < 2 or accepted_x0.ndim < 2:
        raise ValueError("x_t and accepted_x0 must have at least 2 dims [B, ...]")
    if x_t.shape[1:] != accepted_x0.shape[1:]:
        raise ValueError(
            "x_t and accepted_x0 must have matching non-batch dimensions "
            f"(got {x_t.shape[1:]} vs {accepted_x0.shape[1:]})"
        )
    if step_size < 0:
        raise ValueError("step_size must be >= 0")
    if langevin_lambda < 0:
        raise ValueError("langevin_lambda must be >= 0")
    if step_index < 0:
        raise ValueError("step_index must be >= 0")

    x_shape = x_t.shape
    out_dtype = x_t.dtype
    x_flat = x_t.reshape(x_t.shape[0], -1).to(dtype=torch.float32)
    accepted_flat = accepted_x0.reshape(accepted_x0.shape[0], -1).to(dtype=torch.float32)

    phi = stein_variational_vector_field(
        x_t=x_flat,
        support_x0=accepted_flat,
        sigma_t=sigma_t,
        temperature=weight_temperature,
    )

    sigma_t_tensor = torch.as_tensor(sigma_t, device=x_flat.device, dtype=x_flat.dtype)
    alpha_bar_t = 1.0 / torch.clamp(1.0 + sigma_t_tensor * sigma_t_tensor, min=eps)
    one_minus_alpha_bar_t = torch.clamp(1.0 - alpha_bar_t, min=eps)
    eta_t = torch.as_tensor(float(step_size), device=x_flat.device, dtype=x_flat.dtype) * one_minus_alpha_bar_t

    omega = _randn_like_with_generator(x_flat, generator)
    noise_scale = torch.sqrt(torch.clamp(2.0 * eta_t * float(langevin_lambda), min=0.0))
    x_next = x_flat + eta_t / (step_index + 1) * phi + noise_scale * omega

    return x_next.to(dtype=out_dtype).reshape(x_shape), phi.to(dtype=out_dtype).reshape(x_shape)


@torch.no_grad()
def steer_sample(
    model: BaseSDXL,
    prompt: Union[str, List[str]],
    *,
    accepted_x0: Optional[torch.Tensor] = None,
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
    stein_num_steps: int = 1,
    stein_weight_temperature: float = 1.0,
    stein_langevin_lambda: float = 1.0,
    callback_on_step_end: Optional[Any] = None,
    callback_on_step_end_tensor_inputs: Optional[List[str]] = None,
    **kwargs: Any,
) -> Tuple[Any, List[torch.Tensor], List[int]]:
    """
    Prompt-conditioned SDXL sampling with optional Stein updates in the denoising loop.

    This follows `BaseSDXL` sampling flow and injects Stein steps through
    `callback_on_step_end`, rather than using a toy DDPM `p_sample` API.

    Returns:
        Tuple `(pipeline_output, latent_trajectory, trajectory_timesteps)`.
        `latent_trajectory` stores one latent tensor per denoising step (CPU).
        `trajectory_timesteps` stores the matching scheduler timestep per step.
    """
    if stein_num_steps <= 0:
        raise ValueError("stein_num_steps must be > 0")
    if stein_weight_temperature <= 0:
        raise ValueError("stein_weight_temperature must be > 0")
    if stein_langevin_lambda < 0:
        raise ValueError("stein_langevin_lambda must be >= 0")

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
    trajectory_timesteps: List[int] = []
    stein_update_idx = 0

    def _combined_step_callback(pipe, step_idx: int, t, callback_kwargs: Dict[str, Any]):
        nonlocal stein_update_idx
        callback_updates: Dict[str, Any] = {}
        current_latents = callback_kwargs["latents"]
        timestep_value = int(t.item()) if isinstance(t, torch.Tensor) else int(t)

        if use_stein:
            assert accepted_x0 is not None
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
                for _ in range(stein_num_steps):
                    current_latents, _ = stein_step(
                        x_t=current_latents,
                        accepted_x0=accepted_particles,
                        sigma_t=sigma_t,
                        step_size=stein_step_size,
                        step_index=stein_update_idx,
                        weight_temperature=stein_weight_temperature,
                        langevin_lambda=stein_langevin_lambda,
                        generator=generator,
                    )
                    stein_update_idx += 1

        if callback_on_step_end is not None:
            user_kwargs = dict(callback_kwargs)
            user_kwargs["latents"] = current_latents
            user_updates = callback_on_step_end(pipe, step_idx, t, user_kwargs)
            if user_updates is not None:
                callback_updates.update(user_updates)
                current_latents = callback_updates.pop("latents", current_latents)

        latent_trajectory.append(current_latents.detach().cpu())
        trajectory_timesteps.append(timestep_value)
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

    return result, latent_trajectory, trajectory_timesteps


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


def _get_restart_timesteps(
    model: BaseSDXL,
    *,
    num_inference_steps: int,
    steer_start_timestep: int,
    steer_end_timestep: int,
    device: torch.device,
) -> List[int]:
    """Build a scheduler-consistent denoising tail between steer start and end."""
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    schedule = [int(t.item()) if isinstance(t, torch.Tensor) else int(t) for t in model.scheduler.timesteps]

    t_hi = max(steer_start_timestep, steer_end_timestep)
    t_lo = min(steer_start_timestep, steer_end_timestep)
    selected = [t for t in schedule if t_lo <= t <= t_hi]

    if selected:
        return selected

    # Fallback to nearest available scheduler step and continue to the denoising end.
    nearest_idx = min(range(len(schedule)), key=lambda i: abs(schedule[i] - steer_start_timestep))
    return [t for t in schedule[nearest_idx:] if t >= t_lo]


def _scheduler_supports_custom_timesteps(model: BaseSDXL) -> bool:
    """Return whether scheduler.set_timesteps supports a custom `timesteps` argument."""
    return "timesteps" in set(inspect.signature(model.scheduler.set_timesteps).parameters.keys())


@torch.no_grad()
def _decode_latents_to_pil_images(
    model: BaseSDXL,
    latents: torch.Tensor,
) -> List[Any]:
    """Decode latent batches to PIL images using the pipeline VAE path."""
    if latents.ndim != 4:
        raise ValueError(f"Expected latents with shape [B, C, H, W], got {tuple(latents.shape)}")
    if latents.shape[0] == 0:
        return []

    execution_device = next(model.unet.parameters()).device
    latents_on_device = latents.to(device=execution_device)
    decoded = latent_to_decode(model=model, output_type="pt", latents=latents_on_device)
    return list(model.image_processor.postprocess(decoded, output_type="pil"))


def _make_image_grid(images: List[Any]) -> Any:
    """Tile images into a compact grid preserving particle order."""
    if len(images) == 0:
        raise ValueError("Cannot build a grid from an empty image list")

    base_images = [img.convert("RGB") for img in images]
    cell_width, cell_height = base_images[0].size
    ncols = min(4, len(base_images))
    nrows = math.ceil(len(base_images) / ncols)

    grid = Image.new("RGB", (ncols * cell_width, nrows * cell_height))
    for image_idx, image in enumerate(base_images):
        if image.size != (cell_width, cell_height):
            image = image.resize((cell_width, cell_height))
        x_offset = (image_idx % ncols) * cell_width
        y_offset = (image_idx // ncols) * cell_height
        grid.paste(image, (x_offset, y_offset))

    return grid


def _add_title_to_image(image: Any, title: str) -> Any:
    """Add a title strip above an image for easier visual comparison."""
    base_image = image.convert("RGB")
    title_height = 24
    titled = Image.new("RGB", (base_image.width, base_image.height + title_height), color=(24, 24, 24))
    titled.paste(base_image, (0, title_height))

    draw = ImageDraw.Draw(titled)
    draw.text((8, 6), title, fill=(240, 240, 240), font=ImageFont.load_default())
    return titled


def _select_timestep_indices(
    trajectory_timesteps: List[int],
    *,
    timestep_min: int,
    timestep_max: int,
    timestep_stride: int,
) -> List[int]:
    """Select denoising step indices that fall within a timestep window."""
    if timestep_stride <= 0:
        raise ValueError("timestep_stride must be > 0")

    in_range_indices = [
        idx
        for idx, timestep in enumerate(trajectory_timesteps)
        if timestep_min <= int(timestep) <= timestep_max
    ]
    if len(in_range_indices) == 0:
        return []

    selected: List[int] = []
    for in_range_pos, step_idx in enumerate(in_range_indices):
        is_last = in_range_pos == len(in_range_indices) - 1
        if (in_range_pos % timestep_stride) != 0 and not is_last:
            continue
        selected.append(step_idx)
    return selected


@torch.no_grad()
def _save_timestep_views_for_loop(
    model: BaseSDXL,
    *,
    latent_trajectory: List[torch.Tensor],
    trajectory_timesteps: List[int],
    loop_output_dir: Path,
    by_timestep_output_dir: Path,
    timestep_stride: int,
    timestep_min: int,
    timestep_max: int,
    static_images: Optional[List[Any]] = None,
    loop_name: str,
) -> Dict[str, Any]:
    """Save titled timestep views for one loop in by-loop and by-timestep layouts."""
    if len(latent_trajectory) != len(trajectory_timesteps):
        raise ValueError(
            "latent_trajectory and trajectory_timesteps must have the same length "
            f"(got {len(latent_trajectory)} vs {len(trajectory_timesteps)})"
        )

    if timestep_min > timestep_max:
        raise ValueError(f"Invalid timestep window [{timestep_min}, {timestep_max}]")

    loop_output_dir.mkdir(parents=True, exist_ok=True)
    by_timestep_output_dir.mkdir(parents=True, exist_ok=True)
    static_images = list(static_images or [])
    saved_step_indices: List[int] = []
    saved_timestep_values: List[int] = []

    total_timesteps = len(latent_trajectory)
    dynamic_particles = int(latent_trajectory[0].shape[0]) if total_timesteps > 0 else 0
    selected_step_indices = _select_timestep_indices(
        trajectory_timesteps,
        timestep_min=timestep_min,
        timestep_max=timestep_max,
        timestep_stride=timestep_stride,
    )

    for frame_idx, step_idx in enumerate(selected_step_indices):
        step_latents = latent_trajectory[step_idx]
        timestep_value = int(trajectory_timesteps[step_idx])
        dynamic_images = _decode_latents_to_pil_images(model=model, latents=step_latents)
        frame_images = [*static_images, *dynamic_images]
        if len(frame_images) == 0:
            continue

        frame_title = f"{loop_name} | timestep={timestep_value} | frame={frame_idx:03d}"
        grid_image = _add_title_to_image(_make_image_grid(frame_images), frame_title)

        loop_filename = f"frame_{frame_idx:05d}_t{timestep_value:04d}.png"
        grid_image.save(loop_output_dir / loop_filename)

        timestep_dir = by_timestep_output_dir / f"timestep_{timestep_value:04d}"
        timestep_dir.mkdir(parents=True, exist_ok=True)
        grid_image.save(timestep_dir / f"{loop_name}.png")

        saved_step_indices.append(step_idx)
        saved_timestep_values.append(timestep_value)

    manifest = {
        "loop_name": loop_name,
        "total_timesteps": int(total_timesteps),
        "timestep_stride": int(timestep_stride),
        "timestep_range": [int(timestep_min), int(timestep_max)],
        "saved_frames": int(len(saved_step_indices)),
        "saved_step_indices": [int(i) for i in saved_step_indices],
        "saved_timestep_values": [int(t) for t in saved_timestep_values],
        "static_particles": int(len(static_images)),
        "dynamic_particles": int(dynamic_particles),
        "full_pool_particles": int(len(static_images) + dynamic_particles),
        "loop_output_dir": str(loop_output_dir),
    }
    with open(loop_output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def _randn_like_with_generator(
    x: torch.Tensor,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]],
) -> torch.Tensor:
    """Generate Gaussian noise matching x with optional per-sample generators."""
    if isinstance(generator, list):
        if len(generator) != x.shape[0]:
            raise ValueError(
                "When generator is a list, its length must match batch size "
                f"(got {len(generator)} vs {x.shape[0]})"
            )
        parts = [
            torch.randn(
                (1, *x.shape[1:]),
                device=x.device,
                dtype=x.dtype,
                generator=generator[i],
            )
            for i in range(x.shape[0])
        ]
        return torch.cat(parts, dim=0)

    # Some torch versions do not support `generator=` in randn_like.
    # Use randn with explicit shape/device/dtype for broad compatibility.
    return torch.randn(
        x.shape,
        device=x.device,
        dtype=x.dtype,
        generator=generator,
    )


def _renoise_x0_to_timestep(
    model: BaseSDXL,
    x0_latents: torch.Tensor,
    *,
    timestep: int,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Forward diffuse x0 latents to x_t at the requested scheduler timestep."""
    # Avoid scheduler.set_timesteps(timesteps=...), which is unsupported by DDIMScheduler.
    # get_scheduler_sigmas_for_timesteps can recover sigma from scheduler internals
    # (e.g., alphas_cumprod) for direct training timestep indices.
    sigma_t = get_scheduler_sigmas_for_timesteps(
        scheduler=model.scheduler,
        timesteps=[int(timestep)],
        device=x0_latents.device,
    )[0].to(device=x0_latents.device, dtype=x0_latents.dtype)

    alpha_bar_t = 1.0 / torch.clamp(1.0 + sigma_t * sigma_t, min=eps)
    sqrt_alpha_bar_t = torch.sqrt(torch.clamp(alpha_bar_t, min=eps))
    sqrt_one_minus_alpha_bar_t = torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=eps))

    noise = _randn_like_with_generator(x0_latents, generator)
    return sqrt_alpha_bar_t * x0_latents + sqrt_one_minus_alpha_bar_t * noise


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
    stein_num_steps: int = 1,
    stein_weight_temperature: float = 1.0,
    stein_langevin_lambda: float = 1.0,
    accepted_x0: Optional[torch.Tensor] = None,
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

    result, latent_trajectory, trajectory_timesteps = steer_sample(
        model=model,
        prompt=generation_prompt,
        accepted_x0=accepted_x0,
        steer_start_timestep=steer_start_timestep,
        steer_end_timestep=steer_end_timestep,
        stein_step_size=stein_step_size,
        stein_num_steps=stein_num_steps,
        stein_weight_temperature=stein_weight_temperature,
        stein_langevin_lambda=stein_langevin_lambda,
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
        "trajectory_timesteps": trajectory_timesteps,
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
    stein_step_size: float = 0.05,
    stein_num_steps: int = 1,
    stein_weight_temperature: float = 1.0,
    stein_langevin_lambda: float = 1.0,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    guidance_reward_fn: str = "ImageReward",
    metric_to_chase: Optional[str] = None,
    early_stop_epsilon: Optional[float] = None,
    save_timestep_grids: bool = False,
    timestep_grid_stride: int = 5,
    timestep_grid_output_dir: Optional[Union[str, Path]] = None,
    final_denoise_timestep: Optional[int] = None,
    save_warmup_timestep_grid: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Iteratively self-improve particle pools with warmup, steering, and replacement.

    Behavior:
    1) Warmup: collect `num_particles` samples and split at warmup mean reward.
    2) Loop N times:
       - steer rejected particles from `steer_start_timestep` to `steer_end_timestep`
         using the accepted pool,
       - evaluate rewards,
                        - resample the full particle pool via multinomial selection weighted
                                by final rewards,
       - update threshold and refresh accepted/rejected pools for the next loop.
    3) Optional early stop if reward improvement is below epsilon for 2 consecutive transitions.
    """
    if num_loops <= 0:
        raise ValueError("num_loops must be > 0")
    if early_stop_epsilon is not None and early_stop_epsilon < 0:
        raise ValueError("early_stop_epsilon must be >= 0 when provided")

    all_results: List[Any] = []
    all_trajectories: List[List[torch.Tensor]] = []
    all_rewards: List[torch.Tensor] = []
    all_thresholds: List[float] = []
    all_accepted: List[torch.Tensor] = []
    all_rejected: List[torch.Tensor] = []

    best_mean_reward = -float("inf")
    running_threshold = float("nan")
    best_reward = float("-inf")
    std_reward = 0.0
    min_reward = float("inf")
    accepted_pool: Optional[torch.Tensor] = None
    accepted_pool_rewards: Optional[torch.Tensor] = None
    rejected_pool: Optional[torch.Tensor] = None
    accepted_images_pool: List[Any] = []
    rejected_images_pool: List[Any] = []

    previous_loop_mean_reward: Optional[float] = None
    consecutive_small_improvement = 0
    early_stopped = False
    early_stop_reason: Optional[str] = None
    timestep_grid_manifests: List[Dict[str, Any]] = []

    grid_output_base: Optional[Path] = None
    loop_grid_output_dir: Optional[Path] = None
    timestep_grid_by_value_dir: Optional[Path] = None
    display_timestep_min: Optional[int] = None
    display_timestep_max: Optional[int] = None
    if save_timestep_grids:
        if timestep_grid_stride <= 0:
            raise ValueError("timestep_grid_stride must be > 0 when saving timestep grids")
        if timestep_grid_output_dir is None:
            raise ValueError("timestep_grid_output_dir must be provided when saving timestep grids")
        grid_output_base = Path(timestep_grid_output_dir)
        grid_output_base.mkdir(parents=True, exist_ok=True)
        loop_grid_output_dir = grid_output_base / "denoising_by_loop"
        timestep_grid_by_value_dir = grid_output_base / "same_timestep_across_loops"
        loop_grid_output_dir.mkdir(parents=True, exist_ok=True)
        timestep_grid_by_value_dir.mkdir(parents=True, exist_ok=True)

        effective_final_denoise_timestep = (
            int(final_denoise_timestep)
            if final_denoise_timestep is not None
            else 0
        )
        display_timestep_min = min(int(steer_start_timestep), effective_final_denoise_timestep)
        display_timestep_max = max(int(steer_start_timestep), effective_final_denoise_timestep)

    # Warmup: sample K particles and build the initial support using warmup mean reward.
    generation_prompt, num_images_per_prompt, reward_prompts = _resolve_prompt_and_particles(
        prompt=prompt,
        num_particles=num_particles,
    )
    warmup_result, warmup_trajectory, warmup_timesteps = steer_sample(
        model=model,
        prompt=generation_prompt,
        num_images_per_prompt=num_images_per_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        output_type="pil",
        return_dict=True,
        **kwargs,
    )
    warmup_images = list(warmup_result.images)
    warmup_rewards = torch.as_tensor(
        get_reward_function(
            reward_name=guidance_reward_fn,
            images=warmup_images,
            prompts=reward_prompts,
            metric_to_chase=metric_to_chase or "overall_score",
        ),
        dtype=torch.float32,
    )
    warmup_final_latents = warmup_trajectory[-1]
    warmup_threshold = float(warmup_rewards.mean().item())
    running_threshold = warmup_threshold
    warmup_accept_mask = warmup_rewards >= warmup_threshold
    if not bool(warmup_accept_mask.any()):
        # Keep at least one support particle so steering can proceed.
        best_idx = int(torch.argmax(warmup_rewards).item())
        warmup_accept_mask[best_idx] = True
    warmup_reject_mask = ~warmup_accept_mask

    accepted_pool = warmup_final_latents[warmup_accept_mask]
    rejected_pool = warmup_final_latents[warmup_reject_mask]
    accepted_pool_rewards = warmup_rewards[warmup_accept_mask].to(dtype=torch.float32).cpu()
    accepted_images_pool = [img for i, img in enumerate(warmup_images) if bool(warmup_accept_mask[i].item())]
    rejected_images_pool = [img for i, img in enumerate(warmup_images) if bool(warmup_reject_mask[i].item())]
    warmup_accepted_count = int(warmup_accept_mask.sum().item())
    warmup_rejected_count = int(warmup_reject_mask.sum().item())

    # Warmup initializes pools only; resampling starts after loop 1 completes.
    warmup_best_reward = float(warmup_rewards.max().item())
    warmup_mean_reward = float(warmup_rewards.mean().item())
    warmup_std_reward = float(warmup_rewards.std().item())
    warmup_min_reward = float(warmup_rewards.min().item())
    warmup_stats: Dict[str, Any] = {
        "best_reward": warmup_best_reward,
        "mean_reward": warmup_mean_reward,
        "std_reward": warmup_std_reward,
        "min_reward": warmup_min_reward,
        "threshold": float(warmup_threshold),
        "accepted": warmup_accepted_count,
        "rejected": warmup_rejected_count,
        "num_particles": int(warmup_rewards.shape[0]),
    }
    print(
        f"warmup "
        f"best_reward={warmup_best_reward:.4f} "
        f"mean_reward={warmup_mean_reward:.4f} "
        f"std_reward={warmup_std_reward:.4f} "
        f"min_reward={warmup_min_reward:.4f} "
        f"threshold={warmup_threshold:.4f} "
        f"accepted={warmup_accepted_count} "
        f"rejected={warmup_rejected_count}"
    )

    if (
        save_timestep_grids
        and save_warmup_timestep_grid
        and loop_grid_output_dir is not None
        and timestep_grid_by_value_dir is not None
        and display_timestep_min is not None
        and display_timestep_max is not None
    ):
        warmup_manifest = _save_timestep_views_for_loop(
            model=model,
            latent_trajectory=warmup_trajectory,
            trajectory_timesteps=warmup_timesteps,
            loop_output_dir=loop_grid_output_dir / "loop_000",
            by_timestep_output_dir=timestep_grid_by_value_dir,
            timestep_stride=timestep_grid_stride,
            timestep_min=display_timestep_min,
            timestep_max=display_timestep_max,
            static_images=None,
            loop_name="loop_000",
        )
        timestep_grid_manifests.append(warmup_manifest)

    for loop_idx in range(num_loops):
        if accepted_pool is None or accepted_pool.shape[0] == 0:
            early_stopped = True
            early_stop_reason = "No accepted particles available for steering."
            break
        if rejected_pool is None or rejected_pool.shape[0] == 0:
            early_stopped = True
            early_stop_reason = "No rejected particles left to improve."
            break

        supports_custom_timesteps = _scheduler_supports_custom_timesteps(model)
        restart_timesteps: Optional[List[int]] = None
        if supports_custom_timesteps:
            restart_timesteps = _get_restart_timesteps(
                model=model,
                num_inference_steps=num_inference_steps,
                steer_start_timestep=steer_start_timestep,
                steer_end_timestep=steer_end_timestep,
                device=rejected_pool.device,
            )
            if len(restart_timesteps) == 0:
                raise ValueError(
                    "Unable to build restart timesteps for requested steer range "
                    f"[{steer_start_timestep}, {steer_end_timestep}]"
                )
            restart_noise_timestep = int(restart_timesteps[0])
            restart_num_inference_steps = len(restart_timesteps)
        else:
            # Compatibility fallback for schedulers that reject custom timestep schedules.
            model.scheduler.set_timesteps(num_inference_steps, device=rejected_pool.device)
            schedule = [
                int(t.item()) if isinstance(t, torch.Tensor) else int(t)
                for t in model.scheduler.timesteps
            ]
            if len(schedule) == 0:
                raise ValueError("Scheduler produced empty timesteps schedule")
            restart_noise_timestep = int(schedule[0])
            restart_num_inference_steps = int(num_inference_steps)

        generation_prompt, num_images_per_prompt, reward_prompts = _resolve_prompt_and_particles(
            prompt=prompt,
            num_particles=int(rejected_pool.shape[0]),
        )

        restart_latents = _renoise_x0_to_timestep(
            model=model,
            x0_latents=rejected_pool,
            timestep=restart_noise_timestep,
            generator=kwargs.get("generator", None),
        )

        steer_sample_kwargs: Dict[str, Any] = {
            "model": model,
            "prompt": generation_prompt,
            "accepted_x0": accepted_pool,
            "steer_start_timestep": steer_start_timestep,
            "steer_end_timestep": steer_end_timestep,
            "stein_step_size": stein_step_size,
            "stein_num_steps": stein_num_steps,
            "stein_weight_temperature": stein_weight_temperature,
            "stein_langevin_lambda": stein_langevin_lambda,
            "num_images_per_prompt": num_images_per_prompt,
            "num_inference_steps": restart_num_inference_steps,
            "guidance_scale": guidance_scale,
            "latents": restart_latents,
            "output_type": "pil",
            "return_dict": True,
            **kwargs,
        }
        if restart_timesteps is not None:
            steer_sample_kwargs["timesteps"] = restart_timesteps

        result, latent_trajectory, trajectory_timesteps = steer_sample(**steer_sample_kwargs)

        improved_images = list(result.images)
        improved_rewards = torch.as_tensor(
            get_reward_function(
                reward_name=guidance_reward_fn,
                images=improved_images,
                prompts=reward_prompts,
                metric_to_chase=metric_to_chase or "overall_score",
            ),
            dtype=torch.float32,
        )
        improved_latents = latent_trajectory[-1]

        if (
            save_timestep_grids
            and loop_grid_output_dir is not None
            and timestep_grid_by_value_dir is not None
            and display_timestep_min is not None
            and display_timestep_max is not None
        ):
            loop_folder = f"loop_{loop_idx + 1:03d}"
            loop_manifest = _save_timestep_views_for_loop(
                model=model,
                latent_trajectory=latent_trajectory,
                trajectory_timesteps=trajectory_timesteps,
                loop_output_dir=loop_grid_output_dir / loop_folder,
                by_timestep_output_dir=timestep_grid_by_value_dir,
                timestep_stride=timestep_grid_stride,
                timestep_min=display_timestep_min,
                timestep_max=display_timestep_max,
                static_images=accepted_images_pool,
                loop_name=loop_folder,
            )
            timestep_grid_manifests.append(loop_manifest)

        images = [*accepted_images_pool, *improved_images]
        rewards = torch.cat(
            [
                accepted_pool_rewards if accepted_pool_rewards is not None else torch.empty(0, dtype=torch.float32),
                improved_rewards,
            ],
            dim=0,
        )
        final_latents = torch.cat([accepted_pool, improved_latents], dim=0)

        current_particle_count = max(int(rewards.shape[0]), 1)
        best_reward = float(rewards.max().item())
        mean_reward = float(rewards.mean().item())
        std_reward = float(rewards.std().item())
        min_reward = float(rewards.min().item())

        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward

        running_threshold = best_mean_reward
        threshold = running_threshold

        accept_mask = rewards >= threshold
        if not bool(accept_mask.any()):
            best_idx = int(torch.argmax(rewards).item())
            accept_mask[best_idx] = True
        reject_mask = ~accept_mask

        accepted_latents = final_latents[accept_mask]
        rejected_latents = final_latents[reject_mask]
        accepted_rewards_for_sampling = rewards[accept_mask].to(dtype=torch.float32)
        accepted_rewards = accepted_rewards_for_sampling.cpu()

        accepted_images = [img for i, img in enumerate(images) if bool(accept_mask[i].item())]
        rejected_images = [img for i, img in enumerate(images) if bool(reject_mask[i].item())]

        # Resample the full particle pool according to final rewards.
        # This drives the next loop's particles using reward-weighted selection
        # over all current particles, not only the weak subset.
        resample_count = int(final_latents.shape[0])
        if resample_count > 0:
            sampling_probs = torch.softmax(
                rewards.to(device=final_latents.device, dtype=torch.float32),
                dim=0,
            )
            sampled_indices = torch.multinomial(
                sampling_probs,
                num_samples=resample_count,
                replacement=True,
            )
            rejected_latents = final_latents[sampled_indices].clone()
            sampled_indices_list = sampled_indices.detach().cpu().tolist()
            rejected_images = [images[int(i)] for i in sampled_indices_list]

        accepted_pool = accepted_latents
        accepted_pool_rewards = accepted_rewards
        rejected_pool = rejected_latents
        accepted_images_pool = accepted_images
        rejected_images_pool = rejected_images

        all_results.append(result)
        all_trajectories.append(latent_trajectory)
        all_rewards.append(rewards)
        all_thresholds.append(float(threshold))
        all_accepted.append(accepted_latents)
        all_rejected.append(rejected_latents)

        print(
            f"loop={loop_idx + 1}/{num_loops} "
            f"best_reward={best_reward:.4f} "
            f"mean_reward={mean_reward:.4f} "
            f"std_reward={std_reward:.4f} "
            f"min_reward={min_reward:.4f} "
            f"threshold={threshold:.4f} "
            f"accepted={accepted_latents.shape[0]} "
            f"rejected={rejected_latents.shape[0]}"
        )

        if early_stop_epsilon is not None:
            if previous_loop_mean_reward is not None:
                improvement = mean_reward - previous_loop_mean_reward
                if improvement < early_stop_epsilon:
                    consecutive_small_improvement += 1
                else:
                    consecutive_small_improvement = 0

                if consecutive_small_improvement >= 2:
                    early_stopped = True
                    early_stop_reason = (
                        "Mean reward improvement was below epsilon for two consecutive loop transitions."
                    )
                    break
            previous_loop_mean_reward = mean_reward

    return {
        "results": all_results,
        "trajectories": all_trajectories,
        "rewards": all_rewards,
        "warmup_stats": warmup_stats,
        "thresholds": all_thresholds,
        "accepted": all_accepted,
        "rejected": all_rejected,
        "accepted_pool": accepted_pool,
        "accepted_pool_rewards": accepted_pool_rewards,
        "rejected_pool": rejected_pool,
        "accepted_images_pool": accepted_images_pool,
        "rejected_images_pool": rejected_images_pool,
        "best_mean_reward": best_mean_reward,
        "best_reward": best_reward,
        "std_reward": std_reward,
        "min_reward": min_reward,
        "early_stopped": early_stopped,
        "early_stop_reason": early_stop_reason,
        "completed_loops": len(all_rewards),
        "timestep_grid_manifests": timestep_grid_manifests,
        "timestep_grid_output_dir": str(grid_output_base) if grid_output_base is not None else None,
    }


