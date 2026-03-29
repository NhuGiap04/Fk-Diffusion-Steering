from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import math

import torch
from diffusers.image_processor import PipelineImageInput
from .rewards import get_reward_function
from .pipeline_sdxl import (
    BaseSDXL,
    get_scheduler_sigmas_for_timesteps,
    latent_to_decode,
    rescale_noise_cfg,
    XLA_AVAILABLE,
)

if XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm


def score_log_prob_reward(
    x_t: torch.Tensor,
    accepted_x0: torch.Tensor,
    t: int,
    ddpm,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Approximate ``grad_{x_t} log p(x_t | good)`` from accepted clean samples."""
    if x_t.ndim != 2 or accepted_x0.ndim != 2:
        raise ValueError("x_t and accepted_x0 must have shape [B, D]")
    if x_t.shape[1] != accepted_x0.shape[1]:
        raise ValueError("x_t and accepted_x0 must have matching feature dimension")
    if accepted_x0.shape[0] == 0:
        raise ValueError("accepted_x0 is empty; cannot approximate score")

    if not isinstance(t, int):
        raise ValueError("t must be an int diffusion step")
    if t < 0 or t >= ddpm.num_steps:
        raise ValueError(f"t must be in [0, {ddpm.num_steps - 1}]")

    device = x_t.device
    dtype = x_t.dtype
    accepted_x0 = accepted_x0.to(device=device, dtype=dtype)

    alpha_bar_t = ddpm.alpha_bars[t].to(device=device, dtype=dtype)
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    var_t = (1.0 - alpha_bar_t).clamp_min(eps)

    means = sqrt_alpha_bar_t * accepted_x0
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
    t: int,
    ddpm,
    step_size: float = 0.05,
    bandwidth: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One SVGD update step on particles at timestep ``t``."""
    score = score_log_prob_reward(x_t=x_t, accepted_x0=accepted_x0, t=t, ddpm=ddpm)
    phi = stein_variational_vector_field(x=x_t, score=score, bandwidth=bandwidth)
    return x_t + step_size * phi, phi


@torch.no_grad()
def steer_sample(
    ddpm,
    labels: torch.Tensor,
    *,
    accepted_x0: Optional[torch.Tensor] = None,
    steer_start_timestep: Optional[int] = None,
    steer_end_timestep: int = 0,
    stein_step_size: float = 0.05,
    stein_bandwidth: Optional[float] = None,
    guidance_scale: float = 0.0,
    latent_dim: int = 2,
    device: Optional[Union[str, torch.device]] = None,
) -> List[torch.Tensor]:
    """
    Sample one reverse trajectory and optionally apply a Stein step at each
    timestep within ``[steer_end_timestep, steer_start_timestep]``.

    This mirrors the logic in ``2D_examples/steer.py``.

    Returns:
        A list ``[x_T, x_{T-1}, ..., x_0]`` with each tensor moved to CPU.
    """
    if labels.ndim != 1:
        raise ValueError("labels must have shape [num_samples]")

    if device is None:
        if accepted_x0 is not None:
            device = accepted_x0.device
        else:
            device = labels.device

    labels = labels.to(device=device)
    num_samples = labels.shape[0]
    x = torch.randn(num_samples, latent_dim, device=device)
    traj: List[torch.Tensor] = [x.detach().cpu()]

    use_stein = (
        accepted_x0 is not None
        and accepted_x0.shape[0] > 0
        and steer_start_timestep is not None
    )

    if use_stein:
        assert accepted_x0 is not None
        assert steer_start_timestep is not None
        accepted_x0 = accepted_x0.to(device=device, dtype=x.dtype)
        t_hi = max(steer_start_timestep, steer_end_timestep)
        t_lo = min(steer_start_timestep, steer_end_timestep)

    for t in reversed(range(ddpm.num_steps)):
        x = ddpm.p_sample(x, t, labels, guidance_scale=guidance_scale)

        if use_stein and t_lo <= t <= t_hi:
            x, _ = stein_step(
                x_t=x,
                accepted_x0=accepted_x0,
                t=t,
                ddpm=ddpm,
                step_size=stein_step_size,
                bandwidth=stein_bandwidth,
            )

        traj.append(x.detach().cpu())

    return traj


