# Evolve Diffusers

This package provides baseline SDXL sampling and shared scoring utilities.

- `BaseSDXL`: baseline SDXL sampling (no steering logic).
- `steer_pipeline.py`: Stein-style trajectory steering utilities for diffusion samplers.

## What changed in the refactor

- Pipeline usage is class-based. Use `BaseSDXL` for regular generation.
- Shared utilities are exported from the package:
  - `latent_to_decode`
  - `get_reward_function`

## Package exports

From `evolve_diffusers.__init__`:

- `BaseSDXL`
- `latent_to_decode`
- `get_reward_function`

## Stein steering with `steer_pipeline.py`

`steer_pipeline.py` contains a standalone steering path for particle trajectories:

- `score_log_prob_reward`: approximates score from accepted clean samples.
- `stein_variational_vector_field`: computes empirical SVGD update field.
- `stein_step`: applies one Stein update.
- `steer_sample`: runs reverse diffusion and applies Stein updates in a timestep window.

### Expected DDPM interface

`steer_sample` expects a `ddpm` object with:

- `num_steps: int`
- `alpha_bars: torch.Tensor` (length `num_steps`)
- `p_sample(x_t, t, labels, guidance_scale=...) -> x_{t-1}`

### Minimal usage

```python
import torch
from evolve_diffusers.steer_pipeline import steer_sample

device = "cuda" if torch.cuda.is_available() else "cpu"

# labels shape: [num_samples]
labels = torch.full((32,), 2, dtype=torch.long, device=device)

# accepted clean samples from a previous loop: [M, D]
accepted_x0 = torch.randn(128, 2, device=device)

trajectory = steer_sample(
    ddpm=ddpm,
    labels=labels,
    accepted_x0=accepted_x0,
    steer_start_timestep=160,
    steer_end_timestep=20,
    stein_step_size=0.04,
    stein_bandwidth=None,
    guidance_scale=0.0,
    latent_dim=2,
    device=device,
)

# trajectory is [x_T, x_{T-1}, ..., x_0], each tensor on CPU.
print(len(trajectory), trajectory[-1].shape)
```

## Quick start (original pipeline)

Run from `text_to_image/` so local imports resolve as expected.

```bash
cd text_to_image
python - <<'PY'
import torch
from diffusers import DDIMScheduler

from evolve_diffusers import BaseSDXL

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = BaseSDXL.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

prompt = "a cinematic photo of a snow-covered mountain village at sunrise"

result = pipe(
    prompt=prompt,
    num_inference_steps=50,
    guidance_scale=5.0,
    num_images_per_prompt=1,
)

image = result.images[0]
image.save("sample_sdxl_original.png")
print("Saved sample_sdxl_original.png")
PY
```

## Scoring generated images directly

```python
from evolve_diffusers import get_reward_function

scores = get_reward_function(
    reward_name="ImageReward",
    images=result.images,
    prompts=prompts,
)
print(scores)
```

## Notes

- Default `guidance_scale` in these SDXL pipelines is `5.0`.
- GPU is strongly recommended; several reward models are expensive on CPU.
- Reward backends may require additional model downloads and credentials (for example API-backed graders).
