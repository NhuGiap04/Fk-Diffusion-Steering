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
- `split_samples`: runs one sampling pass, evaluates rewards, and splits accepted/rejected particles.
- `iterative_sample_with_stein`: repeats sampling loops and reuses accepted particles for steering.

### `steer_sample` behavior

- Input conditioning is prompt-based (`prompt` / `prompt_2`), same as `BaseSDXL.__call__`.
- Default classifier-free guidance is `guidance_scale=5.0`.
- Steering is injected via `callback_on_step_end` inside the base SDXL denoising loop.
- Return value is `(pipeline_output, latent_trajectory)`.
- `accepted_x0` must match latent tensor shape `[M, C, H_latent, W_latent]` for your chosen resolution.
- If `accepted_x0` is provided and `steer_start_timestep` is omitted, steering runs from the start of the scheduler horizon.

### Expected SDXL interface

`steer_sample` expects a `BaseSDXL` model instance and follows the same denoising path as `BaseSDXL.__call__`.
Steering is injected through `callback_on_step_end` so updates are applied inside the base pipeline loop.

### Minimal usage

```python
import torch
from diffusers import DDIMScheduler
from evolve_diffusers import BaseSDXL
from evolve_diffusers.steer_pipeline import steer_sample

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = BaseSDXL.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

# accepted clean latent particles from a previous loop: [M, 4, H/8, W/8]
accepted_x0 = torch.randn(4, 4, 128, 128, device=device)

result, latent_trajectory = steer_sample(
    model=pipe,
    prompt=["a photo of a blue clock and a white cup"] * 4,
    accepted_x0=accepted_x0,
    steer_start_timestep=160,
    steer_end_timestep=20,
    stein_step_size=0.04,
    stein_bandwidth=None,
    # guidance_scale defaults to 5.0, matching BaseSDXL default CFG.
    num_inference_steps=50,
    output_type="pil",
)

# `result.images` contains final generated images.
# `latent_trajectory` stores one latent tensor per denoising step on CPU.
print(len(result.images), len(latent_trajectory), latent_trajectory[-1].shape)
```

### Iterative loop usage

```python
from evolve_diffusers.steer_pipeline import iterative_sample_with_stein

loop_out = iterative_sample_with_stein(
    model=pipe,
    prompt="a photo of a blue clock and a white cup",
    num_loops=4,
    num_particles=4,
    steer_start_timestep=160,
    steer_end_timestep=20,
    base_threshold=0.0,
    stein_step_size=0.04,
    guidance_scale=5.0,
    guidance_reward_fn="ImageReward",
    num_inference_steps=50,
)

print(loop_out["best_mean_reward"], len(loop_out["trajectories"]))
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
