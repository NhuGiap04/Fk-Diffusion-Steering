# evolve_diffusers

Refactored SDXL pipelines used in this repo.

This package provides baseline SDXL sampling and shared scoring utilities.

- `BaseSDXL`: baseline SDXL sampling (no incremental steering logic).

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
