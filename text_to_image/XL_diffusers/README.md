# XL_diffusers: Basic Text-to-Image Inference

This folder provides a non-FKD SDXL pipeline for standard text-to-image generation.

## What this is

- Pipeline class: `OriginalStableDiffusionXL`
- No FKD resampling/steering logic in the denoising loop
- Reward utilities are still available via `get_reward_function` in `rewards.py`

## Quick start

Run from the `text_to_image` directory so local imports resolve cleanly.

```bash
cd text_to_image
python - <<'PY'
import torch
from diffusers import DDIMScheduler

from XL_diffusers.pipeline_sdxl import OriginalStableDiffusionXL

# 1. Load SDXL base model
pipe = OriginalStableDiffusionXL.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
)

# Optional: match the rest of this repo's common scheduler choice
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# 2. Move to device
if torch.cuda.is_available():
    pipe = pipe.to("cuda")
else:
    pipe = pipe.to("cpu")

# 3. Generate image(s)
prompt = "a cinematic photo of a snow-covered mountain village at sunrise"
images = pipe(
    prompt=prompt,
    num_inference_steps=50,
    guidance_scale=5.0,
    num_images_per_prompt=1,
)[0]

# 4. Save result
images[0].save("sample_sdxl.png")
print("Saved sample_sdxl.png")
PY
```

## Optional: score generated images with reward utilities

```python
from XL_diffusers.rewards import get_reward_function

scores = get_reward_function(
    reward_name="ImageReward",
    images=images,
    prompts=[prompt] * len(images),
)
print(scores)
```

## Notes

- Default SDXL guidance scale in this pipeline is `5.0`.
- If you use CPU, generation will be much slower.
- Some reward backends require extra credentials or model downloads.
