# XL_diffusers: SDXL Inference (Original + Incremental)

This folder provides an SDXL pipeline class with two modes:

1. Original sampling (default)
2. Incremental sampling with reward-weighted latent updates

Pipeline class: OriginalStableDiffusionXL

## Modes

### Original mode (default)

- Set use_incremental=False (or omit it)
- Runs one standard denoising trajectory

### Incremental mode

- Set use_incremental=True
- Repeats sampling for t_loops outer iterations
- In each loop, samples iterative_num_particles particles
- Captures latents at iterative_source_timestep_idx
- Computes rewards from final decoded images
- Updates next loop source using reward-weighted latent mean

## Quick start

Run from the text_to_image directory so local imports resolve cleanly.

~~~bash
cd text_to_image
python - <<'PY'
import torch
from diffusers import DDIMScheduler

from XL_diffusers.pipeline_sdxl import OriginalStableDiffusionXL

pipe = OriginalStableDiffusionXL.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

prompt = "a cinematic photo of a snow-covered mountain village at sunrise"

# Original mode
images = pipe(
    prompt=prompt,
    use_incremental=False,
    num_inference_steps=50,
    guidance_scale=5.0,
    num_images_per_prompt=1,
)[0]

images[0].save("sample_sdxl_original.png")
print("Saved sample_sdxl_original.png")
PY
~~~

## Incremental mode example

Important: iterative_num_particles must match the effective batch size for the call.
The simplest setup is prompt list length equal to iterative_num_particles.

~~~bash
cd text_to_image
python - <<'PY'
import torch
from diffusers import DDIMScheduler

from XL_diffusers.pipeline_sdxl import OriginalStableDiffusionXL

pipe = OriginalStableDiffusionXL.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

n_particles = 4
prompts = ["a photo of a blue clock and a white cup"] * n_particles

images = pipe(
    prompt=prompts,
    use_incremental=True,
    t_loops=3,
    iterative_num_particles=n_particles,
    iterative_source_timestep_idx=0,
    iterative_reward_fn="ImageReward",
    iterative_reward_temperature=1.0,
    iterative_source_noise_scale=1.0,
    iterative_metric_to_chase="overall_score",
    num_inference_steps=50,
    guidance_scale=5.0,
    num_images_per_prompt=1,
)[0]

for i, img in enumerate(images):
    img.save(f"sample_sdxl_incremental_{i:02d}.png")

print("Saved incremental samples")
PY
~~~

## Incremental arguments

- use_incremental: bool switch for mode selection
- t_loops: number of outer loops
- iterative_num_particles: number of particles per loop (must match effective batch)
- iterative_source_timestep_idx: latent capture timestep used for source update
- iterative_reward_fn: reward backend name (for example ImageReward, Clip-Score, HumanPreference, LLMGrader)
- iterative_reward_temperature: softmax temperature multiplier for reward weights
- iterative_source_noise_scale: noise scale when sampling from updated source latent
- iterative_metric_to_chase: metric key used by LLMGrader

## Optional: score generated images directly

~~~python
from XL_diffusers.rewards import get_reward_function

scores = get_reward_function(
    reward_name="ImageReward",
    images=images,
    prompts=prompts,
)
print(scores)
~~~

## Notes

- Default guidance_scale in this pipeline is 5.0.
- CPU inference is much slower than GPU.
- Some reward backends require extra credentials or model downloads.
