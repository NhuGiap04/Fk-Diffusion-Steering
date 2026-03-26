# XL_diffusers

Refactored SDXL pipelines used in this repo.

This package now separates standard sampling and incremental steering into two classes:

- `OriginalStableDiffusionXL`: baseline SDXL sampling (no incremental steering logic).
- `IncrementStableDiffusionXL`: reward-driven iterative steering built on top of `OriginalStableDiffusionXL`.
- `TemperedDiverseRejuvenatedStableDiffusionXL`: training-free tempered multi-island FK steering with lookahead, diversity-aware resampling, rejuvenation, and late immigrants.

## What changed in the refactor

- Incremental behavior is no longer toggled via `use_incremental=True/False`.
- You now choose the pipeline class directly:
  - `OriginalStableDiffusionXL` for regular generation.
  - `IncrementStableDiffusionXL` for iterative reward-weighted updates.
- Shared utilities are exported from the package:
  - `latent_to_decode`
  - `get_reward_function`

## Package exports

From `XL_diffusers.__init__`:

- `OriginalStableDiffusionXL`
- `IncrementStableDiffusionXL`
- `TemperedDiverseRejuvenatedStableDiffusionXL`
- `latent_to_decode`
- `get_reward_function`

## Quick start (original pipeline)

Run from `text_to_image/` so local imports resolve as expected.

```bash
cd text_to_image
python - <<'PY'
import torch
from diffusers import DDIMScheduler

from XL_diffusers import OriginalStableDiffusionXL

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = OriginalStableDiffusionXL.from_pretrained(
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

## Quick start (incremental steering pipeline)

`IncrementStableDiffusionXL` adds iterative steering arguments on top of the normal SDXL call.

```bash
cd text_to_image
python - <<'PY'
import torch
from diffusers import DDIMScheduler

from XL_diffusers import IncrementStableDiffusionXL

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = IncrementStableDiffusionXL.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

n_particles = 4
prompts = ["a photo of a blue clock and a white cup"] * n_particles

result = pipe(
    prompt=prompts,
    num_inference_steps=50,
    guidance_scale=5.0,
    num_images_per_prompt=1,
    t_loops=3,
    iterative_num_particles=n_particles,
    iterative_source_timestep_idx=0,
    iterative_reward_fn="ImageReward",
    iterative_reward_temperature=1.0,
    iterative_source_noise_scale=1.0,
    iterative_metric_to_chase="overall_score",
)

for i, image in enumerate(result.images):
    image.save(f"sample_sdxl_incremental_{i:02d}.png")

print("Saved incremental samples")
PY
```

## Quick start (training-free TDR-FK pipeline)

`TemperedDiverseRejuvenatedStableDiffusionXL` implements a training-free algorithm that combines:

- Multiple lambda-temperature islands.
- Rollout lookahead mean/std reward scoring.
- Diversity-aware ancestor selection.
- Rejuvenation moves after resampling.
- Late immigrant particle injection.
- Final diverse subset selection from the hottest island.

```bash
cd text_to_image
python - <<'PY'
import torch
from diffusers import DDIMScheduler

from XL_diffusers import TemperedDiverseRejuvenatedStableDiffusionXL

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = TemperedDiverseRejuvenatedStableDiffusionXL.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

n_particles = 12
prompts = ["a photo of a blue clock and a white cup"] * n_particles

result = pipe(
  prompt=prompts,
  num_inference_steps=40,
  guidance_scale=5.0,
  num_images_per_prompt=1,
  tdr_num_islands=3,
  tdr_target_lambda=2.0,
  tdr_particles_per_island=[4, 4, 4],
  tdr_resample_frequency=5,
  tdr_ess_threshold_ratio=0.5,
  tdr_rollout_count=2,
  tdr_rollout_steps=2,
  tdr_rejuvenation_steps=1,
  tdr_rejuvenation_noise_scale=0.08,
  tdr_immigrant_fraction=0.15,
  tdr_final_select_m=4,
  tdr_final_diversity_beta=0.2,
  tdr_reward_fn="ImageReward",
  tdr_metric_to_chase="overall_score",
)

for i, image in enumerate(result.images):
  image.save(f"sample_sdxl_tdr_fk_{i:02d}.png")

print("Saved TDR-FK samples")
PY
```

## Incremental steering arguments

Arguments accepted by `IncrementStableDiffusionXL.__call__`:

- `t_loops` (`int`, default `1`): number of outer steering loops.
- `iterative_num_particles` (`Optional[int]`): number of particles; must match effective batch size.
- `iterative_source_timestep_idx` (`int`): source timestep index. Internally realigned to scheduler sigma.
- `iterative_reward_fn` (`str`): reward backend name. Supported options:
  - `ImageReward`
  - `Clip-Score`
  - `HumanPreference`
  - `LLMGrader`
- `iterative_reward_temperature` (`float`): softmax temperature multiplier over rewards.
- `iterative_source_noise_scale` (`float` in `[0, 1]`): controls source-latent mixing/noise level.
- `iterative_metric_to_chase` (`str`): metric key used by `LLMGrader`.

## TDR-FK steering arguments

Arguments accepted by `TemperedDiverseRejuvenatedStableDiffusionXL.__call__`:

- `tdr_num_islands` (`int`): number of temperature islands.
- `tdr_target_lambda` (`float`): highest-temperature island lambda.
- `tdr_island_lambdas` (`Optional[List[float]]`): explicit strictly increasing lambda list.
- `tdr_particles_per_island` (`Optional[List[int]]`): per-island particle counts; must sum to effective batch size.
- `tdr_resample_frequency` (`int`): resampling check interval in denoising step indices.
- `tdr_resampling_t_start`, `tdr_resampling_t_end` (`int`): active step-index window for resampling checks.
- `tdr_ess_threshold_ratio` (`float`): ESS trigger threshold as ratio of island particle count.
- `tdr_rollout_count`, `tdr_rollout_steps` (`int`): training-free lookahead rollout budget.
- `tdr_kappa_start/end`, `tdr_rho_start/end` (`float`): schedules for uncertainty bonus and crowding penalty.
- `tdr_crowding_bandwidth` (`float`): latent crowding kernel bandwidth.
- `tdr_rejuvenation_steps` (`int`) and `tdr_rejuvenation_noise_scale` (`float`): post-resample rejuvenation controls.
- `tdr_promotion_every` (`int`) and `tdr_promotion_count` (`int`): cross-island promotion cadence and count.
- `tdr_immigrant_fraction` (`float`): fraction of island particles replaced by late immigrants when triggered.
- `tdr_immigrant_islands` (`Optional[List[int]]`): island indices receiving immigrants.
- `tdr_immigrant_step_indices` (`Optional[List[int]]`): explicit coarse step indices for immigrant injection.
- `tdr_immigrant_coarse_steps` (`int`): number of coarse mini-steps for restart denoising.
- `tdr_final_select_m` (`int`) and `tdr_final_diversity_beta` (`float`): final diverse subset selection settings.
- `tdr_reward_fn` (`str`) and `tdr_metric_to_chase` (`str`): reward backend and optional metric key.

## Scoring generated images directly

```python
from XL_diffusers import get_reward_function

scores = get_reward_function(
    reward_name="ImageReward",
    images=result.images,
    prompts=prompts,
)
print(scores)
```

## Notes

- Default `guidance_scale` in these SDXL pipelines is `5.0`.
- For incremental mode, use batch size equal to particle count for predictable behavior.
- For TDR-FK mode, effective batch size must equal `sum(tdr_particles_per_island)`.
- TDR-FK is more expensive than incremental mode because it runs lookahead rollouts and extra rejuvenation/restart work.
- GPU is strongly recommended; several reward models are expensive on CPU.
- Reward backends may require additional model downloads and credentials (for example API-backed graders).
