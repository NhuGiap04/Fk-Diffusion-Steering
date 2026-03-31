#!/bin/bash
set -euo pipefail

python launch_evolve_runs.py \
  --prompt_path prompt_files/benchmark_ir.json \
  --model_name stabilityai/stable-diffusion-xl-base-1.0 \
  --save_individual_images \
  --num_loops 4 \
  --num_particles 8 \
  --num_inference_steps 50 \
  --guidance_scale 5.0 \
  --guidance_reward_fn ImageReward \
  --steer_start_timestep 400 \
  --steer_end_timestep 150 \
  --stein_step_size 0.01 \
  --stein_rejected_penalty 0.5 \
  --stein_num_steps 20 \
  --early_stop_epsilon 1e-4 \
  --output_dir benchmark_ir_outputs_evolve
