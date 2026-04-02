#!/bin/bash
set -euo pipefail

python launch_evolve_runs.py \
  --prompt_path prompt_files/benchmark_ir.json \
  --model_name stabilityai/stable-diffusion-xl-base-1.0 \
  --save_individual_images \
  --num_loops 2 \
  --num_particles 8 \
  --num_inference_steps 50 \
  --guidance_scale 5.0 \
  --guidance_reward_fn ImageReward \
  --steer_start_timestep 400 \
  --steer_end_timestep 160 \
  --stein_step_size 0.01 \
  --stein_weight_temperature 1.0 \
  --stein_langevin_lambda 1.0 \
  --stein_num_steps 10 \
  --early_stop_epsilon 1e-4 \
  --output_dir benchmark_ir_outputs_evolve
