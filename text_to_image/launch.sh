#!/bin/bash

python launch_eval_runs.py --use_smc --model_name='runwayml/stable-diffusion-v1-5' --lmbda=10.0 --resample_frequency=20 --resample_t_start=20 --resample_t_end=80 --num_particles=4 --potential_type=max --metrics_to_compute='ImageReward#HPS#PickScore#Aesthetic#CLIP-Score'
