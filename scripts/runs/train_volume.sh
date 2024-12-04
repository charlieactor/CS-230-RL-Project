#!/bin/bash
# Bash script for easy running of train_volume.py:

python scripts/learning/train_volume.py \
    --env-name "VolumeEnv" \
    --num-env-runners 175 \
    --num-envs-per-env-runner 8 \
    --stop-iters 1000 \
    --stop-timesteps 5000000 \
    --stop-reward 275 \
    --train-batch-size 288 \
    --lr 0.001 \
    --initial-epsilon 0.9 \
    --algo APPO \
    --wandb-key $(grep WANDB_API_KEY .env | cut -d '=' -f2) \
    --checkpoint-at-end