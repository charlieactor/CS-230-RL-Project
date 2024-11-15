#!/bin/bash
# Bash script for easy running of train_volume.py:

poetry run python learning/train_volume.py \
    --env-name "VolumeEnv" \
    --num-env-runners 175 \
    --num-envs-per-env-runner 16 \
    --stop-iters 1000 \
    --stop-timesteps 200000000 \
    --stop-reward 200 \
    --train-batch-size 300 \
    --lr 0.001 \
    --initial-epsilon 0.5 \
    --algo APPO \
    --wandb-key $(grep WANDB_API_KEY .env | cut -d '=' -f2) \
    --checkpoint-at-end