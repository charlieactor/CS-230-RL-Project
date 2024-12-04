import random
import re
from sched import scheduler
import subprocess
import time
from dotenv import load_dotenv
from ray import air
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.tune.schedulers import ASHAScheduler
from scripts.learning.volume_env import ENV_NAME
import argparse

from volume_env import register_volume_env


load_dotenv(".env")

register_volume_env()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-envs-per-env-runner",
        type=int,
        default=174,
        help="The number of environments per (remote) EnvRunners to use for the experiment.",
    )
    parser.add_argument(
        "--num-env-runners",
        type=int,
        default=16,
        help="The number of EnvRunners to use for the experiment.",
    )
    parser.add_argument(
        "--stop-timesteps",
        type=int,
        default=int(1e6),
        help="The number of (environment sampling) timesteps to train.",
    )
    return parser.parse_args()


def train():
    args = parse_args()

    config = ImpalaConfig()
    # Update the config object.
    config.training(
        lr=tune.grid_search([0.003, 0.0003]),
        train_batch_size=tune.grid_search([100, 300]),
    )

    config.environment(
        ENV_NAME,
        env_config={
            "volume_dimensions": (10, 10, 2),
            "station_coords": [(0, 0, 0), (2, 0, 0)],
            "non_coords": [(1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 1)],
            "pez_coords": [(0, 5, 0), (1, 5, 0), (2, 5, 0)],
            "pallet_exit_sequence": [
                (i, (0, 0, 0) if i % 2 == 0 else (2, 0, 0)) for i in random.sample(range(12), 12)
            ],
        },
    )
    config.num_env_runners = tune.grid_search([32, 64])  # type: ignore
    config.num_envs_per_env_runner = tune.grid_search([8, 16])
    config.exploration_config = {
        "type": "EpsilonGreedy",
        "initial_epsilon": tune.grid_search([0.1, 0.03]),
        "final_epsilon": 0.001,
        "warmup_timesteps": args.stop_timesteps * 0.2,
        "epsilon_timesteps": args.stop_timesteps * 0.8,
    }
    config.num_gpus = 0

    time_since_epoch = time.time()
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    project = "Tune" + "-" + str(int(time_since_epoch % 10000)) + "-" + re.sub("\\W+", "-", str(ENV_NAME))
    wandb_run_name = project + "-" + commit_hash

    tune_callbacks: list = [
        WandbLoggerCallback(
            project=project,
            upload_checkpoints=True,
            name=wandb_run_name,
            config=config.to_dict(),
        )
    ]

    tune.Tuner(
        "IMPALA",
        run_config=air.RunConfig(
            stop={"num_env_steps_sampled_lifetime": args.stop_timesteps},
            verbose=0,
            callbacks=tune_callbacks,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_at_end=True,
            ),
        ),
        param_space=config.to_dict(),
        tune_config=tune.TuneConfig(
            scheduler=ASHAScheduler(
                time_attr="timesteps_total",
                metric="env_runners/episode_reward_mean",
                mode="max",
                max_t=args.stop_timesteps,
                grace_period=args.stop_timesteps // 10,
                reduction_factor=3,
                brackets=3,
            ),
        ),
    ).fit()


if __name__ == "__main__":
    train()
