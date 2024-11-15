import re
import subprocess
from dotenv import load_dotenv

from artifact import load_artifact_path
from volume_env import register_volume_env

from ray.rllib.utils.test_utils import add_rllib_example_script_args, run_rllib_example_script_experiment
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf import FullyConnectedNetwork
from ray import tune
from ray.tune.registry import get_trainable_cls, register_env
import wandb
from wandb.apis.public.runs import Run


load_dotenv(".env")

register_volume_env()

ModelCatalog.register_custom_model("fully_connected_model", FullyConnectedNetwork)


def get_last_run(project: str) -> Run:
    api = wandb.Api()
    runs = api.runs(project, per_page=1)
    if runs:
        return runs[0]
    else:
        raise ValueError("No runs found for the specified project.")


def train():
    parser = add_rllib_example_script_args()

    parser.add_argument(
        "--env-name",
        type=str,
        default=None,
        help="The registered environment name",
    )
    parser.add_argument(
        "--num-envs-per-env-runner",
        type=int,
        default=None,
        help="The number of environments per (remote) EnvRunners to use for the experiment.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="The learning rate for the model, or initial lr if --lr-end is set.",
    )
    parser.add_argument(
        "--lr-end",
        type=float,
        default=None,
        help="If set, linear lr schedule from --lr to this value over --stop-timesteps.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=100,
        help="The batch size for training.",
    )
    parser.add_argument(
        "--initial-epsilon",
        type=float,
        default=0.2,
        help="The initial epsilon value for epsilon-greedy exploration.",
    )
    parser.add_argument(
        "--from-checkpoint",
        type=str,
        default=None,
        help="The path to the tune experiment to load to resume training.",
    )

    args = parser.parse_args()
    args.checkpoint_freq = args.stop_timesteps // 20

    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    project = args.algo + "-" + re.sub("\\W+", "-", args.env_name)
    args.wandb_run_name = project + "-" + commit_hash

    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment(args.env_name)
        .env_runners(
            num_env_runners=args.num_env_runners, num_envs_per_env_runner=args.num_envs_per_env_runner
        )
    )
    base_config["model"] = {
        "custom_model": "fully_connected_model",
    }

    args.num_env_runners = (
        None  # must set to None so run_rllib_example_script_experiment doesn't override env_runners
    )

    base_config["exploration_config"] = {
        "type": "EpsilonGreedy",
        "initial_epsilon": args.initial_epsilon,
        "final_epsilon": 0.001,
        "warmup_timesteps": args.stop_timesteps * 0.2,
        "epsilon_timesteps": args.stop_timesteps * 0.8,
        "explore": True,
        "train_batch_size": args.train_batch_size,
    }

    if args.lr_end:
        base_config["lr_schedule"] = [(0, args.lr), (args.stop_timesteps, args.lr_end)]
    else:
        base_config["lr"] = args.lr

    trainable = None
    if args.from_checkpoint:
        from_checkpoint = (
            args.from_checkpoint
            if args.from_checkpoint.startswith("/")
            else load_artifact_path(args.from_checkpoint)
        )
        Trainable = get_trainable_cls(args.algo)
        trainable = Trainable(base_config)
        trainable.restore(from_checkpoint)

    results: tune.ResultGrid = run_rllib_example_script_experiment(
        base_config, args, trainable=trainable
    )  # type: ignore

    print(f"Experiment finished. Artifact uploaded to W&B Project {project}")
    print("\n\n\n****Best result****\n\n\n")
    print(results.get_best_result())


if __name__ == "__main__":
    train()
