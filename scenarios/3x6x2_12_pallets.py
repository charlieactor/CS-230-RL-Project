import random
from sqlalchemy import Connection

from scenarios.base import Scenario, ScenarioRunResult
from scripts.learning.artifact import load_artifact_path
from scripts.learning.volume_env import Phase, VolumeEnvironment, register_volume_env
from scripts.learning.volume_env_preprocessor import FlattenGridModel

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog

EXIT_SEQ = [(i, (0, 0, 0) if i % 2 == 0 else (2, 0, 0)) for i in random.sample(range(12), 12)]

disabled_coords = [
    (x, y, z) for x in range(0, 10) for y in range(0, 10) for z in range(0, 2) if x > 2 or y > 5
]


def volume_env_creator() -> VolumeEnvironment:
    return VolumeEnvironment(
        (10, 10, 2),
        [(0, 0, 0), (2, 0, 0)],
        [(1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 1)],
        [(0, 5, 0), (1, 5, 0), (2, 5, 0)],
        disabled_coords,
        EXIT_SEQ,
    )


class ThreeXSixX2__TwelvePallets(Scenario):
    def __init__(
        self,
        pallet_count: int,
    ):
        self.pallet_count = pallet_count

    def run(self, db_conn: Connection) -> ScenarioRunResult:
        # Load the trained model checkpoint
        register_volume_env(EXIT_SEQ)
        with ray.init(local_mode=True):
            # TODO: this is downloaded already, no need to download it again
            # model_path = load_artifact_path(
            #     "mytra/appo-volumeenv/checkpoint_APPO-VolumeEnv-67cdf14423a0987ed00df8c96f28209d20405afa:v1"
            # )
            model_path = "/Users/charlie/src/director/artifacts/checkpoint_APPO-VolumeEnv-67cdf14423a0987ed00df8c96f28209d20405afa:v0"
            model = Algorithm.from_checkpoint(model_path)
            env = model.env_creator(model.config["env_config"])
            if not env:
                raise Exception("Environment not found")

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # ~~~~~~~~Setup complete~~~~~~~~
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            scenario_result = ScenarioRunResult()

            # Induct induct induct!
            # NOTE: 24 inductions is based off of 80% of the volume's storable capacity (30)
            print("Starting inductions")

            obs = env._get_state()  # type: ignore
            entry_order_to_exit_order_and_coord = {
                entry_order: (exit_order, exit_coord)
                for exit_order, (entry_order, exit_coord) in enumerate(EXIT_SEQ)
            }

            storage_coords = env.volume.find_storable_coords()  # type: ignore
            for i in range(self.pallet_count):
                exit_order, exit_coord = entry_order_to_exit_order_and_coord[i]
                action = model.compute_single_action(obs)
                storage_coord = storage_coords[action]  # type: ignore
                print(
                    f"{i} pallet in is {exit_order} pallet out. Exit coord is {exit_coord}. Storing at {storage_coord}"
                )

                obs, reward, done, _, __ = env.step(action)

            assert env.phase == Phase.RETRIEVAL  # type: ignore
            assert not done
            for i in range(self.pallet_count):
                action = model.compute_single_action(
                    obs
                )  # action doesn't really matter anymore but we'll keep it
                print(f"Retrieving pallet {i} (exit order)")
                obs, reward, done, _, __ = env.step(action)

            assert done
            scenario_result.moves.extend(env.all_moves)  # type: ignore
            return scenario_result
