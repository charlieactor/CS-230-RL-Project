from enum import Enum, auto
from typing import Any, Dict, List, Tuple

from director.geometry import Coordinate, manhattan_distance
from director.planner.blended_path_planner import BlendedPathPlanner
from director.volume.builder import VolumeBuilder
from director.volume.pallet import Pallet, PalletState
from director.volume.volume import BotPick, BotPut, PlannerMove, Volume

import gymnasium as gym
import numpy as np
from ray.tune.registry import register_env
import random


class Phase(Enum):
    INDUCTION = auto()
    RETRIEVAL = auto()


class VolumeEnvironment(gym.Env):
    #####################
    # Static properties #
    #####################
    _volume_start_state: Volume
    grid_size: Tuple[int, int, int]
    min_moves_observed: float
    max_moves_observed: float

    #########################
    # Resettable Properties #
    #########################
    volume: Volume
    storage_coords: List[Coordinate]  # action is an index in this list
    entry_ordered_tray_ids: List[int]
    tray_id_to_coord: Dict[int, Coordinate]
    exit_sequence: List[Tuple[int, Coordinate]]  # order that pallet came in, exit coord
    phase: Phase
    all_moves: List[PlannerMove]

    def __init__(
        self,
        volume_dimensions: Coordinate,
        station_coords: List[Coordinate],
        non_coords: List[Coordinate],
        pez_coords: List[Coordinate],
        disabled_coords: List[Coordinate],
        pallet_exit_sequence: List[Tuple[int, Coordinate]],
    ):
        super(VolumeEnvironment, self).__init__()

        self._volume_start_state = self._build_volume(
            volume_dimensions, station_coords, non_coords, pez_coords, disabled_coords
        )
        # TODO: consider resetting these every N episodes
        # TODO: OR keeping a rolling average over last N episodes
        self.min_moves_observed = float("inf")
        self.max_moves_observed = float("-inf")
        self.volume = self._volume_start_state.copy()
        self.exit_sequence = pallet_exit_sequence
        self.grid_size = volume_dimensions

        self._initialize_grid()
        self.entry_ordered_tray_ids = []
        self.tray_id_to_coord = {}
        self.phase = Phase.INDUCTION
        self.all_moves = []

        # NOTE: storage_coords does not include pez stations with trays. It does include pez stations with no trays.
        # Follow up work here would be to include pez stations with trays, but negatively weight those action choices until the cell is storable.
        self.storage_coords = self.volume.find_storable_coords()

        # Define action and observation spaces
        # action_space: the possible actions the agent can take. Returns an int representing an index in the storage_coords list.
        self.action_space = gym.spaces.Discrete(len(self.storage_coords))

        # observation_space: the possible states of the volume, observable by the agent
        # 0 represents empty cell, 1 represents cell with pallet.
        # TODO @Charlie: including exit_coord will likely benefit us here as well, but would require more metadata.
        self.observation_space = gym.spaces.Dict(
            {
                "current_grid": gym.spaces.Box(
                    low=0,
                    high=len(self.exit_sequence),
                    shape=self.grid_size,
                    dtype=np.int32,
                ),
                "next_pallet_retrieval_order_and_coordinate": gym.spaces.Box(
                    low=-1,
                    high=max(
                        len(self.exit_sequence),
                        self.volume.max_coord()[0],
                        self.volume.max_coord()[1],
                        self.volume.max_coord()[2],
                    ),
                    shape=(4,),
                    dtype=np.int32,
                ),
            }
        )

    def _build_volume(
        self,
        dims: Coordinate,
        station_coords: List[Coordinate],
        non_coords: List[Coordinate],
        pez_coords: List[Coordinate],
        disabled_coords: List[Coordinate],
    ) -> Volume:
        v = VolumeBuilder().add_cell_box(dims[0], dims[1], dims[2])
        for c in station_coords:
            v.make_induction_station(c)
            v.make_retrieval_station(c)
        [v.make_pez_station(c, tray_count=Volume.DEFAULT_PEZ_MAX_CAPACITY) for c in pez_coords]
        v.remove_cells(*non_coords)
        v.disable_cells(*disabled_coords)

        # NOTE: this only uses one bot for now, starting in the first station. Fine for current needs but will need to be updated for multibot.
        v.add_bot("Learny Boy", station_coords[0])
        return v.build()

    def _initialize_grid(self) -> None:
        self.grid = np.zeros(self.grid_size, dtype=np.int32)

    def reset(
        self, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        super().reset(seed=seed, options=options)
        self.volume = self._volume_start_state.copy()
        self.volume.find_storable_coords()
        self._initialize_grid()
        self.entry_ordered_tray_ids = []
        self.tray_id_to_coord = {}
        self.phase = Phase.INDUCTION
        # Shuffle the exit sequence for each episode
        random.shuffle(self.exit_sequence)
        self.all_moves = []
        return self._get_state(), {}

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        pallets_removed = 0
        done = False

        # Get the planned path to move the item
        path_planner = BlendedPathPlanner(self.volume)

        match self.phase:
            case Phase.INDUCTION:
                storage_coord = self.storage_coords[action]

                # TODO @charlie: placeholder for induction coord. Need to update
                station_coord = (0, 0, 0)

                # The action is an index of the storage_coords list. Get the storage coord from the list

                # Assuming one bot for now
                bot = self.volume.find_closest_available_bot_to_coord(station_coord)
                assert bot is not None, "No available bots found"

                pez_coord = self.volume.find_pez_for_pick(bot)
                assert pez_coord is not None, "No available PEZ found for pick"

                # Update the entry order to tray id mapping to track this for retrieval
                tray = path_planner._volume.pez_stations[pez_coord].peek()
                assert tray, "No tray found in PEZ"
                self.entry_ordered_tray_ids.append(tray.id)

                #########################################
                # Planning all moves for this induction #
                #########################################
                print(
                    f"Bot at {bot.coordinate}. Picking up tray {tray.id} from {pez_coord} to bring to {station_coord}, then storing at {storage_coord}."
                )
                bot_to_pez_moves = path_planner.plan_bot_pick_tray(bot.coordinate, pez_coord)
                tray_to_station_moves = path_planner.plan_bot_put_tray(pez_coord, station_coord)
                num_pallets = len(self.volume.pallets)
                path_planner._volume.add_pallet(
                    Pallet(num_pallets, tray.id, PalletState.AWAITING_INDUCT, {}), tray
                )
                num_pallets += 1

                # *A pallet is placed on the tray at the station, ready to be stored*
                bot_to_pallet_moves = path_planner.plan_bot_pick_tray(station_coord, station_coord)
                pallet_to_storage_moves = path_planner.plan_bot_put_tray(station_coord, storage_coord)
                all_moves = (
                    bot_to_pez_moves + tray_to_station_moves + bot_to_pallet_moves + pallet_to_storage_moves
                )

                if num_pallets == len(self.exit_sequence):
                    self.phase = Phase.RETRIEVAL

            case Phase.RETRIEVAL:
                pallets_removed = len(self.exit_sequence) - len(self.volume.pallets)
                entry_order, exit_coord = self.exit_sequence[pallets_removed]
                print(
                    f"{pallets_removed} pallets removed so far. Next pallet to remove was the {entry_order}th to enter."
                )
                print(f"Exit sequence: {self.exit_sequence}. Next up: {self.exit_sequence[pallets_removed]}")
                print(f"Tray IDs in entry order: {self.entry_ordered_tray_ids}")
                print(f"Tray ID to coord: {self.tray_id_to_coord}")
                tray_id = self.entry_ordered_tray_ids[entry_order]
                print(f"Tray to remove: {tray_id}")
                tray_coord = self.tray_id_to_coord[tray_id]
                print(f"Tray coord: {tray_coord}")

                # Get the planned path to move the item
                path_planner = BlendedPathPlanner(self.volume)

                # Assuming one bot for now
                bot = self.volume.find_closest_available_bot_to_coord(tray_coord)
                assert bot is not None, "No available bots found"

                pez_coord = self.volume.find_pez_for_drop(exit_coord, bot)
                assert pez_coord is not None, "No available PEZ found for drop"

                #########################################
                # Planning all moves for this retrieval #
                #########################################
                bot_to_pallet_moves = path_planner.plan_bot_pick_tray(bot.coordinate, tray_coord)
                pallet_to_station_moves = path_planner.plan_bot_put_tray(tray_coord, exit_coord)
                # remove the pallet from the tray
                del path_planner._volume.pallets[exit_coord]
                pallets_removed += 1
                # *pallet is picked from the station. An empty tray remains, to be taken to a PEZ*
                bot_to_pallet_moves = path_planner.plan_bot_pick_tray(exit_coord, exit_coord)
                try:
                    tray_to_pez_moves = path_planner.plan_bot_put_tray(exit_coord, pez_coord)
                except Exception as e:
                    print(f"Error: {e}")
                    print(
                        f"Volume state: bot at {path_planner._volume.bots.keys()}. Trays at {path_planner._volume.used_trays.keys()}"
                    )
                    raise e

                all_moves = (
                    bot_to_pallet_moves + pallet_to_station_moves + bot_to_pallet_moves + tray_to_pez_moves
                )

        #####################
        # All moves planned #
        #####################

        # Update the volume, grid, and stored_pallet_coords after storing the item
        self.volume = path_planner._volume  # this one has the updated state after plans complete.

        # Since those moves could have shuffled things around, update the grid and tray_id_to_coord accordingly
        self._update_stored_coords_and_action_space()

        # once all trays have been removed from the volume, we are done.
        done = self.phase == Phase.RETRIEVAL and pallets_removed == len(self.exit_sequence)

        # Calculate the reward based on the path
        self.all_moves += all_moves
        reward = self._calculate_reward(all_moves)
        truncated = False

        # return the state( grid + exit sequence), reward, done flag, and info
        return self._get_state(), reward, done, truncated, {}

    def _update_stored_coords_and_action_space(self) -> None:
        self._initialize_grid()
        self.tray_id_to_coord = {}
        for coord, tray in self.volume.used_trays.items():
            self.grid[coord] = 1
            self.tray_id_to_coord[tray.id] = coord

    def render(self, mode="human"):
        """
        Renders the environment. For simplicity, we'll render the top view (z=0 slice) of the grid
        along with the current phase (storing vs retrieval).
        """
        if mode == "human":
            # Print a simple top-down view of the grid (2D projection of the 3D grid)
            print(f"Items Stored: {len(self.tray_id_to_coord)} / {len(self.exit_sequence)}")

            # Display the top-down view (slice of the grid at z=0)
            top_view = self.grid[:, :, 0]  # Take the first slice along the z-axis (z=0)
            print("Top View (z=0 slice):")
            print(top_view)

            # Show the current item being stored or retrieved
            print(f"Storing item at index {len(self.tray_id_to_coord) - 1}")

    def _get_state(self) -> Dict[str, np.ndarray]:
        """
        Returns the state, including:
        - Channel 0: The grid (0 for empty, 1 for occupied).
        - Channel 1: Retrieval order (higher values for later retrieval).
        """
        # Retrieval order metadata (Channel 1)
        grid = np.zeros_like(self.grid)
        for out_order, (entry_order, _) in enumerate(self.exit_sequence):
            # Can't add for pallets not yet in the volume (or already removed)
            if entry_order >= len(self.entry_ordered_tray_ids):
                continue
            tray_id = self.entry_ordered_tray_ids[entry_order]

            tray_coord = self.tray_id_to_coord.get(tray_id)
            if tray_coord is not None:
                grid[tray_coord] = out_order + 1  # Retrieval order starts at 1

        next_pallet_info = self._get_next_pallet_exit_order_and_coord()

        return {
            "current_grid": grid,
            "next_pallet_retrieval_order_and_coordinate": next_pallet_info,
        }

    def _calculate_reward(self, moves: List[PlannerMove]) -> float:
        move_sum = len(moves)

        # # Handle initialization (first pass)
        if self.min_moves_observed == float("inf") or self.max_moves_observed == float("-inf"):
            # Set initial bounds
            self.min_moves_observed = move_sum
            self.max_moves_observed = move_sum
            return 0  # Neutral reward for the first pass

        # Use the current observed bounds to calculate the reward
        if self.max_moves_observed == self.min_moves_observed:
            # Avoid division by zero during initialization
            normalized_move_reward = 0.1
        else:
            normalized_move_reward = (
                2
                * (
                    1
                    - (move_sum - self.min_moves_observed)
                    / (self.max_moves_observed - self.min_moves_observed)
                )
                - 1
            )

        # Clamp the normalized reward to [-1, 1]
        normalized_move_reward = max(min(normalized_move_reward, 1), -1)

        # Update bounds AFTER calculating the reward
        self.min_moves_observed = min(self.min_moves_observed, move_sum)
        self.max_moves_observed = max(self.max_moves_observed, move_sum)

        # Debugging
        print(
            f"Move Sum: {move_sum}, Min Observed: {self.min_moves_observed}, "
            f"Max Observed: {self.max_moves_observed}, Normalized Reward: {normalized_move_reward}"
        )

        return normalized_move_reward * 10  # scaled by 10 for some reason, idk.

    def _get_next_pallet_exit_order_and_coord(self) -> np.ndarray:
        if len(self.entry_ordered_tray_ids) < len(self.exit_sequence):
            next_pallet_exit_order, next_pallet_exit_coord = next(
                (exit_order, exit_coord)
                for exit_order, (entry_order, exit_coord) in enumerate(self.exit_sequence)
                if entry_order == len(self.entry_ordered_tray_ids)
            )
        else:
            next_pallet_exit_order = -1
            next_pallet_exit_coord = (-1, -1, -1)

        return np.array([next_pallet_exit_order, *next_pallet_exit_coord], dtype=np.int32)


####################
# Env Config Stuff #
####################
disabled_coords = [
    (x, y, z) for x in range(0, 10) for y in range(0, 10) for z in range(0, 2) if x > 2 or y > 5
]
DEFAULT_VOLUME_ENV_CONFIG = {
    "volume_dimensions": (10, 10, 2),  # TODO: only works for DQN for now
    "station_coords": [(0, 0, 0), (2, 0, 0)],
    "non_coords": [(1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 1)],  # TODO: only works for DQN for now
    "pez_coords": [(0, 5, 0), (1, 5, 0), (2, 5, 0)],
    "disabled_coords": disabled_coords,
}

ENV_NAME = "VolumeEnv"


def register_volume_env(pallet_exit_seq: List[Tuple[int, Coordinate]] | None = None):
    if pallet_exit_seq is None:
        pallet_exit_seq = [(i, (0, 0, 0) if i % 2 == 0 else (2, 0, 0)) for i in random.sample(range(12), 12)]

    def create_volume_env(env_config):
        DEFAULT_VOLUME_ENV_CONFIG["pallet_exit_sequence"] = pallet_exit_seq
        return VolumeEnvironment(**{**DEFAULT_VOLUME_ENV_CONFIG, **env_config})

    register_env(ENV_NAME, create_volume_env)
