from enum import Enum, auto
from typing import Dict, List, Tuple

from director.geometry import Coordinate # Coordinate is an (X, Y, Z) tuple
from director.planner.blended_path_planner import BlendedPathPlanner # Blended path planner uses A* to find a way from point A to point B, with backtracking to solve for blockages.
from director.volume.builder import VolumeBuilder 
from director.volume.volume import PlannerMove, Volume

import gymnasium as gym
import numpy as np
from ray.tune.registry import register_env
import random


class Phase(Enum):
    INDUCT = auto()
    RETRIEVE = auto()


class VolumeEnvironment(gym.Env):
    def __init__(
        self,
        volume_dimensions: Coordinate,
        station_coords: List[Coordinate],
        non_coords: List[Coordinate],
        pez_coords: List[Coordinate],
        pallet_exit_sequence: List[Tuple[int, Coordinate]],
        max_steps: int,
    ):
        super(VolumeEnvironment, self).__init__()
        self.volume = self._build_volume(volume_dimensions, station_coords, non_coords, pez_coords)
        self.max_steps = max_steps

        max_coord = self.volume.max_coord()
        self.grid_size = (max_coord[0] + 1, max_coord[1] + 1, max_coord[2] + 1)
        self._initialize_grid()
        self.stored_pallet_coords = []  # Locations of stored pallets
        # NOTE: storage_locations does not include pez stations with trays. It does include pez stations with no trays.
        # Follow up work here would be to include pez stations with trays, but negatively weight those action choices until the cell is storable.
        self.storage_locations = [cell.coordinate for cell in self.volume.find_storable_cells()]
        self.exit_sequence = pallet_exit_sequence

        self.items_removed = 0  # Tracks how many items have been removed
        self.phase = Phase.INDUCT

        # Define action and observation spaces (I have no idea what I'm doing)
        # action_space: the possible actions the agent can take
        self.action_space = gym.spaces.Discrete(len(self.storage_locations))

        # observation_space: the possible states of the volume, observable by the agent
        # 0 represents empty cell, 1 represents cell with pallet.
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.grid_size, dtype=np.float32)
        print(f"!!!!! Observation space shape: {self.observation_space.shape} !!!!!")

    def _build_volume(
        self,
        dims: Coordinate,
        station_coords: List[Coordinate],
        non_coords: List[Coordinate],
        pez_coords: List[Coordinate],
    ) -> Volume:
        v = VolumeBuilder().add_cell_box(dims[0], dims[1], dims[2])
        for c in station_coords:
            v.make_induction_station(c)
            v.make_retrieval_station(c)
        [v.make_pez_station(c, tray_count=Volume.DEFAULT_PEZ_MAX_CAPACITY) for c in pez_coords]
        v.remove_cells(*non_coords)

        # NOTE: this only uses one bot for now, starting in the first station. Fine for current needs but will need to be updated for multibot.
        v.add_bot("Learny Boy", station_coords[0])
        return v.build()

    def _initialize_grid(self) -> None:
        self.grid = np.zeros(self.grid_size, dtype=int)
        assert np.all((self.grid >= 0) & (self.grid <= 1)), "Grid contains invalid values"

    def reset(self) -> np.ndarray:
        self._initialize_grid()
        self.stored_pallet_coords = []
        self.items_removed = 0
        self.phase = Phase.INDUCT
        return self._get_state().astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        # TODO @charlie: placeholder for induction location. Need to update
        station_coord = (0, 0, 0)
        done = False

        match self.phase:
            case Phase.INDUCT:
                # The action is an index of the storage_locations list. Get the storage location from the list
                storage_coord = self.storage_locations[action]

                # Get the planned path to move the item
                path_planner = BlendedPathPlanner(self.volume)

                # Assuming one bot for now
                bot = self.volume.find_closest_available_bot_to_coord(station_coord)
                assert bot is not None, "No available bots found"

                # TODO @charlie: add bot to PEZ pick and pez to induction put
                pez_coord = self.volume.find_pez_for_pick(bot)
                assert pez_coord is not None, "No available PEZ found for pick"

                #########################################
                # Planning all moves for this induction #
                #########################################
                bot_to_pez_moves = path_planner.plan_bot_pick_tray(bot.coordinate, pez_coord)
                tray_to_station_moves = path_planner.plan_bot_put_tray(pez_coord, station_coord)
                # *A pallet is placed on the tray at the station, ready to be stored*
                bot_to_pallet_moves = path_planner.plan_bot_pick_tray(station_coord, station_coord)
                pallet_to_storage_moves = path_planner.plan_bot_put_tray(station_coord, storage_coord)
                all_moves = (
                    bot_to_pez_moves + tray_to_station_moves + bot_to_pallet_moves + pallet_to_storage_moves
                )
                #####################
                # All moves planned #
                #####################

                # Update the volume, grid, and stored_pallet_coords after storing the item
                self.volume = path_planner._volume  # this one has the updated state after plans complete.
                self.grid[storage_coord] = 1
                self.stored_pallet_coords.append(storage_coord)

                if len(self.stored_pallet_coords) == len(self.storage_locations):
                    self.phase = Phase.RETRIEVE

            case Phase.RETRIEVE:
                # Check if task is complete (all items stored and then removed in order)
                pallet_index, exit_coord = self.exit_sequence[self.items_removed]
                pallet_coord = self.stored_pallet_coords[pallet_index]

                # Get the planned path to move the item
                path_planner = BlendedPathPlanner(self.volume)

                # Assuming one bot for now
                bot = self.volume.find_closest_available_bot_to_coord(pallet_coord)
                assert bot is not None, "No available bots found"

                pez_coord = self.volume.find_pez_for_drop(exit_coord, bot)
                assert pez_coord is not None, "No available PEZ found for drop"

                #########################################
                # Planning all moves for this retrieval #
                #########################################
                bot_to_pallet_moves = path_planner.plan_bot_pick_tray(bot.coordinate, pallet_coord)
                pallet_to_station_moves = path_planner.plan_bot_put_tray(pallet_coord, exit_coord)
                # *pallet is picked from the station. An empty tray remains, to be taken to a PEZ*
                bot_to_pallet_moves = path_planner.plan_bot_pick_tray(exit_coord, exit_coord)
                tray_to_pez_moves = path_planner.plan_bot_put_tray(exit_coord, pez_coord)
                all_moves = (
                    bot_to_pallet_moves + pallet_to_station_moves + bot_to_pallet_moves + tray_to_pez_moves
                )
                #####################
                # All moves planned #
                #####################

                # Update the volume, grid, and items_removed count after removing the item
                self.volume = path_planner._volume  # this one has the updated state after plans complete.
                self.grid[pallet_coord] = 0
                self.items_removed += 1

                if self.items_removed == len(self.exit_sequence):
                    done = True

        # Calculate the reward based on the path
        reward = self._calculate_reward(all_moves)

        # return the state( grid + exit sequence), reward, done flag, and info
        return self._get_state(), reward, done, {}

    def render(self, mode="human"):
        """
        Renders the environment. For simplicity, we'll render the top view (z=0 slice) of the grid
        along with the current phase (storing vs retrieval).
        """
        if mode == "human":
            # Print a simple top-down view of the grid (2D projection of the 3D grid)
            print(f"Current Phase: {self.phase.name}")
            print(f"Items Stored: {len(self.stored_pallet_coords)} / {len(self.exit_sequence)}")

            # Display the top-down view (slice of the grid at z=0)
            top_view = self.grid[:, :, 0]  # Take the first slice along the z-axis (z=0)
            print("Top View (z=0 slice):")
            print(top_view)

            # Show the current item being stored or retrieved
            match self.phase:
                case Phase.INDUCT:
                    print(f"Storing item at index {len(self.stored_pallet_coords) - 1}")

                case Phase.RETRIEVE:
                    if self.items_removed < len(self.exit_sequence):
                        item_to_remove = self.exit_sequence[self.items_removed]
                        print(f"Retrieving item at index {item_to_remove}")

    def _get_state(self) -> np.ndarray:
        """
        Returns the state, which includes:
        - The grid representing every cell in the volume with 0 (empty) or 1 (pallet)
        - The exit sequence indicating the retrieval order and location.
        """
        assert np.all((self.grid >= 0) & (self.grid <= 1)), "Grid contains invalid values"
        return self.grid.flatten()

    def _calculate_reward(self, moves: List[PlannerMove]) -> int:
        # Negatively weight moves that involve changing z-coordinates by 5x
        return sum(-5 if move.start_coordinate()[2] != move.end_coordinate()[2] else -1 for move in moves)


DEFAULT_VOLUME_ENV_CONFIG = {
    "volume_dimensions": (3, 6, 2),
    "station_coords": [(0, 0, 0), (2, 0, 0)],
    "non_coords": [(1, 0, 0)],
    "pez_coords": [(0, 5, 0), (1, 5, 0), (2, 5, 0)],
    "pallet_exit_sequence": [
        (i, (0, 0, 0) if i % 2 == 0 else (2, 0, 0)) for i in random.sample(range(24), 24)
    ],
    "max_steps": 400,
}

ENV_NAME = "VolumeEnv"


def register_volume_env():
    def create_volume_env(env_config):
        return VolumeEnvironment(**{**DEFAULT_VOLUME_ENV_CONFIG, **env_config})

    register_env(ENV_NAME, create_volume_env)
