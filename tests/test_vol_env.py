import torch
import numpy as np
import random

from scripts.learning.volume_env import Phase, VolumeEnvironment
from scripts.learning.volume_env_preprocessor import FlattenGridModel


def build_vol_env() -> VolumeEnvironment:
    dims = (3, 6, 2)
    station_coords = [(0, 0, 0), (2, 0, 0)]
    pez_coords = [(0, 5, 0), (1, 5, 0), (2, 5, 0)]
    pallet_exit_sequence = [(i, (0, 0, 0) if i % 2 == 0 else (2, 0, 0)) for i in random.sample(range(12), 12)]

    return VolumeEnvironment(dims, station_coords, [], pez_coords, [], pallet_exit_sequence)


def test_step():
    vol_env = build_vol_env()
    assert len(vol_env.volume.used_trays) == 0, "No used trays yet"
    assert np.all(vol_env.grid == 0), "All grid cells are empty"
    assert vol_env.phase == Phase.INDUCTION, "Phase is induction"

    # actions represents the storage coord in vol_env.storage_coords that the agent selects
    actions = [i for i in random.sample(range(12), 12)]
    obs, _ = vol_env.reset()
    for coord in vol_env.volume.cells.keys():
        assert obs["current_grid"][coord] == 0, "All cells are empty"

    assert (
        obs["next_pallet_retrieval_order_and_coordinate"].all()
        == vol_env._get_next_pallet_exit_order_and_coord().all()
    ), "Next pallet retrieval order is correct"

    for i in range(len(actions)):
        storage_index = actions[i]
        storage_coord = vol_env.storage_coords[storage_index]
        state, reward, done, _, _ = vol_env.step(storage_index)
        assert state["current_grid"].all() == vol_env.grid.all(), "Returned grid matches env grid"

        assert (
            -20 <= reward <= 20
        ), "Reward is scaled between -20 and 20 (normalized between 1 and -1, plus/minus shuffle factor, scaled by 10)"

        assert not done, "Not done yet"
        assert vol_env.grid[storage_coord] == 1, "Grid cell is now occupied"
        assert storage_coord in vol_env.volume.used_trays, "Storage location is now used"
        tray = vol_env.volume.used_trays[storage_coord]
        assert vol_env.entry_ordered_tray_ids[i] == tray.id, "Tray id is correctly stored for entry order"
        assert vol_env.tray_id_to_coord[tray.id] == storage_coord, "Tray location is correctly stored"

        for coord, tray in vol_env.volume.used_trays.items():
            assert vol_env.tray_id_to_coord[tray.id] == coord, "Tray location is correctly stored"

        used_tray_coords = set(vol_env.volume.used_trays.keys())
        for coord, value in np.ndenumerate(vol_env.grid):
            if coord in used_tray_coords:
                assert value == 1, f"Grid cell {coord} should be occupied"
            else:
                assert value == 0, f"Grid cell {coord} should be empty"

    assert vol_env.phase == Phase.RETRIEVAL, "Phase is now retrieval"
    for i in range(len(actions)):
        state, reward, done, _, _ = vol_env.step(-1)  # action irrelevant in retreival phase.
        assert state["current_grid"].all() == vol_env.grid.all(), "Returned grid matches env grid"
        assert (
            -20 <= reward <= 20
        ), "Reward is scaled between -20 and 20 (normalized between 1 and -1, plus/minus shuffle factor, scaled by 10)"

        if i == len(actions) - 1:
            assert len(vol_env.volume.used_trays) == 0, "All trays have been stored back in pez"
            assert len(vol_env.tray_id_to_coord) == 0, "Tray id to location map is empty"
            assert done
        else:
            assert not done, "Not done yet"


def test_reset():
    vol_env = build_vol_env()

    # actions represents the storage coord in vol_env.storage_coords that the agent selects
    actions = [i for i in random.sample(range(12), 12)]
    [vol_env.step(actions[i]) for i in range(len(actions))]

    # confirm the various fields are updated and stuff
    assert not np.all(vol_env.grid == 0), "Not all grid cells are empty"
    assert len(vol_env.entry_ordered_tray_ids) == 12
    assert len(vol_env.volume.used_trays) == 12
    assert len(vol_env.tray_id_to_coord) == 12
    assert vol_env.phase == Phase.RETRIEVAL

    vol_env.reset()
    assert np.all(vol_env.grid == 0), "All grid cells are empty"
    assert len(vol_env.entry_ordered_tray_ids) == 0
    assert len(vol_env.volume.used_trays) == 0
    assert len(vol_env.tray_id_to_coord) == 0
    assert vol_env.phase == Phase.INDUCTION


def test_preprocessor():
    env = build_vol_env()
    obs, info = env.reset()

    action = 0
    obs, reward, done, truncated, info = env.step(action)

    input_dict = {
        "obs": {
            "current_grid": torch.tensor(obs["current_grid"]).unsqueeze(0),  # Shape: [1, 3, 6, 2]
            "next_pallet_retrieval_order_and_coordinate": torch.tensor(
                obs["next_pallet_retrieval_order_and_coordinate"]
            ).unsqueeze(
                0
            ),  # Shape: [1, 4]
        },
        "infos": [info],  # List containing the info dict
    }
    # Initialize the custom model
    model = FlattenGridModel(
        obs_space=env.observation_space,  # type: ignore
        action_space=env.action_space,
        num_outputs=env.action_space.n,  # type:ignore
        model_config={},
        name="flatten_grid_model",
    )

    q_vals, state = model.forward(input_dict, state=[], seq_lens=None)
    assert q_vals.shape == (1, env.action_space.n), "Q-values shape is correct"  # type: ignore
    assert q_vals.data[0][0] == float("-inf"), "Q-values are masked correctly"
