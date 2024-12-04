from typing import Dict
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.typing import ModelConfigDict, TensorType
import gymnasium as gym


class FlattenGridModel(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Dict | gym.spaces.Box,
        action_space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ) -> None:
        """
        Custom model that flattens a grid-like observation and processes it using fully connected layers.

        Args:
            obs_space (Box): The observation space of the environment.
            action_space: The action space of the environment.
            num_outputs (int): Number of outputs (actions).
            model_config (ModelConfigDict): Configuration dictionary for the model.
            name (str): Name of the model.
        """
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        if isinstance(obs_space, gym.spaces.Dict):
            # Extract shapes from the observation space
            current_grid_shape = obs_space["current_grid"].shape  # e.g., (3, 6, 2)
            next_pallet_info_shape = obs_space[
                "next_pallet_retrieval_order_and_coordinate"
            ].shape  # e.g., (4,)
            assert current_grid_shape and next_pallet_info_shape

            # Calculate flatten sizes
            self.flatten_size = int(torch.prod(torch.tensor(current_grid_shape)))  # 3 * 6 * 2 = 36
            self.total_input_size = self.flatten_size + next_pallet_info_shape[0]  # 36 + 4 = 40
        else:
            self.flatten_size = int(torch.prod(torch.tensor(obs_space.shape)))
            self.total_input_size = self.flatten_size

        # Define the network layers
        self.fc1 = nn.Linear(self.total_input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.q_head = nn.Linear(64, num_outputs)

        # activation func
        self.relu = nn.ReLU()

    def forward(
        self,
        input_dict: Dict[str, TensorType],  # type: ignore
        state: list,
        seq_lens: TensorType,  # type: ignore
    ) -> tuple[TensorType, list]:  # type: ignore
        """
        Forward pass for the model.

        Args:
            input_dict (Dict[str, TensorType]): Dictionary containing the input tensors.
            state (list): List of RNN states (if any).
            seq_lens (TensorType): Sequence lengths tensor (for RNNs).

        Returns:
            tuple[TensorType, list]: Output logits and updated state.
        """
        # Extract the observation components from the input_dict
        current_grid = input_dict["obs"]["current_grid"]
        next_pallet_info = input_dict["obs"]["next_pallet_retrieval_order_and_coordinate"]

        # Convert each component to float if necessary
        if current_grid.dtype != torch.float:
            current_grid = current_grid.float()  # Ensure compatibility with PyTorch layers

        if next_pallet_info.dtype != torch.float:
            next_pallet_info = next_pallet_info.float()  # Ensure compatibility with PyTorch layers

        # Flatten the grid observation (if current_grid is a 3D grid)
        grid_flattened = current_grid.reshape(current_grid.shape[0], -1)

        # Ensure next_pallet_info is compatible for concatenation
        next_pallet_info = (
            next_pallet_info.unsqueeze(0) if next_pallet_info.ndimension() == 1 else next_pallet_info
        )

        # Concatenate the grid with the next_pallet_info (retrieval order + coordinates)
        x = torch.cat([grid_flattened, next_pallet_info], dim=-1)

        # Pass through the fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        q_values = self.q_head(x)

        # handle action masking
        infos = input_dict.get("infos", [])
        if len(infos) > 0 and isinstance(infos[0], dict) and "action_mask" in input_dict["infos"][0]:
            print("************ ACTION MASK ****************")
            # Extract the action mask from each environment
            action_masks = torch.tensor(
                [info["action_mask"] for info in input_dict["infos"]],
                device=q_values.device,
                dtype=torch.float32,
            )  # [batch, num_actions]

            # Set Q-values of invalid actions to a very low value
            q_values = q_values.masked_fill(action_masks == 0, float("-inf"))

        return q_values, state

    def value_function(self) -> TensorType:  # type: ignore
        """
        Placeholder for DQN's value function. Not used but required by RLlib.

        Returns:
            TensorType: A tensor of zeros.
        """
        # DQN doesn't use a separate value function, but RLlib expects this method
        return torch.zeros(1)
