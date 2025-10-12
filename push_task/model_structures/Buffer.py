import torch
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Callable
import pickle 
import os 
class Buffer:
    def __init__(
        self, buffer_size: int, observation_space: Dict[str, Any], action_space: Any
    ):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space

        self.observations = {
            key: np.zeros((buffer_size, *space.shape), dtype=space.dtype)
            for key, space in observation_space.items()
        }
        self.actions = np.zeros(
            (buffer_size, *action_space.shape), dtype=action_space.dtype
        )

        self.position = 0
        self.size = 0

    # def add(self, observations: Dict[str, np.ndarray], actions: np.ndarray):
    #     # Check if the input is batched
    #     is_batched = len(actions.shape) > len(self.action_space.shape)

    #     if not is_batched:
    #         observations = {k: v[np.newaxis, ...] for k, v in observations.items()}
    #         actions = actions[np.newaxis, ...]

    #     batch_size = actions.shape[0]
    #     pos = self.position
    #     for i in range(batch_size):
    #         for key, value in observations.items():
    #             self.observations[key][(pos + i) % self.buffer_size] = value[i]
    #         self.actions[(pos + i) % self.buffer_size] = actions[i]
    #     self.size += batch_size
    #     self.size = min(self.size, self.buffer_size)
    #     self.position = (self.position + batch_size) % self.buffer_size
    
    def add(self, observations: Dict[str, np.ndarray], actions: np.ndarray):
        # Check if the input is batched
        is_batched = len(actions.shape) > len(self.action_space.shape)

        if not is_batched:
            observations = {k: v[np.newaxis, ...] for k, v in observations.items()}
            actions = actions[np.newaxis, ...]

        batch_size = actions.shape[0]

        # Ensure that we remove the oldest elements explicitly when buffer is full
        if self.size + batch_size > self.buffer_size:
            overflow = (self.size + batch_size) - self.buffer_size
            start = self.position  # Oldest data position

            # Shift elements in-place (optional, since overwriting is happening)
            for key in self.observations:
                self.observations[key] = np.roll(self.observations[key], -overflow, axis=0)
            self.actions = np.roll(self.actions, -overflow, axis=0)

            # Adjust size and position
            self.size = self.buffer_size
            self.position = 0  # Reset to start

        pos = self.position
        for i in range(batch_size):
            for key, value in observations.items():
                self.observations[key][(pos + i) % self.buffer_size] = value[i]
            self.actions[(pos + i) % self.buffer_size] = actions[i]

        self.size = min(self.size + batch_size, self.buffer_size)
        self.position = (self.position + batch_size) % self.buffer_size
    def sample(
        self, batch_size: int, device: torch.device
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            {
                key: torch.tensor(obs[indices], dtype=torch.float32, device=device)
                for key, obs in self.observations.items()
            },
            torch.tensor(self.actions[indices], dtype=torch.float32, device=device),
        )
    def sample_seq(
        self, batch_size: int, device: torch.device, seq_len=10
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Samples sequences of length `seq_len` from self.observations and self.actions.

        Args:
            batch_size (int): Number of sequences in a batch.
            device (torch.device): Target device (CPU or GPU).
            seq_len (int): Length of each sequence.

        Returns:
            Tuple[
                Dict[str, torch.Tensor],  # Observations with shape (batch_size, seq_len, obs_dim)
                torch.Tensor  # Actions with shape (batch_size, seq_len, action_dim)
            ]
        """
        # Ensure indices start from at least `seq_len - 1` to allow full sequence sampling
        indices = np.random.randint(seq_len - 1, self.size, size=batch_size)

        # Extract sequences for each index
        obs_sequences = {
            key: torch.tensor(
                [obs[idx - (seq_len - 1) : idx + 1] for idx in indices],  # Collect sequences
                dtype=torch.float32,
                device=device
            )
            for key, obs in self.observations.items()
        }

        action_sequences = torch.tensor(
            [self.actions[idx - (seq_len - 1) : idx + 1] for idx in indices], 
            dtype=torch.float32,
            device=device
        )

        return obs_sequences, action_sequences
    def sample_all_seq_batches(self, batch_size: int, device: torch.device, seq_len=10):
        """
        Samples all possible sequences and returns them as a list instead of yielding.
        
        Args:
            batch_size (int): Number of sequences per batch.
            device (torch.device): Target device (CPU or GPU).
            seq_len (int): Length of each sequence.

        Returns:
            List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]
            - A list where each element is a tuple containing:
                - Observations with shape (batch_size, seq_len, obs_dim)
                - Actions with shape (batch_size, seq_len, action_dim)
        """
        # Ensure indices start from at least `seq_len - 1` to allow full sequence sampling
        valid_indices = np.arange(seq_len - 1, self.size)  # Only valid sequence start points
        np.random.shuffle(valid_indices)  # Shuffle batch indices but keep sequence order

        # Split shuffled indices into batches
        batches = [valid_indices[i : i + batch_size] for i in range(0, len(valid_indices), batch_size)]
        
        all_batches = []  # Store all batches

        for batch_indices in batches:
            # Extract sequences for each index in the batch
            obs_sequences = {
                key: torch.tensor(
                    np.array([obs[idx - (seq_len - 1) : idx + 1] for idx in batch_indices]),  # ✅ No extra wrapping
                    dtype=torch.float32,
                    device=device
                )
                for key, obs in self.observations.items()
            }

            action_sequences = torch.tensor(
                np.array([self.actions[idx - (seq_len - 1) : idx + 1] for idx in batch_indices]),
                dtype=torch.float32,
                device=device
            )

            all_batches.append((obs_sequences, action_sequences))

        return all_batches  

    def __len__(self):
        return self.size

    def save(self, fname, identifier=None):
    # Extract directory and filename
        dir_path = os.path.dirname(fname)
        base_name, ext = os.path.splitext(os.path.basename(fname))

        # Ensure directory exists
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if identifier is None:
            with open(fname, "wb") as f:
                pickle.dump(self, f)
        else:
        # Modify filename by appending self.__len__()
            modified_fname = os.path.join(dir_path, f"{base_name}_{identifier}{ext}")

            # Save the file
            with open(modified_fname, "wb") as f:
                pickle.dump(self, f)
            print(f"Saved successfully to {modified_fname}")

    def load(self, fname):
        with open(fname, "rb") as f:
            self = pickle.load(f)

