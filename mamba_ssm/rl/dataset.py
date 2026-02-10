"""Utilities for preparing offline decision-making datasets for the Decision Mamba stack."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Dict

import torch
from torch.utils.data import Dataset


@dataclass
class DecisionTrajectory:
    """Container for a single offline trajectory."""

    states: torch.Tensor  # (T, state_dim)
    actions: torch.Tensor  # (T,)
    rewards: torch.Tensor  # (T,)
    rtg: Optional[torch.Tensor] = None  # (T,) or (T, 1)

    def __post_init__(self) -> None:
        if self.states.dim() != 2:
            raise ValueError("states must have shape (T, state_dim)")
        if self.actions.dim() != 1:
            raise ValueError("actions must have shape (T,)")
        if self.rewards.dim() != 1:
            raise ValueError("rewards must have shape (T,)")
        if self.states.size(0) != self.actions.size(0) or self.states.size(0) != self.rewards.size(0):
            raise ValueError("states, actions and rewards must share the same trajectory length")
        if self.rtg is not None and self.rtg.size(0) != self.states.size(0):
            raise ValueError("rtg must have the same temporal length as states")


def compute_return_to_go(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute discounted return-to-go for a 1D reward tensor."""

    if rewards.dim() != 1:
        raise ValueError("rewards must be a 1D tensor")
    rtg = torch.zeros_like(rewards, dtype=torch.float32)
    running = torch.tensor(0.0, dtype=torch.float32, device=rewards.device)
    for idx in reversed(range(rewards.shape[0])):
        running = rewards[idx].float() + gamma * running
        rtg[idx] = running
    return rtg


class OfflineDecisionDataset(Dataset):
    """Offline dataset that yields context windows for the Decision Mamba actor."""

    def __init__(
        self,
        trajectories: Iterable[DecisionTrajectory],
        *,
        context_len: int = 1,
        gamma: float = 1.0,
        compute_rtg_if_missing: bool = True,
    ) -> None:
        if context_len <= 0:
            raise ValueError("context_len must be positive")
        self.context_len = context_len
        self.gamma = gamma
        self.samples: List[Dict[str, torch.Tensor]] = []

        traj_list = list(trajectories)
        if not traj_list:
            raise ValueError("trajectories must contain at least one element")

        self.state_dim = traj_list[0].states.size(-1)
        self.action_dim = int(max(int(traj.actions.max().item()) for traj in traj_list) + 1)

        for traj in traj_list:
            states = traj.states.float()
            actions = traj.actions.long()
            rewards = traj.rewards.float()
            if traj.rtg is not None:
                rtg = traj.rtg.float()
            elif compute_rtg_if_missing:
                rtg = compute_return_to_go(rewards, gamma)
            else:
                raise ValueError("RTG missing and compute_rtg_if_missing is False")

            if rtg.dim() == 1:
                rtg = rtg.unsqueeze(-1)

            traj_len = states.size(0)
            for idx in range(traj_len):
                end = idx + 1
                start = max(0, end - context_len)

                state_window = states[start:end]
                rtg_window = rtg[start:end]

                pad_len = context_len - state_window.size(0)
                if pad_len > 0:
                    state_window = torch.cat([
                        torch.zeros(pad_len, self.state_dim, dtype=states.dtype),
                        state_window,
                    ], dim=0)
                    rtg_window = torch.cat([
                        torch.zeros(pad_len, rtg.size(-1), dtype=rtg.dtype),
                        rtg_window,
                    ], dim=0)
                    mask = torch.cat([
                        torch.zeros(pad_len, dtype=torch.float32),
                        torch.ones(state_window.size(0) - pad_len, dtype=torch.float32),
                    ], dim=0)
                else:
                    mask = torch.ones(context_len, dtype=torch.float32)

                self.samples.append(
                    {
                        "state_seq": state_window,
                        "rtg_seq": rtg_window,
                        "mask": mask,
                        "action": actions[idx],
                        "reward": rewards[idx],
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        # Return clones to avoid in-place edits during batching
        action = sample["action"]
        reward = sample["reward"]
        if isinstance(action, torch.Tensor):
            action_tensor = action.clone().long()
        else:
            action_tensor = torch.tensor(action, dtype=torch.long)
        if isinstance(reward, torch.Tensor):
            reward_tensor = reward.clone().float()
        else:
            reward_tensor = torch.tensor(reward, dtype=torch.float32)
        return {
            "state_seq": sample["state_seq"].clone(),
            "rtg_seq": sample["rtg_seq"].clone(),
            "mask": sample["mask"].clone(),
            "action": action_tensor,
            "reward": reward_tensor,
        }

    @property
    def num_actions(self) -> int:
        return self.action_dim

    @property
    def state_dimension(self) -> int:
        return self.state_dim

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        state_seq = torch.stack([item["state_seq"] for item in batch], dim=0)
        rtg_seq = torch.stack([item["rtg_seq"] for item in batch], dim=0)
        mask = torch.stack([item["mask"] for item in batch], dim=0)
        action = torch.stack([item["action"] for item in batch], dim=0)
        reward = torch.stack([item["reward"] for item in batch], dim=0)
        return {
            "state_seq": state_seq,
            "rtg_seq": rtg_seq,
            "mask": mask,
            "action": action,
            "reward": reward,
        }
