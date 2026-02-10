#!/usr/bin/env python3
"""Example training script for the Decision Mamba offline RL agent."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader

from mamba_ssm.rl import (
    DecisionMambaActor,
    DecisionMambaCQLTrainer,
    DecisionMambaConfig,
    DecisionTrajectory,
    OfflineDecisionDataset,
    TrainerConfig,
    TwinQCritic,
    compute_return_to_go,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Decision Mamba offline RL agent")
    parser.add_argument("--dataset", type=Path, default=None, help="Path to a torch-saved dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--context-len", type=int, default=4, help="Context window length")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for RTG")
    parser.add_argument("--d-model", type=int, default=256, help="Decision Mamba hidden size")
    parser.add_argument("--actor-lr", type=float, default=3e-4, help="Actor learning rate")
    parser.add_argument("--critic-lr", type=float, default=3e-4, help="Critic learning rate")
    parser.add_argument("--critic-hidden", type=int, nargs="*", default=[256, 256], help="Critic hidden sizes")
    parser.add_argument("--cql-alpha", type=float, default=1.0, help="CQL regularisation strength")
    parser.add_argument("--cql-num-samples", type=int, default=10, help="Number of policy samples for CQL")
    parser.add_argument("--actor-update-freq", type=int, default=5, help="Update actor every d critic steps")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient for target critics")
    parser.add_argument("--grad-clip", type=float, default=None, help="Gradient clipping value")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--synthetic-trajectories", type=int, default=64, help="Synthetic trajectories when no dataset")
    parser.add_argument("--synthetic-length", type=int, default=6, help="Length of synthetic trajectories")
    parser.add_argument("--synthetic-state-dim", type=int, default=16, help="State dimensionality for synthetic data")
    parser.add_argument("--synthetic-action-dim", type=int, default=6, help="Number of actions for synthetic data")
    parser.add_argument("--log-interval", type=int, default=10, help="Steps between metric logs")
    return parser.parse_args()


def _load_trajectories_from_file(path: Path, gamma: float) -> List[DecisionTrajectory]:
    payload = torch.load(path)
    if isinstance(payload, dict) and "trajectories" in payload:
        payload = payload["trajectories"]
    trajectories = []
    for item in payload:
        if not isinstance(item, Dict):
            raise ValueError("Each trajectory entry must be a dict-like object")
        states = torch.tensor(item["states"], dtype=torch.float32)
        actions = torch.tensor(item["actions"], dtype=torch.long)
        rewards = torch.tensor(item["rewards"], dtype=torch.float32)
        rtg = item.get("rtg")
        rtg_tensor = (
            torch.tensor(rtg, dtype=torch.float32)
            if rtg is not None
            else compute_return_to_go(rewards, gamma)
        )
        if rtg_tensor.dim() == 1:
            rtg_tensor = rtg_tensor.unsqueeze(-1)
        trajectories.append(DecisionTrajectory(states=states, actions=actions, rewards=rewards, rtg=rtg_tensor))
    return trajectories


def _build_synthetic_dataset(
    *,
    num_trajectories: int,
    length: int,
    state_dim: int,
    action_dim: int,
    gamma: float,
) -> List[DecisionTrajectory]:
    trajectories = []
    for _ in range(num_trajectories):
        states = torch.randn(length, state_dim)
        actions = torch.randint(low=0, high=action_dim, size=(length,))
        rewards = torch.randn(length)
        rtg = compute_return_to_go(rewards, gamma).unsqueeze(-1)
        trajectories.append(DecisionTrajectory(states=states, actions=actions, rewards=rewards, rtg=rtg))
    return trajectories


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.dataset is not None and args.dataset.exists():
        trajectories = _load_trajectories_from_file(args.dataset, args.gamma)
    else:
        trajectories = _build_synthetic_dataset(
            num_trajectories=args.synthetic_trajectories,
            length=args.synthetic_length,
            state_dim=args.synthetic_state_dim,
            action_dim=args.synthetic_action_dim,
            gamma=args.gamma,
        )

    dataset = OfflineDecisionDataset(
        trajectories,
        context_len=args.context_len,
        gamma=args.gamma,
        compute_rtg_if_missing=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=OfflineDecisionDataset.collate_fn,
        drop_last=False,
    )

    actor_cfg = DecisionMambaConfig(
        state_dim=dataset.state_dimension,
        action_dim=dataset.num_actions,
        rtg_dim=dataset.samples[0]["rtg_seq"].size(-1),
        d_model=args.d_model,
        max_seq_len=args.context_len,
    )
    actor = DecisionMambaActor(actor_cfg)
    critic = TwinQCritic(
        state_dim=dataset.state_dimension,
        action_dim=dataset.num_actions,
        hidden_sizes=args.critic_hidden,
    )
    trainer_cfg = TrainerConfig(
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        tau=args.tau,
        cql_alpha=args.cql_alpha,
        cql_num_samples=args.cql_num_samples,
        actor_update_frequency=args.actor_update_freq,
        gradient_clip=args.grad_clip,
        device=args.device,
    )
    trainer = DecisionMambaCQLTrainer(actor, critic, trainer_cfg)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        for batch in dataloader:
            metrics = trainer.step(batch)
            global_step += 1
            if args.log_interval and global_step % args.log_interval == 0:
                print(
                    f"Epoch {epoch} Step {global_step}: "
                    + ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())
                )

    print("Training complete.")


if __name__ == "__main__":
    main()
