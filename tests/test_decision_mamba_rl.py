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


def _build_dummy_dataset(num_trajectories: int = 8, length: int = 4):
    trajectories = []
    state_dim = 8
    action_dim = 5
    gamma = 0.95
    for _ in range(num_trajectories):
        states = torch.randn(length, state_dim)
        actions = torch.randint(low=0, high=action_dim, size=(length,))
        rewards = torch.randn(length)
        rtg = compute_return_to_go(rewards, gamma).unsqueeze(-1)
        trajectories.append(DecisionTrajectory(states=states, actions=actions, rewards=rewards, rtg=rtg))
    dataset = OfflineDecisionDataset(trajectories, context_len=3, gamma=gamma)
    return dataset, state_dim, action_dim


def test_decision_mamba_trainer_step_runs():
    dataset, state_dim, action_dim = _build_dummy_dataset()
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=OfflineDecisionDataset.collate_fn,
    )
    batch = next(iter(dataloader))

    actor_cfg = DecisionMambaConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=64,
        max_seq_len=dataset.context_len,
        rtg_dim=batch["rtg_seq"].size(-1),
    )
    actor = DecisionMambaActor(actor_cfg)
    critic = TwinQCritic(state_dim=state_dim, action_dim=action_dim, hidden_sizes=(64, 64))
    trainer_cfg = TrainerConfig(
        actor_lr=1e-3,
        critic_lr=1e-3,
        tau=0.01,
        cql_alpha=0.5,
        cql_num_samples=4,
        actor_update_frequency=1,
        gradient_clip=1.0,
        device="cpu",
    )
    trainer = DecisionMambaCQLTrainer(actor, critic, trainer_cfg)

    metrics = trainer.step(batch)
    assert "critic_loss" in metrics
    assert "actor_loss" in metrics
    for value in metrics.values():
        assert torch.isfinite(torch.tensor(value)), f"Non-finite metric detected: {metrics}"
