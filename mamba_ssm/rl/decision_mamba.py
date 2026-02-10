"""Decision Mamba policy and Conservative Q-Learning trainer.

This module implements an offline RL stack that uses a Mamba block as the actor (policy)
within an actor-critic architecture. It supports the twin-critic variant with Conservative
Q-Learning (CQL) regularisation and delayed policy updates, as motivated by the paper
summarised in the user request.
"""
from __future__ import annotations

import math
import copy
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

try:
    from mamba_ssm.modules.mamba_simple import Mamba as _Mamba
    _USING_FALLBACK_MAMBA = False
except Exception:  # pragma: no cover - triggered when CUDA extensions are unavailable.
    import warnings

    class _FallbackMamba(nn.Module):
        """CPU-friendly fallback used when the CUDA selective scan kernel is unavailable."""

        def __init__(self, d_model: int, **_: Any) -> None:
            super().__init__()
            self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, batch_first=True)

        def forward(self, hidden_states: torch.Tensor, inference_params: Any = None) -> torch.Tensor:
            outputs, _ = self.gru(hidden_states)
            return outputs

        def allocate_inference_cache(self, batch_size: int, max_seqlen: int, **kwargs: Any) -> Any:
            del batch_size, max_seqlen, kwargs
            return None

    warnings.warn(
        "Falling back to a GRU-based Decision Mamba implementation because the CUDA selective "
        "scan extension could not be imported. Install the compiled ops for full performance.",
        RuntimeWarning,
    )
    _Mamba = _FallbackMamba
    _USING_FALLBACK_MAMBA = True

Mamba = _Mamba


@dataclass
class DecisionMambaConfig:
    """Configuration for the Decision Mamba actor.

    Parameters
    ----------
    state_dim: int
        Dimensionality of the environment state vector per time-step.
    action_dim: int
        Number of discrete actions produced by the policy network.
    rtg_dim: int, default=1
        Dimensionality of the return-to-go token fed to the encoder. For the
        single-step scenario described in the paper this defaults to 1 (the
        immediate reward).
    d_model: int, default=256
        Width of the Decision Mamba hidden representation.
    max_seq_len: int, default=128
        Maximum context length (number of tokens) supported by the policy.
    include_action_context: bool, default=False
        Whether previous actions are fed back into the encoder as auxiliary
        tokens. When enabled, ``action_context_dim`` must be specified at actor
        construction time.
    """

    state_dim: int
    action_dim: int
    rtg_dim: int = 1
    d_model: int = 256
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: str | int = "auto"
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    conv_bias: bool = True
    bias: bool = False
    use_fast_path: bool = True
    max_seq_len: int = 128
    include_action_context: bool = False

    def mamba_kwargs(self) -> Dict[str, Any]:
        return {
            "d_model": self.d_model,
            "d_state": self.d_state,
            "d_conv": self.d_conv,
            "expand": self.expand,
            "dt_rank": self.dt_rank,
            "dt_min": self.dt_min,
            "dt_max": self.dt_max,
            "dt_scale": self.dt_scale,
            "dt_init_floor": self.dt_init_floor,
            "conv_bias": self.conv_bias,
            "bias": self.bias,
            "use_fast_path": self.use_fast_path,
        }


class DecisionMambaActor(nn.Module):
    """Actor network that embeds (RTG, state, [previous action]) tokens and feeds them
    through a Mamba block to produce the action distribution."""

    def __init__(
        self,
        config: DecisionMambaConfig,
        *,
        action_context_dim: Optional[int] = None,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.config = config
        self.state_proj = nn.Linear(config.state_dim, config.d_model, **factory_kwargs)
        self.rtg_proj = nn.Linear(config.rtg_dim, config.d_model, **factory_kwargs)

        if config.include_action_context:
            if action_context_dim is None:
                raise ValueError(
                    "action_context_dim must be provided when include_action_context is True"
                )
            self.action_proj = nn.Linear(action_context_dim, config.d_model, **factory_kwargs)
        else:
            self.action_proj = None

        self.positional_emb = nn.Embedding(config.max_seq_len, config.d_model, **factory_kwargs)
        nn.init.normal_(self.positional_emb.weight, std=0.02)

        self.mamba = Mamba(layer_idx=0, **config.mamba_kwargs(), **factory_kwargs)
        self.norm = nn.LayerNorm(config.d_model, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.policy_head = nn.Linear(config.d_model, config.action_dim, **factory_kwargs)

    def forward(
        self,
        state_seq: torch.Tensor,
        rtg_seq: torch.Tensor,
        *,
        action_context_seq: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_sequence: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Run the actor on a batch of sequences.

        Parameters
        ----------
        state_seq: tensor of shape (B, T, state_dim)
            Environment states for each timestep in the context window.
        rtg_seq: tensor of shape (B, T, rtg_dim)
            Return-to-go tokens (or immediate rewards in the single-step case).
        action_context_seq: optional tensor of shape (B, T, action_context_dim)
            Previous actions to feed back into the encoder.
        mask: optional tensor of shape (B, T)
            Binary mask indicating valid timesteps (1 for real tokens, 0 for padded).
        return_sequence: bool
            If True, return the entire hidden sequence alongside the policy logits.
        """

        if state_seq.dim() != 3:
            raise ValueError("state_seq must be a 3D tensor (batch, seq_len, state_dim)")
        if rtg_seq.shape[:2] != state_seq.shape[:2]:
            raise ValueError("rtg_seq must match state_seq on batch and sequence dimensions")
        if mask is not None and mask.shape[:2] != state_seq.shape[:2]:
            raise ValueError("mask must match state_seq on batch and sequence dimensions")
        if action_context_seq is not None and action_context_seq.shape[:2] != state_seq.shape[:2]:
            raise ValueError(
                "action_context_seq must match state_seq on batch and sequence dimensions"
            )

        batch_size, seq_len, _ = state_seq.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds configured max_seq_len={self.config.max_seq_len}"
            )

        tokens = self.state_proj(state_seq) + self.rtg_proj(rtg_seq)
        if self.action_proj is not None:
            if action_context_seq is None:
                raise ValueError("action_context_seq required when action context is enabled")
            tokens = tokens + self.action_proj(action_context_seq)

        position_ids = torch.arange(seq_len, device=tokens.device)
        tokens = tokens + self.positional_emb(position_ids)[None, :, :]

        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1)

        hidden = self.mamba(tokens)
        hidden = self.norm(hidden)
        hidden = self.dropout(hidden)

        if mask is not None:
            valid_lengths = mask.sum(dim=1).clamp(min=1).long()
            last_indices = (valid_lengths - 1).clamp(min=0)
            batch_idx = torch.arange(batch_size, device=hidden.device)
            final_hidden = hidden[batch_idx, last_indices]
        else:
            final_hidden = hidden[:, -1]

        logits = self.policy_head(final_hidden)
        if return_sequence:
            return logits, hidden
        return logits

    def sample(
        self,
        state_seq: torch.Tensor,
        rtg_seq: torch.Tensor,
        *,
        action_context_seq: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Categorical]:
        """Sample an action from the policy distribution."""

        logits = self.forward(
            state_seq,
            rtg_seq,
            action_context_seq=action_context_seq,
            mask=mask,
            return_sequence=False,
        )
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        scaled_logits = logits / temperature
        dist = Categorical(logits=scaled_logits)
        if deterministic:
            action = torch.argmax(dist.logits, dim=-1)
        else:
            action = dist.sample()
        return action, dist


class _QNetwork(nn.Module):
    """Single Q-network used in the twin-critic ensemble."""

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        layers = []
        input_dim = state_dim + action_dim
        last_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        if state.shape[0] != action_onehot.shape[0]:
            raise ValueError("state and action_onehot must have matching batch size")
        x = torch.cat([state, action_onehot], dim=-1)
        return self.net(x)


class TwinQCritic(nn.Module):
    """Twin-critic head that mitigates positive bias."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.q1 = _QNetwork(state_dim, action_dim, hidden_sizes)
        self.q2 = _QNetwork(state_dim, action_dim, hidden_sizes)

    def forward(self, state: torch.Tensor, action_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state, action_onehot), self.q2(state, action_onehot)


@dataclass
class TrainerConfig:
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    tau: float = 0.005
    cql_alpha: float = 1.0
    cql_num_samples: int = 10
    actor_update_frequency: int = 5
    gradient_clip: Optional[float] = None
    device: Optional[torch.device | str] = None

    def resolve_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DecisionMambaCQLTrainer:
    """Offline RL trainer that optimises the Decision Mamba actor with twin critics and CQL."""

    def __init__(
        self,
        actor: DecisionMambaActor,
        critic: TwinQCritic,
        config: TrainerConfig,
    ) -> None:
        self.cfg = config
        self.device = config.resolve_device()

        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        for param in self.target_critic.parameters():
            param.requires_grad = False

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)
        self.total_updates = 0
        self.action_dim = actor.config.action_dim

    def _gather_last_state(self, seq: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return seq[:, -1]
        valid_lengths = mask.sum(dim=1).clamp(min=1).long()
        last_indices = (valid_lengths - 1).clamp(min=0)
        batch_idx = torch.arange(seq.size(0), device=seq.device)
        return seq[batch_idx, last_indices]

    def update_critic(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        state_seq = batch["state_seq"].to(self.device)
        rtg_seq = batch["rtg_seq"].to(self.device)
        reward = batch["reward"].to(self.device).unsqueeze(-1)
        action_idx = batch["action"].to(self.device)
        mask = batch.get("mask")
        mask = mask.to(self.device) if mask is not None else None

        state_last = self._gather_last_state(state_seq, mask)
        action_onehot = F.one_hot(action_idx, num_classes=self.action_dim).float()

        q1, q2 = self.critic(state_last, action_onehot)
        bellman = F.mse_loss(q1, reward) + F.mse_loss(q2, reward)

        action_context_seq = batch.get("action_context_seq")
        if action_context_seq is not None:
            action_context_seq = action_context_seq.to(self.device)

        with torch.no_grad():
            logits = self.actor(
                state_seq,
                rtg_seq,
                action_context_seq=action_context_seq,
                mask=mask,
                return_sequence=False,
            )
            dist = Categorical(logits=logits)
            sampled_actions = dist.sample((self.cfg.cql_num_samples,))
            sampled_onehot = F.one_hot(sampled_actions, num_classes=self.action_dim).float()

        state_rep = state_last.unsqueeze(0).expand(self.cfg.cql_num_samples, -1, -1)
        q1_rand = self.critic.q1(
            state_rep.reshape(-1, state_last.size(-1)),
            sampled_onehot.reshape(-1, self.action_dim),
        ).view(self.cfg.cql_num_samples, -1)
        q2_rand = self.critic.q2(
            state_rep.reshape(-1, state_last.size(-1)),
            sampled_onehot.reshape(-1, self.action_dim),
        ).view(self.cfg.cql_num_samples, -1)

        logsumexp_q1 = torch.logsumexp(q1_rand, dim=0) - math.log(self.cfg.cql_num_samples)
        logsumexp_q2 = torch.logsumexp(q2_rand, dim=0) - math.log(self.cfg.cql_num_samples)
        cql_reg = (logsumexp_q1 - q1.squeeze(-1)) + (logsumexp_q2 - q2.squeeze(-1))
        cql_loss = cql_reg.mean()

        loss = bellman + self.cfg.cql_alpha * cql_loss

        self.critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.gradient_clip)
        self.critic_opt.step()

        return {
            "critic_loss": loss.item(),
            "bellman": bellman.item(),
            "cql": cql_loss.item(),
        }

    def update_actor(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        state_seq = batch["state_seq"].to(self.device)
        rtg_seq = batch["rtg_seq"].to(self.device)
        mask = batch.get("mask")
        mask = mask.to(self.device) if mask is not None else None

        action_context_seq = batch.get("action_context_seq")
        if action_context_seq is not None:
            action_context_seq = action_context_seq.to(self.device)

        logits = self.actor(
            state_seq,
            rtg_seq,
            action_context_seq=action_context_seq,
            mask=mask,
            return_sequence=False,
        )
        dist = Categorical(logits=logits)
        action = dist.sample()
        action_onehot = F.one_hot(action, num_classes=self.action_dim).float()

        state_last = self._gather_last_state(state_seq, mask)
        q1_pi, q2_pi = self.critic(state_last, action_onehot)
        actor_loss = -torch.min(q1_pi, q2_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.cfg.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.gradient_clip)
        self.actor_opt.step()

        return {"actor_loss": actor_loss.item()}

    def soft_update_target(self) -> None:
        with torch.no_grad():
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.lerp_(param.data, self.cfg.tau)

    def step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        metrics = self.update_critic(batch)
        if self.total_updates % self.cfg.actor_update_frequency == 0:
            metrics.update(self.update_actor(batch))
            self.soft_update_target()
        self.total_updates += 1
        return metrics
