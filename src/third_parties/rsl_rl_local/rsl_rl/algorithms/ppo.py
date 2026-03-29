# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage
# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from torch.distributions import Normal


class ActorCriticLSTM(nn.Module):
    """Actor-Critic with LSTM for both actor and critic.

    Replaces the MLP-based ActorCritic from rsl_rl.modules.
    Drop-in compatible with the PPO class — satisfies all method/attribute
    contracts that ppo.py depends on:

        act(), evaluate(), get_actions_log_prob(), get_hidden_states(),
        reset(), entropy, action_mean, action_std, is_recurrent
    """

    is_recurrent: bool = True  # Tells PPO to use recurrent_mini_batch_generator

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        # LSTM config
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 1,
        # MLP heads on top of LSTM output
        actor_hidden_dims: list[int] = [128, 64],
        critic_hidden_dims: list[int] = [128, 64],
        activation: str = "elu",
        # Action distribution init
        init_noise_std: float = 1.0,
    ):
        super().__init__()

        activation_fn = _get_activation(activation)

        # ----------------------------------------------------------------
        # ACTOR: LSTM backbone + MLP head
        # ----------------------------------------------------------------
        self.actor_lstm = nn.LSTM(
            input_size=num_actor_obs,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,  # input shape: (batch, seq, features)
        )

        actor_layers = []
        in_size = lstm_hidden_size
        for out_size in actor_hidden_dims:
            actor_layers += [nn.Linear(in_size, out_size), activation_fn()]
            in_size = out_size
        actor_layers.append(nn.Linear(in_size, num_actions))
        self.actor_head = nn.Sequential(*actor_layers)

        # ----------------------------------------------------------------
        # CRITIC: LSTM backbone + MLP head
        #
        # TIP: If your critic receives *privileged* (full-state) observations
        #      replace the LSTM with a plain MLP:
        #
        #      self.critic_lstm = None
        #      self.critic_mlp  = nn.Sequential(nn.Linear(num_critic_obs, 256),
        #                                        ELU(), nn.Linear(256, 128),
        #                                        ELU(), nn.Linear(128, 1))
        #
        # and update evaluate() accordingly.
        # ----------------------------------------------------------------
        self.critic_lstm = nn.LSTM(
            input_size=num_critic_obs,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

        critic_layers = []
        in_size = lstm_hidden_size
        for out_size in critic_hidden_dims:
            critic_layers += [nn.Linear(in_size, out_size), activation_fn()]
            in_size = out_size
        critic_layers.append(nn.Linear(in_size, 1))
        self.critic_head = nn.Sequential(*critic_layers)

        # ----------------------------------------------------------------
        # Action distribution parameters
        # ----------------------------------------------------------------
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None

        # ----------------------------------------------------------------
        # Hidden state storage  (num_layers, num_envs, hidden_size)
        # Initialised lazily in _init_hidden() on first forward pass.
        # ----------------------------------------------------------------
        self._actor_hidden: tuple[torch.Tensor, torch.Tensor] | None = None
        self._critic_hidden: tuple[torch.Tensor, torch.Tensor] | None = None
        self._num_layers = lstm_num_layers
        self._hidden_size = lstm_hidden_size
        self._device = "cpu"

        # Cached distribution statistics (set by act / evaluate)
        self.action_mean: torch.Tensor = None
        self.action_std: torch.Tensor = None
        self.entropy: torch.Tensor = None

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_hidden(self, batch_size: int, device: torch.device):
        """Return a zeroed (h, c) pair for the given batch size."""
        h = torch.zeros(self._num_layers, batch_size, self._hidden_size, device=device)
        c = torch.zeros(self._num_layers, batch_size, self._hidden_size, device=device)
        return h, c

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        # Track device so we can init hidden states on the right device
        if args:
            self._device = args[0]
        return result

    # ------------------------------------------------------------------
    # PPO interface — act()
    # ------------------------------------------------------------------

    def act(
        self,
        observations: torch.Tensor,
        hidden_states: tuple | None = None,
        masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample actions from the current policy.

        Args:
            observations: shape (batch, num_actor_obs)  — single time-step
            hidden_states: optional tuple of (actor_hc, critic_hc) from
                           get_hidden_states(); used during PPO update to
                           restore collection-time context.
            masks: episode masks for resetting hidden state at boundaries
                   shape (batch,); 1 = alive, 0 = episode just ended.

        Returns:
            Sampled actions, shape (batch, num_actions).
        """
        device = observations.device
        batch_size = observations.shape[0]

        # Restore or initialise actor hidden state
        if hidden_states is not None:
            actor_hc = hidden_states[0]
        else:
            if self._actor_hidden is None or self._actor_hidden[0].shape[1] != batch_size:
                self._actor_hidden = self._init_hidden(batch_size, device)
            actor_hc = self._actor_hidden

        # Apply episode masks to reset finished environments
        if masks is not None:
            actor_hc = _apply_masks(actor_hc, masks)

        # LSTM forward — add seq_len=1 dimension
        obs_seq = observations.unsqueeze(1)  # (batch, 1, obs_dim)
        lstm_out, actor_hc = self.actor_lstm(obs_seq, actor_hc)
        lstm_out = lstm_out.squeeze(1)  # (batch, hidden_size)

        # Store updated hidden state for next step
        self._actor_hidden = actor_hc

        # Compute action mean via MLP head
        mean = self.actor_head(lstm_out)
        std = self.std.expand_as(mean)

        # Build distribution and sample
        self.distribution = Normal(mean, std)
        self.action_mean = mean
        self.action_std = std

        return self.distribution.sample()

    # ------------------------------------------------------------------
    # PPO interface — evaluate()
    # ------------------------------------------------------------------

    def evaluate(
        self,
        critic_observations: torch.Tensor,
        hidden_states: tuple | None = None,
        masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute state values V(s).

        Args:
            critic_observations: shape (batch, num_critic_obs)
            hidden_states: optional tuple (actor_hc, critic_hc)
            masks: episode masks, shape (batch,)

        Returns:
            Values, shape (batch, 1).
        """
        device = critic_observations.device
        batch_size = critic_observations.shape[0]

        if hidden_states is not None:
            critic_hc = hidden_states[1]
        else:
            if self._critic_hidden is None or self._critic_hidden[0].shape[1] != batch_size:
                self._critic_hidden = self._init_hidden(batch_size, device)
            critic_hc = self._critic_hidden

        if masks is not None:
            critic_hc = _apply_masks(critic_hc, masks)

        obs_seq = critic_observations.unsqueeze(1)  # (batch, 1, obs_dim)
        lstm_out, critic_hc = self.critic_lstm(obs_seq, critic_hc)
        lstm_out = lstm_out.squeeze(1)

        self._critic_hidden = critic_hc

        return self.critic_head(lstm_out)

    # ------------------------------------------------------------------
    # PPO interface — distribution utilities
    # ------------------------------------------------------------------

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Log-probability of *actions* under the last computed distribution."""
        assert self.distribution is not None, "Call act() before get_actions_log_prob()"
        log_prob = self.distribution.log_prob(actions).sum(dim=-1)
        self.entropy = self.distribution.entropy().sum(dim=-1)
        return log_prob

    # ------------------------------------------------------------------
    # PPO interface — recurrent state management
    # ------------------------------------------------------------------

    def get_hidden_states(self) -> tuple:
        """Return a snapshot of (actor_hc, critic_hc) for storage.

        Called by PPO.act() before every environment step so the rollout
        buffer can record the hidden state at collection time — needed to
        reproduce the same LSTM context during the update phase.
        """
        return (self._actor_hidden, self._critic_hidden)

    def reset(self, dones: torch.Tensor | None = None):
        """Reset hidden states for finished environments.

        Args:
            dones: bool tensor, shape (num_envs,). True = episode just ended.
                   If None, reset all environments.
        """
        if dones is None:
            self._actor_hidden = None
            self._critic_hidden = None
            return

        if self._actor_hidden is not None:
            self._actor_hidden = _apply_masks(self._actor_hidden, ~dones.bool())
        if self._critic_hidden is not None:
            self._critic_hidden = _apply_masks(self._critic_hidden, ~dones.bool())

    # ------------------------------------------------------------------
    # Train / test mode helpers (called by PPO.train_mode / test_mode)
    # ------------------------------------------------------------------

    def train(self, mode: bool = True):
        super().train(mode)
        return self

    def test(self):
        self.eval()


# ======================================================================
# Utility functions
# ======================================================================

def _apply_masks(
    hc: tuple[torch.Tensor, torch.Tensor],
    masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero out hidden/cell states for environments where mask == 0.

    Args:
        hc:    (h, c) tensors of shape (num_layers, batch, hidden)
        masks: 1-D float/bool tensor of shape (batch,)
               1 (or True)  = keep state
               0 (or False) = reset to zero
    """
    h, c = hc
    m = masks.float().view(1, -1, 1)  # broadcast over layers & hidden dim
    return h * m, c * m


def _get_activation(name: str) -> type:
    activations = {
        "elu":    nn.ELU,
        "relu":   nn.ReLU,
        "tanh":   nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "selu":   nn.SELU,
        "leaky_relu": nn.LeakyReLU,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation '{name}'. Choose from: {list(activations)}")
    return activations[name]

class PPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""
    # actor_critic = ActorCriticLSTM(
    #     num_actor_obs=48,        # your obs dim
    #     num_critic_obs=48,       # can differ if using privileged obs
    #     num_actions=12,          # your action dim
    #     lstm_hidden_size=256,
    #     lstm_num_layers=1,
    #     actor_hidden_dims=[128, 64],
    #     critic_hidden_dims=[128, 64],
    # )
    actor_critic: ActorCritic
    """The actor critic module."""

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        # Create optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        # Create rollout storage
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        # create rollout storage
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            None,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        # compute value for the last step
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0

        # generator for mini batches
        if self.actor_critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        for (
            observations,
            critic_observations,
            sampled_actions,
            value_targets,
            advantage_estimates,
            discounted_returns,
            prev_log_probs,
            prev_mean_actions,
            prev_action_stds,
            hidden_states,
            episode_masks,
            _,  # rnd_state_batch - not used anymore
        ) in generator:
            # TODO ----- START -----
            # Implement the PPO update step

            # ----------------------------------------------------------------
            # 1. Re-evaluate the sampled actions under the CURRENT policy
            # ----------------------------------------------------------------
            # For recurrent networks, restore the hidden state that was active
            # at collection time so the LSTM/GRU sees the same context.
            if self.actor_critic.is_recurrent:
                self.actor_critic.act(observations, hidden_states=hidden_states, masks=episode_masks)
            else:
                self.actor_critic.act(observations)

            # log π_θ(a|s) for the actions that were actually collected
            current_log_probs = self.actor_critic.get_actions_log_prob(sampled_actions)
            # V_θ(s) from the critic
            current_values = self.actor_critic.evaluate(critic_observations)
            # H[π_θ(·|s)] – entropy bonus encourages exploration
            entropy = self.actor_critic.entropy  # shape [B]

            # ----------------------------------------------------------------
            # 2. Squeeze stored tensors from [B, 1] → [B] so shapes are
            #    consistent with the [B]-shaped outputs above.
            # ----------------------------------------------------------------
            prev_log_probs = prev_log_probs.squeeze(-1)
            advantage_estimates = advantage_estimates.squeeze(-1)
            discounted_returns = discounted_returns.squeeze(-1)
            value_targets = value_targets.squeeze(-1)
            current_values = current_values.squeeze(-1)

            # ----------------------------------------------------------------
            # 3. Optionally normalize advantages within this mini-batch
            # ----------------------------------------------------------------
            if self.normalize_advantage_per_mini_batch:
                advantage_estimates = (advantage_estimates - advantage_estimates.mean()) / (
                    advantage_estimates.std() + 1e-8
                )

            # ----------------------------------------------------------------
            # 4. Adaptive learning-rate schedule (KL-divergence based)
            # ----------------------------------------------------------------
            # KL(old || new) for a diagonal Gaussian:
            #   KL = log(σ_new/σ_old) + (σ_old² + (μ_old-μ_new)²)/(2σ_new²) - 0.5
            if self.schedule == "adaptive" and self.desired_kl is not None and self.desired_kl > 0.0:
                with torch.no_grad():
                    kl = torch.sum(
                        torch.log(self.actor_critic.action_std / (prev_action_stds + 1.0e-8) + 1.0e-5)
                        + (
                            prev_action_stds.pow(2)
                            + (prev_mean_actions - self.actor_critic.action_mean).pow(2)
                        )
                        / (2.0 * self.actor_critic.action_std.pow(2))
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = kl.mean()

                if kl_mean > 2.0 * self.desired_kl:
                    self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                elif 0.0 < kl_mean < 0.5 * self.desired_kl:
                    self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)

                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate

            # ----------------------------------------------------------------
            # 5. Surrogate (policy / actor) loss  — PPO-Clip
            # ----------------------------------------------------------------
            # Importance-sampling ratio  r_t(θ) = π_θ(a|s) / π_θ_old(a|s)
            ratio = torch.exp(current_log_probs - prev_log_probs)

            # L_CLIP = E[ min(r·A, clip(r, 1-ε, 1+ε)·A) ]
            # We *minimise* the negative of this, hence the minus sign.
            surrogate_unclipped = ratio * advantage_estimates
            surrogate_clipped = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage_estimates
            surrogate_loss = -torch.min(surrogate_unclipped, surrogate_clipped).mean()

            # ----------------------------------------------------------------
            # 6. Value (critic) loss
            # ----------------------------------------------------------------
            if self.use_clipped_value_loss:
                # Clipped value loss prevents the value estimate from drifting
                # too far from the old estimate in a single update step.
                value_pred_clipped = value_targets + (current_values - value_targets).clamp(
                    -self.clip_param, self.clip_param
                )
                value_loss_unclipped = (current_values - discounted_returns).pow(2)
                value_loss_clipped = (value_pred_clipped - discounted_returns).pow(2)
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
            else:
                value_loss = (discounted_returns - current_values).pow(2).mean()

            # ----------------------------------------------------------------
            # 7. Combined loss and gradient step
            # ----------------------------------------------------------------
            # L = L_CLIP + c1·L_VF - c2·H[π]
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            # TODO ----- END -----

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        # Clear the storage
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_entropy