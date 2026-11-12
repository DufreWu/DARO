"""
network_torch.py
----------------
PyTorch neural network definitions for the DARO SAC agent.

Architecture (paper §IV-B-3):
  - Critic : two Q-networks (clipped double-Q), each with a main network
             parameterised by θ₁/θ₂ and a soft-update target network.
  - Actor  : one stochastic policy network parameterised by φ that
             outputs a Gaussian distribution over actions.

State vector  s(t) = [v_obs, f_obs, SOC, SOH, thr_1…N, QoS_1…N]
Action vector a(t) = [v_cmd, f_cmd]  (motor speed + CPU frequency)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -5
LOG_STD_MAX = 2
EPSILON = 1e-6


# ---------------------------------------------------------------------------
# Shared MLP backbone
# ---------------------------------------------------------------------------

def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    """Build a fully-connected network from a list of layer sizes."""
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Critic (Q-network)
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """
    Single Q-network: Q(s, a) → scalar.

    Paper uses two hidden layers of 256 and 256 neurons (§V-A).
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_sizes=(256, 256)):
        super().__init__()
        self.net = mlp(
            [state_dim + action_dim] + list(hidden_sizes) + [1]
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class TwinQNetwork(nn.Module):
    """
    Clipped double-Q critic: maintains two independent Q-networks and
    their Polyak-averaged target networks (paper §IV-B-3).
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_sizes=(256, 256), polyak: float = 0.995):
        super().__init__()
        self.polyak = polyak

        self.q1 = QNetwork(state_dim, action_dim, hidden_sizes)
        self.q2 = QNetwork(state_dim, action_dim, hidden_sizes)

        # Target networks — identical architecture, initialised from mains
        self.q1_target = QNetwork(state_dim, action_dim, hidden_sizes)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_sizes)
        self._hard_update()

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """Return Q1(s,a) and Q2(s,a)."""
        return self.q1(state, action), self.q2(state, action)

    def min_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """min(Q1, Q2) — used in actor update to reduce overestimation."""
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)

    def target_min_q(self, state: torch.Tensor,
                     action: torch.Tensor) -> torch.Tensor:
        """min(Q1_target, Q2_target) — Bellman target."""
        q1t = self.q1_target(state, action)
        q2t = self.q2_target(state, action)
        return torch.min(q1t, q2t)

    # ------------------------------------------------------------------
    # Network updates
    # ------------------------------------------------------------------

    @torch.no_grad()
    def soft_update(self):
        """Polyak averaging: θ_target ← τ·θ_target + (1-τ)·θ  (§IV-B-3)."""
        tau = 1.0 - self.polyak
        for p, p_t in zip(self.q1.parameters(), self.q1_target.parameters()):
            p_t.data.mul_(self.polyak).add_(tau * p.data)
        for p, p_t in zip(self.q2.parameters(), self.q2_target.parameters()):
            p_t.data.mul_(self.polyak).add_(tau * p.data)

    @torch.no_grad()
    def _hard_update(self):
        """Initialise targets as exact copies of the main networks."""
        for p, p_t in zip(self.q1.parameters(), self.q1_target.parameters()):
            p_t.data.copy_(p.data)
        for p, p_t in zip(self.q2.parameters(), self.q2_target.parameters()):
            p_t.data.copy_(p.data)


# ---------------------------------------------------------------------------
# Actor (stochastic policy)
# ---------------------------------------------------------------------------

class GaussianActor(nn.Module):
    """
    Stochastic Gaussian policy π_φ(a|s).

    Outputs mean and log_std of a squashed Normal distribution.
    The reparameterisation trick is used during training (§IV-B-3, Fig. 8).

    Action bounds are enforced via tanh squashing followed by affine
    rescaling into [action_low, action_high].
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_sizes=(128, 64),
                 action_low=None, action_high=None,
                 polyak: float = 0.995):
        super().__init__()
        self.polyak = polyak
        self.action_dim = action_dim

        # Shared trunk
        trunk_sizes = [state_dim] + list(hidden_sizes)
        self.trunk = mlp(trunk_sizes)

        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_sizes[-1], action_dim)

        # Action rescaling buffers
        if action_low is None:
            action_low = -torch.ones(action_dim)
        if action_high is None:
            action_high = torch.ones(action_dim)

        action_low = torch.as_tensor(action_low, dtype=torch.float32)
        action_high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

        # Target network (Polyak-averaged copy)
        self.target = None  # built lazily via build_target()

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def _trunk_features(self, state: torch.Tensor) -> torch.Tensor:
        x = state
        for layer in self.trunk:
            x = layer(x)
        return x

    def _distribution(self, state: torch.Tensor):
        """Return a Normal distribution before squashing."""
        feat = self._trunk_features(state)
        mean = self.mean_head(feat)
        log_std = self.log_std_head(feat).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        return Normal(mean, std)

    def forward(self, state: torch.Tensor):
        """
        Sample action and compute log-probability.

        Returns
        -------
        action    : rescaled action in [action_low, action_high]
        log_prob  : log π(a|s) with tanh-correction applied
        mean_action : deterministic action (for evaluation)
        """
        dist = self._distribution(state)
        x_t = dist.rsample()                     # reparameterised sample
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        # Log-prob with Jacobian correction for tanh squashing
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + EPSILON)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean_action = torch.tanh(dist.mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

    @torch.no_grad()
    def act(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Single-observation action selection (no gradient needed)."""
        action, _, mean_action = self.forward(state)
        return mean_action if deterministic else action

    # ------------------------------------------------------------------
    # Target network
    # ------------------------------------------------------------------

    def build_target(self):
        """Construct and hard-initialise the target actor network."""
        import copy
        self.target = copy.deepcopy(self)
        self.target.target = None   # avoid recursive copy
        for p in self.target.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def soft_update(self):
        """Polyak update of target actor (φ_target ← τ·φ_target + (1-τ)·φ)."""
        if self.target is None:
            return
        tau = 1.0 - self.polyak
        for p, p_t in zip(self.parameters(), self.target.parameters()):
            p_t.data.mul_(self.polyak).add_(tau * p.data)