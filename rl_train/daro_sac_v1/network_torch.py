# network_torch.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def mlp(sizes, activation=nn.ReLU, out_activation=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i+1])]
        if i < len(sizes) - 2:
            layers += [activation()]
        elif out_activation is not None:
            layers += [out_activation()]
    net = nn.Sequential(*layers)
    # init
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
    return net


class QNetwork(nn.Module):
    """Single Q(s,a). Input is [state, action] concatenated."""
    def __init__(self, input_size: int, output_size: int = 1):
        super().__init__()
        assert output_size == 1, "Q network outputs scalar Q"
        self.q = mlp([input_size, 128, 64, 1])
        last = [m for m in self.q if isinstance(m, nn.Linear)][-1]
        nn.init.uniform_(last.weight, -1e-3, 1e-3)
        nn.init.zeros_(last.bias)

    def forward(self, state_action: torch.Tensor) -> torch.Tensor:
        return self.q(state_action)


class TwinQ(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.q1 = QNetwork(state_dim + action_dim, 1)
        self.q2 = QNetwork(state_dim + action_dim, 1)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa), self.q2(sa)


class PolicyNetwork(nn.Module):
    """
    Tanh-squashed Gaussian policy:
      - forward(): returns (mu, log_std)
      - sample_action(s): returns (a_squashed, log_prob)
      - act(s, deterministic): returns action in [-1,1]
    """
    def __init__(self, input_size: int, action_size: int, log_std_bounds=(-20.0, 2.0)):
        super().__init__()
        self.net = mlp([input_size, 128, 64])
        self.mu = nn.Linear(64, action_size)
        self.log_std = nn.Linear(64, action_size)
        self.log_std_min, self.log_std_max = log_std_bounds

        nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.mu.bias)
        nn.init.uniform_(self.log_std.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.log_std.bias)

    def forward(self, state):
        h = self.net(state)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(self.log_std_min, self.log_std_max)
        return mu, log_std

    @staticmethod
    def _tanh_correction(u: torch.Tensor) -> torch.Tensor:
        # log |det d(tanh(u))/du| = sum log(1 - tanh(u)^2)
        eps = 1e-6
        return torch.sum(torch.log(1 - torch.tanh(u)**2 + eps), dim=-1, keepdim=True)

    def sample_action(self, state):
        mu, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        u = dist.rsample()                          # reparameterized sample
        a = torch.tanh(u)                           # squash to [-1,1]
        logp_u = torch.sum(dist.log_prob(u), dim=-1, keepdim=True)
        logp_a = logp_u - self._tanh_correction(u) # corrected log-prob
        return a, logp_a.squeeze(-1)                # shape: (B, A), (B,)

    @torch.no_grad()
    def act(self, state, deterministic: bool = False):
        mu, log_std = self.forward(state)
        if deterministic:
            u = mu
        else:
            std = torch.exp(log_std)
            u = torch.distributions.Normal(mu, std).sample()
        return torch.tanh(u)
