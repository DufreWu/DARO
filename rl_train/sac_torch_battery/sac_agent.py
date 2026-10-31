import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from network_torch import PolicyNetwork, TwinQ

# ---- Action grids you already have ----
cpu_fre_list = cpu_fre_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
speed_list   = [2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]

def _nearest(val: float, choices: List[float]) -> float:
    return min(choices, key=lambda c: abs(c - val))

def _affine_from_tanh(a_tanh: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    # map a in [-1,1] -> [lo, hi]
    return (a_tanh + 1.0) * 0.5 * (hi - lo) + lo

@dataclass
class SACCfg:
    state_dim: int
    action_dim: int = 2  # [f_cpu, speed] as continuous
    gamma: float = 0.99
    tau: float = 0.05
    actor_lr: float = 0.005    # 3e-4
    critic_lr: float = 0.005    # 3e-4
    alpha_lr: float = 0.005    # 3e-4
    init_alpha: float = 0.2
    target_entropy: float = None  # default to -action_dim if None
    batch_size: int = 64
    buffer_size: int = 500
    train_start: int = 100
    device: str = "cpu"  # set "cuda" if you have GPU


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.s = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.a = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.r = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.s2 = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.d = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

    def add(self, s, a, r, s2, d):
        i = self.ptr
        self.s[i]  = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        self.a[i]  = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        self.r[i]  = torch.as_tensor([r], dtype=torch.float32, device=self.device)
        self.s2[i] = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        self.d[i]  = torch.as_tensor([d], dtype=torch.float32, device=self.device)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]


class SACAgent:
    def __init__(self, cfg: SACCfg,
                 act_low: Tuple[float, float],
                 act_high: Tuple[float, float]):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # Continuous action ranges for scaling (Hz and m/s)
        self.act_low  = torch.tensor(act_low,  dtype=torch.float32, device=self.device)  # (f_min, v_min)
        self.act_high = torch.tensor(act_high, dtype=torch.float32, device=self.device)  # (f_max, v_max)

        # Networks
        self.actor = PolicyNetwork(cfg.state_dim, cfg.action_dim).to(self.device)
        self.critic = TwinQ(cfg.state_dim, cfg.action_dim).to(self.device)
        self.critic_tgt = TwinQ(cfg.state_dim, cfg.action_dim).to(self.device)
        self.critic_tgt.load_state_dict(self.critic.state_dict())
        for p in self.critic_tgt.parameters():
            p.requires_grad_(False)

        # Optims
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        # Entropy (auto alpha)
        if cfg.target_entropy is None:
            cfg.target_entropy = -float(cfg.action_dim)
        self.log_alpha = torch.tensor(math.log(cfg.init_alpha), dtype=torch.float32,
                                      device=self.device, requires_grad=True)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=cfg.alpha_lr)

        # Buffer
        self.buf = ReplayBuffer(cfg.buffer_size, cfg.state_dim, cfg.action_dim, self.device)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, state_np: np.ndarray, deterministic=False) -> Tuple[np.ndarray, Tuple[float,float]]:
        s = torch.as_tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        a_tanh = self.actor.act(s, deterministic=deterministic)   # in [-1,1]
        a_cont = _affine_from_tanh(a_tanh, self.act_low, self.act_high).squeeze(0)  # (2,)
        # Quantize to nearest valid choices for env execution
        f_cmd = _nearest(float(a_cont[0].item()), cpu_fre_list)
        # v_cmd = _nearest(float(a_cont[1].item()), speed_list)
        v_cmd = float(a_cont[1].item())
        print(f"select_action()-f_cmd = {f_cmd}, v_cmd = {v_cmd}")
        return a_cont.cpu().numpy(), (f_cmd, v_cmd)

    def store(self, s, a_cont, r, s2, d):
        # a_cont is the continuous (pre-quantized) action used by critics
        self.buf.add(s, a_cont, r, s2, float(d))

    def update(self):
        if self.buf.size < self.cfg.train_start:
            return {}

        s, a, r, s2, d = self.buf.sample(self.cfg.batch_size)

        # --- Critic update ---
        with torch.no_grad():
            a2_tanh, logp_a2 = self.actor.sample_action(s2)            # [-1,1]
            a2 = _affine_from_tanh(a2_tanh, self.act_low, self.act_high)
            q1_t, q2_t = self.critic_tgt(s2, a2)
            q_t_min = torch.min(q1_t, q2_t) - self.alpha * logp_a2.unsqueeze(-1)
            y = r + (1.0 - d) * self.cfg.gamma * q_t_min

        q1, q2 = self.critic(s, a)
        critic_loss = ((q1 - y)**2 + (q2 - y)**2).mean()

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # --- Actor update ---
        a_pi_tanh, logp_a_pi = self.actor.sample_action(s)
        a_pi = _affine_from_tanh(a_pi_tanh, self.act_low, self.act_high)
        q1_pi, q2_pi = self.critic(s, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * logp_a_pi.unsqueeze(-1) - q_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # --- Alpha update (auto entropy) ---
        alpha_loss = -(self.log_alpha * (logp_a_pi.detach() + self.cfg.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        # --- Target soft update ---
        with torch.no_grad():
            for p_t, p in zip(self.critic_tgt.parameters(), self.critic.parameters()):
                p_t.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
        }

    def save(self, path: str):
        """
        Save the SAC agent's models in TorchScript format for C++ compatibility.

        Args:
            path (str): The base file path to save the models (e.g., 'sac_agent').
        """
        # Convert models to TorchScript
        actor_scripted = torch.jit.script(self.actor)
        critic_scripted = torch.jit.script(self.critic)
        critic_tgt_scripted = torch.jit.script(self.critic_tgt)

        # Save the scripted models
        actor_scripted.save(f"{path}/sac_policy.pt")
        critic_scripted.save(f"{path}/sac_critic.pt")
        critic_tgt_scripted.save(f"{path}/sac_alpha.pt")

        print(f"Models saved for C++ at {path}_actor.pt, {path}_critic.pt, and {path}_critic_tgt.pt")

