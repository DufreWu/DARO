"""
sac_agent.py
------------
Soft Actor-Critic (SAC) agent for DARO (§IV-B-3, Fig. 8).

Implements:
  - Replay buffer D (ring buffer, uniform sampling)
  - Clipped double-Q critic update (Bellman target, §IV-B-3)
  - KL-regularised actor update (Eq. 14)
  - Automatic entropy tuning (temperature α)
  - Polyak-averaged target networks (soft update)

Hyperparameters follow §V-A:
  lr = 3e-4, γ = 0.99, |D| = 1e5, batch = 128, α_init = 0.2
  two hidden layers: 128, 64 neurons for all networks
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from network_torch import TwinQNetwork, GaussianActor


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Fixed-size FIFO replay buffer D storing experience tuples
    {s_t, a_t, r_t, s_{t+1}, done_t}  (§IV-B-3).

    Uses pre-allocated numpy arrays for memory efficiency.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 capacity: int = 100_000):
        self.capacity = capacity
        self.ptr = 0          # write pointer
        self.size = 0         # current fill level

        self.states      = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.actions     = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards     = np.zeros((capacity, 1),          dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.dones       = np.zeros((capacity, 1),          dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        """Add one transition to the buffer (overwrites oldest when full)."""
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr]       = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        """
        Sample a random minibatch (s, a, r, s′) of size batch_size (step 1
        in Fig. 8).
        """
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states[idx]).to(device),
            torch.FloatTensor(self.actions[idx]).to(device),
            torch.FloatTensor(self.rewards[idx]).to(device),
            torch.FloatTensor(self.next_states[idx]).to(device),
            torch.FloatTensor(self.dones[idx]).to(device),
        )

    def __len__(self) -> int:
        return self.size


# ---------------------------------------------------------------------------
# SAC agent
# ---------------------------------------------------------------------------

class SACAgent:
    """
    SAC agent for DARO energy-efficient control.

    Notation aligns with the paper:
      θ₁, θ₂           — critic main network parameters
      θ̄₁, θ̄₂           — critic target network parameters  (Polyak)
      φ                 — actor main network parameters
      φ̄                 — actor target network parameters   (Polyak)
      α (log_alpha)     — SAC temperature (entropy weight)

    Update order per training step (Fig. 8):
      1. Sample minibatch from D
      2. Compute clipped double-Q Bellman target
      3. Update critic networks (θ₁, θ₂) with MSE loss
      4. Update actor (φ) with KL-regularised Q-loss (Eq. 14)
      5. Update temperature α
      6. Soft-update target networks (Polyak)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low,
        action_high,
        hidden_sizes=(128, 64),
        lr: float = 3e-4,
        gamma: float = 0.99,
        polyak: float = 0.995,
        alpha_init: float = 0.2,
        auto_alpha: bool = True,
        target_entropy: float = None,
        buffer_capacity: int = 100_000,
        batch_size: int = 128,
        device: str = 'auto',
    ):
        # Device selection
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available()
                                       else 'cpu')
        else:
            self.device = torch.device(device)

        self.gamma      = gamma
        self.batch_size = batch_size
        self.auto_alpha = auto_alpha

        # Convert action bounds to tensors
        act_low  = torch.FloatTensor(action_low).to(self.device)
        act_high = torch.FloatTensor(action_high).to(self.device)

        # ---- Networks ----
        self.critic = TwinQNetwork(
            state_dim, action_dim, hidden_sizes, polyak=polyak
        ).to(self.device)

        self.actor = GaussianActor(
            state_dim, action_dim, hidden_sizes,
            action_low=act_low, action_high=act_high,
            polyak=polyak
        ).to(self.device)
        self.actor.build_target()

        # ---- Optimisers ----
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)
        self.actor_optim  = optim.Adam(self.actor.parameters(),  lr=lr)

        # ---- Entropy / temperature α ----
        if target_entropy is None:
            # Heuristic: −dim(A) (Haarnoja et al., [27])
            target_entropy = -float(action_dim)
        self.target_entropy = target_entropy

        self.log_alpha = torch.tensor(
            np.log(alpha_init), dtype=torch.float32,
            requires_grad=True, device=self.device
        )
        self.alpha = alpha_init
        if auto_alpha:
            self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)

        # ---- Replay buffer ----
        self.buffer = ReplayBuffer(state_dim, action_dim, buffer_capacity)

        # ---- Logging ----
        self.total_updates = 0

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray,
                       deterministic: bool = False) -> np.ndarray:
        """Select action for a single observation (no gradient)."""
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor.act(s, deterministic=deterministic)
        return action.cpu().numpy().flatten()

    def store(self, state, action, reward, next_state, done):
        """Push one transition to the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    # SAC update (Fig. 8)
    # ------------------------------------------------------------------

    def update(self) -> dict:
        """
        One gradient update step for critic, actor, and temperature.

        Returns a dict of scalar loss values for logging.
        """
        if len(self.buffer) < self.batch_size:
            return {}

        # Step 1: sample minibatch (s, a, r, s′, done)
        s, a, r, s_next, done = self.buffer.sample(self.batch_size, self.device)

        # ------------------------------------------------------------------
        # Step 2: Bellman target  y = r + γ(1-done) · [min Q_target(s′,a′) - α·log π(a′|s′)]
        # ------------------------------------------------------------------
        with torch.no_grad():
            a_next, log_pi_next, _ = self.actor(s_next)
            q_target_next = self.critic.target_min_q(s_next, a_next)
            y = r + self.gamma * (1.0 - done) * (q_target_next - self.alpha * log_pi_next)

        # ------------------------------------------------------------------
        # Step 3: Critic update  MSE( Q_i(s,a), y )  i=1,2
        # ------------------------------------------------------------------
        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # ------------------------------------------------------------------
        # Step 4: Actor update  J_π = E[ α·log π(a|s) - min Q(s,a) ]  (Eq. 14)
        # ------------------------------------------------------------------
        a_new, log_pi, _ = self.actor(s)
        min_q = self.critic.min_q(s, a_new)
        actor_loss = (self.alpha * log_pi - min_q).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # ------------------------------------------------------------------
        # Step 5: Temperature update (automatic entropy tuning)
        # ------------------------------------------------------------------
        alpha_loss = torch.tensor(0.0)
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = float(self.log_alpha.exp())

        # ------------------------------------------------------------------
        # Step 6: Soft update target networks (Polyak)
        # ------------------------------------------------------------------
        self.critic.soft_update()
        self.actor.soft_update()

        self.total_updates += 1

        return {
            'critic_loss': float(critic_loss),
            'actor_loss':  float(actor_loss),
            'alpha_loss':  float(alpha_loss),
            'alpha':       self.alpha,
            'mean_q':      float(min_q.mean()),
        }

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save(self, path: str):
        """
        Persist model weights to disk.

        Saves critic, actor, and temperature; optimiser states are included
        to allow resuming training.
        """
        os.makedirs(path, exist_ok=True)
        torch.save({
            'critic_state':        self.critic.state_dict(),
            'actor_state':         self.actor.state_dict(),
            'actor_target_state':  self.actor.target.state_dict(),
            'log_alpha':           self.log_alpha.detach().cpu(),
            'critic_optim':        self.critic_optim.state_dict(),
            'actor_optim':         self.actor_optim.state_dict(),
            'total_updates':       self.total_updates,
        }, os.path.join(path, 'daro_checkpoint.pt'))
        print(f"[DARO] Checkpoint saved → {path}")

    def load(self, path: str, eval_only: bool = False):
        """
        Load model weights from a checkpoint directory.

        Parameters
        ----------
        eval_only : if True, load weights only (no optimiser state);
                    sets networks to eval mode for deployment.
        """
        ckpt_path = os.path.join(path, 'daro_checkpoint.pt')
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.critic.load_state_dict(ckpt['critic_state'])
        self.actor.load_state_dict(ckpt['actor_state'])
        self.actor.target.load_state_dict(ckpt['actor_target_state'])
        self.log_alpha = ckpt['log_alpha'].requires_grad_(not eval_only)
        self.log_alpha = self.log_alpha.to(self.device)
        self.alpha = float(self.log_alpha.exp())
        self.total_updates = ckpt.get('total_updates', 0)

        if not eval_only:
            self.critic_optim.load_state_dict(ckpt['critic_optim'])
            self.actor_optim.load_state_dict(ckpt['actor_optim'])

        if eval_only:
            self.critic.eval()
            self.actor.eval()

        print(f"[DARO] Checkpoint loaded ← {ckpt_path} "
              f"(updates={self.total_updates}, α={self.alpha:.4f})")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def model_size_kb(self) -> dict:
        """
        Report parameter counts and approximate on-disk size.

        Paper reports 85.43 kB per critic and 46.89 kB for the actor (§V-B).
        """
        def _count(model):
            return sum(p.numel() for p in model.parameters())

        def _kb(model):
            return _count(model) * 4 / 1024   # float32 → bytes → kB

        return {
            'critic_params': _count(self.critic),
            'critic_kb':     _kb(self.critic),
            'actor_params':  _count(self.actor),
            'actor_kb':      _kb(self.actor),
        }