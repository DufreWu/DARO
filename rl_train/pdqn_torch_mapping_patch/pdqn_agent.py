import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os


# ========================
# Actor Network
# ========================
class ParamActor(nn.Module):
    def __init__(self, state_dim, action_dim, param_dim):
        super().__init__()
        self.action_dim = action_dim
        self.param_dim = param_dim

        self.fc1 = nn.Linear(state_dim, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 128)
        self.ln2 = nn.LayerNorm(128)

        self.out = nn.Linear(128, action_dim * param_dim)

        nn.init.uniform_(self.out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.out.bias, -3e-3, 3e-3)

    def forward(self, s):
        x = F.relu(self.ln1(self.fc1(s)))
        x = F.relu(self.ln2(self.fc2(x)))

        x = torch.tanh(self.out(x))  # [-1,1]

        return x.view(-1, self.action_dim, self.param_dim)


# ========================
# Q Network (CORRECT PDQN)
# ========================
class QNetwork(nn.Module):
    def __init__(self, state_dim, param_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim + param_dim, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 128)
        self.ln2 = nn.LayerNorm(128)
        self.out = nn.Linear(128, 1)

        nn.init.uniform_(self.out.weight, -3e-4, 3e-4)
        nn.init.uniform_(self.out.bias, -3e-4, 3e-4)

    def forward(self, s, theta):
        x = torch.cat([s, theta], dim=1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.out(x)


# ========================
# Replay Buffer
# ========================
class ReplayBuffer:
    def __init__(self, capacity, state_dim, param_dim, device):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.device = device

        self.s = torch.zeros((capacity, state_dim), device=device)
        self.a = torch.zeros((capacity,), dtype=torch.long, device=device)
        self.theta = torch.zeros((capacity, param_dim), device=device)
        self.r = torch.zeros((capacity,), device=device)
        self.s2 = torch.zeros((capacity, state_dim), device=device)
        self.d = torch.zeros((capacity,), device=device)

    def append(self, s, a, theta, r, s2, d):
        i = self.ptr
        self.s[i] = torch.as_tensor(s, device=self.device)
        self.a[i] = int(a)
        self.theta[i] = torch.as_tensor(theta, device=self.device)
        self.r[i] = float(r)
        self.s2[i] = torch.as_tensor(s2, device=self.device)
        self.d[i] = float(d)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,))
        return (
            self.s[idx],
            self.a[idx],
            self.theta[idx],
            self.r[idx],
            self.s2[idx],
            self.d[idx],
        )


# ========================
# PDQN Agent
# ========================
class PDQNAgent:
    def __init__(self, state_dim, action_dim, param_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.param_dim = param_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        self.actor = ParamActor(state_dim, action_dim, param_dim).to(self.device)
        self.actor_target = ParamActor(state_dim, action_dim, param_dim).to(self.device)

        self.q = QNetwork(state_dim, param_dim).to(self.device)
        self.q_target = QNetwork(state_dim, param_dim).to(self.device)

        # optim
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=0.0002)
        self.q_opt = optim.Adam(self.q.parameters(), lr=0.0005)

        # replay
        self.memory = ReplayBuffer(100000, state_dim, param_dim, self.device)

        # hyperparams
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.train_start = 200

        self.tau = 0.01
        self.param_noise_std = 0.1

        self.hard_update()

    def hard_update(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.q_target.load_state_dict(self.q.state_dict())

    def soft_update(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)

    # ========================
    # Action Selection
    # ========================
    def get_action(self, state, training=True):
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            theta_all = self.actor(s)

            if training:
                theta_all += torch.randn_like(theta_all) * self.param_noise_std

            theta = theta_all.squeeze(0)  # (A, param_dim)

            # evaluate all actions
            s_rep = s.repeat(self.action_dim, 1)
            q_values = self.q(s_rep, theta).squeeze().cpu().numpy()

        if training and random.random() < self.epsilon:
            a = random.randrange(self.action_dim)
        else:
            a = np.argmax(q_values)

        return a, theta[a].cpu().numpy()

    # ========================
    # Store transition
    # ========================
    def store(self, s, a, theta, r, s2, d):
        self.memory.append(s, a, theta, r, s2, d)

    # ========================
    # Training
    # ========================
    def train(self):
        if self.memory.size < self.train_start:
            return

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        s, a, theta, r, s2, d = self.memory.sample(self.batch_size)

        s = s.to(self.device)
        a = a.to(self.device)
        theta = theta.to(self.device)
        r = r.to(self.device)
        s2 = s2.to(self.device)
        d = d.to(self.device)

        B = s.size(0)

        # ===== Q update =====
        current_q = self.q(s, theta).squeeze()

        with torch.no_grad():
            next_theta_all = self.actor_target(s2)  # (B, A, param_dim)

            q_list = []
            for i in range(self.action_dim):
                theta_i = next_theta_all[:, i, :]
                q_i = self.q_target(s2, theta_i)
                q_list.append(q_i)

            q_next = torch.cat(q_list, dim=1)
            max_q_next = q_next.max(dim=1).values

            target_q = r + (1 - d) * self.gamma * max_q_next

        q_loss = F.mse_loss(current_q, target_q)

        self.q_opt.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.q_opt.step()

        # ===== Actor update =====
        theta_all = self.actor(s)

        q_list = []
        for i in range(self.action_dim):
            theta_i = theta_all[:, i, :]
            q_i = self.q(s, theta_i)
            q_list.append(q_i)

        q_all = torch.cat(q_list, dim=1)
        actor_loss = -q_all.max(dim=1).values.mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # ===== Target update =====
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.q_target, self.q)

    # ========================
    # Save / Load
    # ========================
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{path}/actor.pt")
        torch.save(self.q.state_dict(), f"{path}/q.pt")

    def load(self, path):
        self.actor.load_state_dict(torch.load(f"{path}/actor.pt"))
        self.q.load_state_dict(torch.load(f"{path}/q.pt"))
        self.hard_update()