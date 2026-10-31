import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
from network_torch import ParamActor, QNetwork

class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int, action_param_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.s = torch.zeros((capacity, state_dim))
        self.a = torch.zeros((capacity, 1), dtype=torch.long)
        self.a1 = torch.zeros((capacity, action_param_dim))
        self.r = torch.zeros((capacity, 1))
        self.s2 = torch.zeros((capacity, state_dim))
        self.d = torch.zeros((capacity, 1))

    def append(self, s, a, a1, r, s2, d):
        i = self.ptr
        self.s[i]  = torch.as_tensor(s)
        self.a[i]  = int(a)
        self.a1[i] = torch.as_tensor(a1)
        self.r[i]  = float(r)
        self.s2[i] = torch.as_tensor(s2)
        self.d[i]  = float(d)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = torch.randint(0, self.size, (batch_size,))
        return self.s[idx], self.a[idx].squeeze(1), self.a1[idx],self.r[idx].squeeze(1), self.s2[idx], self.d[idx].squeeze(1)

class PDQNAgent:
    def __init__(self, state_size, action_size, action_parameter_size):
        self.state_size = state_size                    # state dim
        self.action_size = action_size                  # number of discrete actions
        self.action_parameter_size = action_parameter_size  # dim of continuous params

        # Hyperparameters
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.95
        self.discount_factor = 0.99
        self.actor_lr = 0.0002
        self.q_lr = 0.001
        self.memory_maxlen = 100000
        self.batch_size = 64
        self.train_start = 200

        # Soft-update factors (use small taus for stability)
        self.tau_q = 0.01
        self.tau_actor = 0.01

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.actor_model = ParamActor(self.state_size, self.action_parameter_size).to(self.device)
        self.target_actor_model = ParamActor(self.state_size, self.action_parameter_size).to(self.device)

        self.q_model = QNetwork(self.state_size, self.action_size, self.action_parameter_size).to(self.device)
        self.target_q_model = QNetwork(self.state_size, self.action_size, self.action_parameter_size).to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.actor_lr)
        self.q_optimizer = optim.Adam(self.q_model.parameters(), lr=self.q_lr)

        # Replay
        self.memory = ReplayBuffer(capacity=self.memory_maxlen, 
                                   state_dim=self.state_size, 
                                   action_dim=self.action_size,  # discrete action as int
                                   action_param_dim=self.action_parameter_size)

        # Init targets
        self.hard_update_targets()

    def hard_update_targets(self):
        self.target_actor_model.load_state_dict(self.actor_model.state_dict())
        self.target_q_model.load_state_dict(self.q_model.state_dict())

    def soft_update(self, target, source, tau):
        """Soft update of target network parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        with torch.no_grad():
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                if target_param.shape != source_param.shape:
                    print(f"Parameter shape mismatch: target {target_param.shape} vs source {source_param.shape}")
                    continue
                target_param.data.copy_(
                    tau * source_param.data + (1.0 - tau) * target_param.data
                )

    def update_target_model(self):
        """Update both target networks using soft update"""
        try:
            # First verify parameter counts match
            actor_params = sum(p.numel() for p in self.actor_model.parameters())
            target_actor_params = sum(p.numel() for p in self.target_actor_model.parameters())
            if actor_params != target_actor_params:
                print(f"Parameter count mismatch in actor networks: {actor_params} vs {target_actor_params}")
                
            q_params = sum(p.numel() for p in self.q_model.parameters())
            target_q_params = sum(p.numel() for p in self.target_q_model.parameters())
            if q_params != target_q_params:
                print(f"Parameter count mismatch in Q networks: {q_params} vs {target_q_params}")
                
            # Perform soft updates
            self.soft_update(self.target_actor_model, self.actor_model, self.tau_actor)
            self.soft_update(self.target_q_model, self.q_model, self.tau_q)
        except Exception as e:
            print(f"Error in update_target_model: {str(e)}")
            raise

    # -------- acting --------
    def get_action(self, state, is_training=True):
        """Return (discrete_action:int, continuous_params:np.ndarray)"""
        # 1) state to tensor
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        # 2) actor proposes continuous parameters (deterministic PDQN style)
        with torch.no_grad():
            param_tensor = self.actor_model.forward(state_tensor)                   # (1, param_dim)
            # noise = torch.from_numpy(np.random.randn(self.action_parameter_size)).float()
            # param_tensor += 0.05 * noise.to(self.device)
        action_params = param_tensor.squeeze(0).cpu().numpy()               # (param_dim,)

        # 3) epsilon-greedy over discrete actions
        if is_training and random.random() < self.epsilon:
            action_idx = random.randrange(self.action_size)
            # keep param shape consistent during exploration
            action_params = np.random.uniform(-1.0, 1.0, size=self.action_parameter_size)
            action_params += np.random.normal(0, 0.1, size=self.action_parameter_size)
            action_params = np.clip(action_params, -1.0, 1.0)
            log_mode = "Exploration"
        else:
            # evaluate Q(s, params) over discrete actions and choose argmax
            with torch.no_grad():
                q_input = torch.cat([state_tensor, torch.from_numpy(action_params).to(self.device).unsqueeze(0)], dim=1)
                q_values = self.q_model.forward(q_input).detach().cpu().data.numpy()                            
                action_idx = np.argmax(q_values)
            log_mode = "Exploitation"

        return action_idx, action_params

    # -------- memory --------
    def append_sample(self, state, action, action_param, reward, next_state, done):
        # ensure param is np.float32 vector
        action_param = np.asarray(action_param, dtype=np.float32).ravel()
        self.memory.append(state, action, action_param, float(reward), next_state, float(done))

    # -------- training --------
    def train_model(self):
        if self.memory.size < self.train_start:
            return

        # epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # ---- Sample batch ----
        states, actions, params, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # ---- Q update ----
        all_q = self.q_model(torch.cat([states, params], dim=1))  # (batch, action_size)
        current_q = all_q.gather(1, actions.view(-1, 1)).squeeze()  # (batch, 1)

        with torch.no_grad():
            pred_next_params = self.target_actor_model(next_states)
            pred_q_all = self.target_q_model(torch.cat([next_states, pred_next_params], dim=1))
            max_pred_q = pred_q_all.max(dim=1).values
            target_q = rewards + (1.0 - dones) * self.discount_factor * max_pred_q

        q_loss = nn.MSELoss()(current_q, target_q)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.q_model.parameters(), 1.0)
        self.q_optimizer.step()

        # ---- Actor update ----
        # with torch.no_grad():
        new_params = self.actor_model(states)
        q_for_actor = self.q_model(torch.cat([states, new_params], dim=1))
        actor_loss = -torch.mean(torch.sum(q_for_actor, dim=1))

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_model.parameters(), 1.0)
        self.actor_optimizer.step()

        # ---- Target updates ----
        self.update_target_model()


    # -------- io --------
    def save_model(self, directory):
        os.makedirs(directory, exist_ok=True)
        torch.save(self.actor_model.state_dict(), os.path.join(directory, 'pdqn_actor_model.pt'))
        torch.save(self.q_model.state_dict(), os.path.join(directory, 'pdqn_q_model.pt'))
        print(f"Models saved to {directory}")

    def load_model(self, directory):
        try:
            self.actor_model.load_state_dict(torch.load(os.path.join(directory, 'pdqn_actor_model.pt'),
                                                        map_location=self.device))
            self.q_model.load_state_dict(torch.load(os.path.join(directory, 'pdqn_q_model.pt'),
                                                    map_location=self.device))
            self.hard_update_targets()
            print(f"Models loaded from {directory}")
        except Exception as e:
            print(f"Error loading models: {e}")
