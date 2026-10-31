# ddpg_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.out(x))  # output in [-1, 1]

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


# Parameters
state_dim = 6
action_dim = 2

# Initialize and load your trained model weights here if available
actor = Actor(state_dim, action_dim)
critic = Critic(state_dim, action_dim)

# OPTIONAL: Load trained weights if you already have them
# actor.load_state_dict(torch.load("actor.pth"))
# critic.load_state_dict(torch.load("critic.pth"))

# Script the models
scripted_actor = torch.jit.script(actor)
scripted_critic = torch.jit.script(critic)

# Save to .pt for C++ use
scripted_actor.save("ddpg_actor.pt")
scripted_critic.save("ddpg_critic.pt")
print("Saved ddpg_actor.pt and ddpg_critic.pt")

model = torch.jit.load("ddpg_actor.pt")
x = torch.rand(1, 6)
print(model(x))
