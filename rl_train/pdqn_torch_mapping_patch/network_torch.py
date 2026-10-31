import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from config import config, robot_env

class ParamActor(nn.Module):
    def __init__(self, state_dim, param_dim, hidden1_dim=128, hidden2_dim=128):
        super(ParamActor, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden1_dim)
        self.ln1 = nn.LayerNorm(hidden1_dim)
        
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.ln2 = nn.LayerNorm(hidden2_dim)
        
        self.output_layer = nn.Linear(hidden2_dim, param_dim)

        # Initialize layers
        self._init_weights()

    def _init_weights(self):
        # Using Kaiming for ReLU layers
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        # Very small weights for the output to start near zero (middle of Tanh)
        nn.init.uniform_(self.output_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.output_layer.bias, -3e-3, 3e-3)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        # Tanh keeps params in [-1, 1]. 
        # Ensure your environment scales this (e.g., speed = actor_out * 5.0)
        return torch.tanh(self.output_layer(x))


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, param_dim, hidden1_dim=128, hidden2_dim=128):
        super(QNetwork, self).__init__()

        # Input is State + ALL parameters
        input_dim = state_dim + param_dim

        self.fc1 = nn.Linear(input_dim, hidden1_dim)
        self.ln1 = nn.LayerNorm(hidden1_dim)
        
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.ln2 = nn.LayerNorm(hidden2_dim)
        
        self.output_layer = nn.Linear(hidden2_dim, action_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.uniform_(self.output_layer.weight, -3e-4, 3e-4)
        nn.init.uniform_(self.output_layer.bias, -3e-4, 3e-4)

    def forward(self, x):
        # x should be concat(state, action_parameters)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.output_layer(x)
    

state_dim=config["state_size"]
action_dim=config["action_size"]         
param_dim=config["action_parameter_size"]
print(f"State Dim: {state_dim}, Action Dim: {action_dim}, Param Dim: {param_dim}")

actor_model = ParamActor(state_dim, param_dim)
actor_model.load_state_dict(torch.load("./save_model/pdqn_actor_model.pt", map_location="cpu"))
actor_model.eval()

q_model = QNetwork(state_dim, action_dim, param_dim)
q_model.load_state_dict(torch.load("./save_model/pdqn_q_model.pt", map_location="cpu"))
q_model.eval()

# ====== SUMMARY ======
print("\n=== Param Model ===")
summary(actor_model, input_size=(1, state_dim))

print("\n=== Q Network ===")
summary(q_model, input_size=(1, state_dim + param_dim))
