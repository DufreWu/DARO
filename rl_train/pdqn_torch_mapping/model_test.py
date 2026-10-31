import torch
import numpy as np

# ============================================================
# Load TorchScript models
# ============================================================
actor_model = torch.jit.load("./save_model/pdqn_actor_model.pt")
q_model = torch.jit.load("./save_model/pdqn_q_model.pt")

actor_model.eval()
q_model.eval()

# ============================================================
# Input state
# ============================================================
input_data = [
    4.6377487, 4.266667, 2.1278203, 2., 90.,
    0., 0., 0., 0., 18.,
    0., 0., 0., 0., 25.,
]

num_actions = 6  # <-- MUST match training

with torch.no_grad():

    # Add batch dimension
    state = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

    # --------------------------------------------------------
    # Actor: continuous parameters θ(s)
    # --------------------------------------------------------
    actor_param = actor_model(state)        # shape: (1, param_dim)

    # --------------------------------------------------------
    # Q evaluation for each discrete action
    # --------------------------------------------------------
    q_values = []

    for action_idx in range(num_actions):
        action_tensor = torch.tensor([[action_idx]], dtype=torch.long)
        q_val = q_model(state, action_tensor, actor_param)  # shape: (1, 1)
        q_values.append(q_val.item())

    q_values = np.array(q_values)

    # --------------------------------------------------------
    # Select best action
    # --------------------------------------------------------
    action = int(np.argmax(q_values))

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------
    print("Input state:")
    print(input_data)

    print("\nActor parameters θ(s):")
    print(actor_param.squeeze().cpu().numpy())

    print("\nQ-values per action:")
    for i, q in enumerate(q_values):
        print(f"  Action {i}: Q = {q:.4f}")

    print(f"\nSelected action: {action}")
