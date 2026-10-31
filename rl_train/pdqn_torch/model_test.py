import torch
import numpy as np

actor_model = torch.jit.load("./save_model/model_actor_torch.pt")  
q_model = torch.jit.load("./save_model/model_Q_torch.pt")

# state: 10 5 7534 20479 0
input_data = [10, 5, 6.534, 6.479, 5]  # Use a list instead of a set

with torch.no_grad():  # Disable gradient computation for inference
    input_tensor = torch.tensor(input_data, dtype=torch.float32)  # Convert input to tensor

    actor_parameter = actor_model(input_tensor)

    # Concatenate input and actor parameter, and add batch dimension
    q_values = q_model(torch.cat((input_tensor, actor_parameter), dim=0).unsqueeze(0))
    print("q_values shape:", q_values.shape)

    # Use argmax on the correct dimension
    action = q_values.argmax(1).item()
    print(f"Input: {input_data}")
    print(f"Actor param: {actor_parameter}")
    print(f"Output: {action}")