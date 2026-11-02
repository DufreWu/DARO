import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
scripted_model = torch.jit.script(model)
scripted_model.save("simple_model.pt")