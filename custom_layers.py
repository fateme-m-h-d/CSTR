import torch
import torch.nn as nn

class exponent(nn.Module):
    def __init__(self):
        super(exponent, self).__init__()

    def forward(self, x):
        return torch.cos(x)

data=torch.tensor([0, torch.pi/2])
print(data)
power=exponent()
result=power(data)
print(result)