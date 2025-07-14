import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import pandas as pd

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"


class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, z0_dim):
        super(NN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))    #first layer (maps the input to the hidden layer)
        for _ in range(hidden_num - 1):                         #a loop adds hidden layers (with 2 hidden_num we just have 1 hidden layer here! the article cliams 2 hidden layers)
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, z0_dim))       #final layer (maps the last hidden layer to the output layer)

    def forward(self, x):                                       # x: input data
        for layer in self.layers[:-1]:                          # excludes the last layer because the activation is not applied to the output layer (The -1 index in Python is shorthand to access the last item of a list, so self.layers[-1] refers to the final layer that maps the last hidden layer to the output layer)
            x = F.relu(layer(x))                                # applies the ReLU activation function to the output of each layer
        z0 = self.layers[-1](x)                                 # output layer
        return z0

    def reset_parameters(self):                                # reinitializing the model's weights for retraining
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()


class NNOPT(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, z0_dim, A_list, B_list, b_list):
        super(NNOPT, self).__init__()  

        self.layers = nn.ModuleList()                        # a list of learnable layers for neural network
        self.layers.append(nn.Linear(input_dim, hidden_dim)) # a linear layer
        for _ in range(hidden_num - 1):                      # (_) indicate that the variable's value is not important and will not be used in the loop body.
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, z0_dim))
        
        self.fc1_list = nn.ModuleList()
        self.fc2_list = nn.ModuleList()
        for A, B, b in zip(A_list, B_list, b_list):
            chunk  = B.t() @ torch.inverse(B @ B.t())          # (z×1)(1×z)
            Astar  = -chunk @ A                                # (z×x)
            Bstar  = torch.eye(z0_dim) - chunk @ B             # (z×z)
            bstar  = (chunk @ b.view(-1,1)).squeeze(-1)        # (z)

            fc1 = nn.Linear(z0_dim, z0_dim, bias=False)
            fc1.weight = nn.Parameter(Bstar, requires_grad=False)
            fc2 = nn.Linear(input_dim, z0_dim, bias=False)
            fc2.weight = nn.Parameter(Astar, requires_grad=False)
            fc2.bias   = nn.Parameter(bstar, requires_grad=False)

            self.fc1_list.append(fc1)
            self.fc2_list.append(fc2)
            

    def custom_sigmoid(self, x, transition_point=[0.375, 0.425, 0.46875], steepness=8000):
        transition_width = 100 / steepness
        w = (x - transition_point) / transition_width
        return torch.sigmoid(w)
    
    # def inverse_custom_sigmoid(x, transition_point, steepness):
    #     return 1.0 - custom_sigmoid(self, x, transition_point, steepness)

    def forward(self, x, transition_points=[0.375, 0.425, 0.46875], steepness=8000):
        x0 = x                                             # input               
        for layer in self.layers[:-1]:
            x0 = F.relu(layer(x0))
        z0 = self.layers[-1](x0) 
        
        # fixed branches masked by their sigmoids
        fixed_outputs = []
        for i, (fc1, fc2) in enumerate(zip(self.fc1_list, self.fc2_list)):
            z_fixed = fc1(z0) + fc2(x)
            if transition_points is None:
                mask = 1.0        # no gating
            else:                 # same mask logic you used before
                if i == 0:
                    mask = 1.0 - self.custom_sigmoid(x, transition_points[0], steepness)
                elif i == len(self.fc1_list)-1:
                    mask = self.custom_sigmoid(x, transition_points[-1], steepness)
                else:
                    mask = (self.custom_sigmoid(x, transition_points[i-1], steepness) *
                        (1.0 - self.custom_sigmoid(x, transition_points[i], steepness)))
            fixed_outputs.append(z_fixed * mask)
        return sum(fixed_outputs)
        

    def reset_parameters(self):                            # reinitialize the weights of certain layers
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.fc1_list and module is not self.fc2_list:
                module.reset_parameters()