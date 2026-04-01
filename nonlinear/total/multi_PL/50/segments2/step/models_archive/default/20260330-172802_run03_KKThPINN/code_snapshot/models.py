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
    def __init__(self, input_dim, hidden_dim, hidden_num, z0_dim, A_list, B_list, b_list, T_edges, C_edges):
        super(NNOPT, self).__init__()
        
        # --- store region edges (NOT trainable, moves with model.to(device)) ---
        self.register_buffer("T_edges", torch.tensor(T_edges))
        self.register_buffer("C_edges", torch.tensor(C_edges))


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
            

    def custom_sigmoid(self, x, transition_point, steepness):
        transition_width = 100 / steepness
        w = (x - transition_point) / transition_width
        return torch.sigmoid(w)
    
    def window_1d(self, x, L, U, steepness):
        left  = self.custom_sigmoid(x, L, steepness)
        right = 1.0 - self.custom_sigmoid(x, U, steepness)
        return left * right
    
    def hard_window_1d(self, x, L, U, is_last=False):
        if is_last:
            return ((x >= L) & (x <= U)).to(x.dtype)
        else:
            return ((x >= L) & (x < U)).to(x.dtype)
    
    def get_masks_2d(self, x, steepT=8e5, steepC=8e5, normalize=True, hard=False):
        T_edges = self.T_edges.to(device=x.device, dtype=x.dtype)
        C_edges = self.C_edges.to(device=x.device, dtype=x.dtype)

        T = x[:, 0:1]
        C = x[:, 1:2]

        masks = []
        nT = len(T_edges) - 1
        nC = len(C_edges) - 1

        for i in range(nT):
            if hard:
                wT = self.hard_window_1d(T, T_edges[i], T_edges[i+1], is_last=(i == nT - 1))
            else:
                wT = self.window_1d(T, T_edges[i], T_edges[i+1], steepT)

            for j in range(nC):
                if hard:
                    wC = self.hard_window_1d(C, C_edges[j], C_edges[j+1], is_last=(j == nC - 1))
                else:
                    wC = self.window_1d(C, C_edges[j], C_edges[j+1], steepC)

                masks.append(wT * wC)

        masks = torch.cat(masks, dim=1)

        assert masks.shape[1] == len(self.fc1_list), \
            f"Mask count {masks.shape[1]} != #branches {len(self.fc1_list)}. Check edges/order."

        if normalize:
            masks = masks / (masks.sum(dim=1, keepdim=True) + 1e-12)

        return masks

    def forward(self, x, steepT=8e5, steepC=8e5, hard=False):
        # NN
        x0 = x
        for layer in self.layers[:-1]:
            x0 = F.relu(layer(x0))
        z0 = self.layers[-1](x0)

        # 2D masks
        masks = self.get_masks_2d(x, steepT=steepT, steepC=steepC, normalize=True, hard=hard)

        # blend regions
        out = torch.zeros_like(z0)
        for r, (fc1, fc2) in enumerate(zip(self.fc1_list, self.fc2_list)):
            z_fixed = fc1(z0) + fc2(x)
            w = masks[:, r:r+1]
            out = out + z_fixed * w

        return out

    def reset_parameters(self):                            # reinitialize the weights of certain layers
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.fc1_list and module is not self.fc2_list:
                module.reset_parameters()


