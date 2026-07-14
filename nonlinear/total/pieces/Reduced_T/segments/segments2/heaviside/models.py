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
    def __init__(self, input_dim, hidden_dim, hidden_num, z0_dim, A_list, B_list, b_list, T_edges):
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
        
        self.register_buffer("T_edges", torch.as_tensor(T_edges))
            

    def get_masks(self, x, steepness=None):
        """
        Hard Heaviside/indicator masks.
        One active region per sample.

        x is scaled temperature with shape (batch, 1).
        self.T_edges is also scaled.
        """
        n = len(self.fc1_list)

        if n == 1:
            return torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)

        T_edges = self.T_edges.to(dtype=x.dtype, device=x.device)
        x1d = x[:, 0]

        masks = torch.zeros((x.shape[0], n), dtype=x.dtype, device=x.device)

        for i in range(n):
            lo = T_edges[i]
            hi = T_edges[i + 1]

            if i < n - 1:
                mask_i = (x1d >= lo) & (x1d < hi)
            else:
                mask_i = (x1d >= lo) & (x1d <= hi)

            masks[mask_i, i] = 1.0

        # Optional safety for points slightly outside the region range
        masks[x1d < T_edges[0], 0] = 1.0
        masks[x1d > T_edges[-1], -1] = 1.0

        return masks
    def forward(self, x, steepness=800000):
        x0 = x                                             # input               
        for layer in self.layers[:-1]:
            x0 = F.relu(layer(x0))
        z0 = self.layers[-1](x0)
        
        masks = self.get_masks(x, steepness=steepness)
        
        # fixed branches masked by their sigmoids
        fixed_outputs = []
        for i, (fc1, fc2) in enumerate(zip(self.fc1_list, self.fc2_list)):
            z_fixed = fc1(z0) + fc2(x)
            mask = masks[:, i:i+1]
            fixed_outputs.append(z_fixed * mask)
        return sum(fixed_outputs)
        

    def reset_parameters(self):                            # reinitialize the weights of certain layers
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.fc1_list and module is not self.fc2_list:
                module.reset_parameters()


# class NNOPT(nn.Module):
#     def __init__(self, input_dim, hidden_dim, hidden_num, z0_dim, A_list, B_list, b_list):
#         super(NNOPT, self).__init__()  

#         self.layers = nn.ModuleList()
#         self.layers.append(nn.Linear(input_dim, hidden_dim))
#         for _ in range(hidden_num - 1):
#             self.layers.append(nn.Linear(hidden_dim, hidden_dim))
#         self.layers.append(nn.Linear(hidden_dim, z0_dim))
        
#         self.fc1_list = nn.ModuleList()
#         self.fc2_list = nn.ModuleList()
#         for A, B, b in zip(A_list, B_list, b_list):
#             chunk  = B.t() @ torch.inverse(B @ B.t())
#             Astar  = -chunk @ A
#             Bstar  = torch.eye(z0_dim) - chunk @ B
#             bstar  = (chunk @ b.view(-1,1)).squeeze(-1)

#             fc1 = nn.Linear(z0_dim, z0_dim, bias=False)
#             fc1.weight = nn.Parameter(Bstar, requires_grad=False)
#             fc2 = nn.Linear(input_dim, z0_dim, bias=False)
#             fc2.weight = nn.Parameter(Astar, requires_grad=False)
#             fc2.bias   = nn.Parameter(bstar, requires_grad=False)

#             self.fc1_list.append(fc1)
#             self.fc2_list.append(fc2)

#     def custom_sigmoid(self, x, transition_point, steepness=500000):
#         transition_width = 100 / steepness
#         w = (x - transition_point) / transition_width
#         return torch.sigmoid(w)
    
#     def compute_normalized_masks(self, x, transition_points, steepness):
#         """
#         Compute masks that are guaranteed to sum to 1.0 everywhere
#         """
#         if transition_points is None:
#             # No gating - use first region only
#             return [torch.ones_like(x)] + [torch.zeros_like(x)] * (len(self.fc1_list) - 1)
        
#         n_regions = len(self.fc1_list)
#         raw_masks = []
        
#         # Compute raw masks using your original logic
#         for i in range(n_regions):
#             if i == 0:
#                 mask = 1.0 - self.custom_sigmoid(x, transition_points[0], steepness)
#             elif i == n_regions - 1:
#                 mask = self.custom_sigmoid(x, transition_points[-1], steepness)
#             else:
#                 mask = (self.custom_sigmoid(x, transition_points[i-1], steepness) *
#                        (1.0 - self.custom_sigmoid(x, transition_points[i], steepness)))
#             raw_masks.append(mask)
        
#         # Normalize masks to ensure they sum to 1
#         raw_masks = torch.stack(raw_masks, dim=0)  # shape: (n_regions, batch, 1)
#         mask_sum = torch.sum(raw_masks, dim=0, keepdim=True)  # shape: (1, batch, 1)
#         mask_sum = torch.clamp(mask_sum, min=1e-8)  # avoid division by zero
#         normalized_masks = raw_masks / mask_sum  # shape: (n_regions, batch, 1)
        
#         return [normalized_masks[i] for i in range(n_regions)]

#     def forward(self, x, transition_points=[0.375, 0.425, 0.45, 0.5, 0.625], steepness=500000):
#         x0 = x
#         for layer in self.layers[:-1]:
#             x0 = F.relu(layer(x0))
#         z0 = self.layers[-1](x0) 
        
#         # Get normalized masks
#         masks = self.compute_normalized_masks(x, transition_points, steepness)
        
#         # Compute weighted sum of region outputs
#         fixed_outputs = []
#         for i, (fc1, fc2, mask) in enumerate(zip(self.fc1_list, self.fc2_list, masks)):
#             z_fixed = fc1(z0) + fc2(x)
#             fixed_outputs.append(z_fixed * mask)
        
#         return sum(fixed_outputs)

#     def reset_parameters(self):
#         for module in self.modules():
#             if isinstance(module, nn.Linear) and module not in self.fc1_list and module not in self.fc2_list:
#                 module.reset_parameters()