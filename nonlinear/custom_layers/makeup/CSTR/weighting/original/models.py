import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
# from scaler_utils import scaler



# Constants
# Cao = 1 #mol/L
Cbo = 2 #mol/L
Cco = 0 #mol/L

V = 10 #L
Q = 1 #L/s
tau = V/Q #s
Afo = 10e12
Eaf = 90000 #J/mol
Aro = 10e10
Ear = 80000 #J/mol
R = 8.314 #J/mol


T_ISO = 350.0

kf_const = Afo * np.exp(-Eaf / (R * T_ISO))
kr_const = Aro * np.exp(-Ear / (R * T_ISO))

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"


class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, z0_inner_dim):
        super(NN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))    #first layer (maps the input to the hidden layer)
        for _ in range(hidden_num - 1):                         #a loop adds hidden layers (with 2 hidden_num we just have 1 hidden layer here! the article cliams 2 hidden layers)
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, z0_inner_dim))       #final layer (maps the last hidden layer to the output layer)

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
    def __init__(self, input_dim, hidden_dim, hidden_num, z0_inner_dim, z0_dim, A, B, b, scaler, z4_activation="softplus_eps"):
        super(NNOPT, self).__init__()
        self.scaler = scaler
        self.A = A                                           # orthogonal projection (satisfying hard linear constriants)
        self.B = B
        self.b = b
        self.chunk = torch.mm(B.t(),                         # B.t= transpose of B
                             torch.inverse(                  # inverse of a squared matrix
                                 torch.mm(B, B.t())
                             )
                             )                               # 𝐵𝑇 (𝐵𝐵𝑇 )-1
        self.Astar = - torch.mm(self.chunk, self.A)          # -𝐵𝑇 (𝐵𝐵𝑇 )-1 * A
        self.Bstar = torch.eye(z0_dim).to(device) - torch.mm(self.chunk, self.B)   # I - 𝐵𝑇 (𝐵𝐵𝑇 )-1 * B
        self.bstar = torch.matmul(self.chunk, self.b).squeeze(-1)                  # 𝐵𝑇 (𝐵𝐵𝑇 )-1 * b    # I don't understand .squeeze(-1)   

        self.layers = nn.ModuleList()                        # a list of learnable layers for neural network
        self.layers.append(nn.Linear(input_dim, hidden_dim)) # a linear layer
        for _ in range(hidden_num - 1):                      # (_) indicate that the variable's value is not important and will not be used in the loop body.
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, z0_inner_dim))

        self.fc_fixed1 = nn.Linear(z0_dim, z0_dim, bias=False)  ########    #fixed layers fc_fixed1 (a linear transformation with no bias and fixed weights) and fc_fixed2. 
        self.fc_fixed1.weight = nn.Parameter(self.Bstar, requires_grad=False)  # fixed weight (not learnable- will not be updated during training), Bstar
        self.fc_fixed2 = nn.Linear(input_dim, z0_dim, bias=False)  ########    # input dimensions
        self.fc_fixed2.weight = nn.Parameter(self.Astar, requires_grad=False)  # Astar as weights
        self.fc_fixed2.bias = nn.Parameter(self.bstar, requires_grad=False)    # bstar as bias
        self.z4_activation = z4_activation
        self.use_kkt_projection = True

    def forward(self, x):
        # x is scaled Cao
        x0 = x
        for layer in self.layers[:-1]:
            x0 = F.relu(layer(x0))
        z0 = self.layers[-1](x0)

        scale_Cao = torch.as_tensor(self.scaler.scale_[0], device=x.device, dtype=x.dtype)
        scale_Ca  = torch.as_tensor(self.scaler.scale_[1], device=x.device, dtype=x.dtype)
        scale_Cb  = torch.as_tensor(self.scaler.scale_[2], device=x.device, dtype=x.dtype)
        scale_Cc  = torch.as_tensor(self.scaler.scale_[3], device=x.device, dtype=x.dtype)
        scale_h   = torch.as_tensor(self.scaler.scale_[4], device=x.device, dtype=x.dtype)

        Cao_in = x[:, 0:1] * scale_Cao

        Ca_hat = z0[:, 0:1]
        Cb_hat = z0[:, 1:2]
        Cc_hat = z0[:, 2:3]

        Ca = Ca_hat * scale_Ca
        Cb = Cb_hat * scale_Cb
        Cc = Cc_hat * scale_Cc

        # lifting: h = Ca * Cb^2
        h = Ca * (Cb ** 2)

        basis_outputs = torch.cat(
            [
                Ca_hat,
                Cb_hat,
                Cc_hat,
                h / scale_h,
            ],
            dim=1
        )

        if not self.use_kkt_projection:
            return basis_outputs

        # KKT projection in lifted space [Ca, Cb, Cc, h]
        z = self.fc_fixed1(basis_outputs) + self.fc_fixed2(x)

        Ca_p = z[:, 0:1] * scale_Ca
        Cb_p = z[:, 1:2] * scale_Cb
        Cc_p = z[:, 2:3] * scale_Cc
        h_p  = z[:, 3:4] * scale_h

        eps = torch.as_tensor(1e-12, device=x.device, dtype=x.dtype)

        # First version: no softplus
        if self.z4_activation == "raw":
            h_used = h_p
        elif self.z4_activation == "softplus_eps":
            h_used = F.softplus(h_p, beta=100.0, threshold=20.0) + eps
        elif self.z4_activation == "relu":
            h_used = F.relu(h_p) + eps
        else:
            raise ValueError(f"Unknown z4_activation: {self.z4_activation}")

        # Recovery 1: from h = Ca * Cb^2
        Cb1 = torch.sqrt(h_used / (Ca_p + eps))

        # Recovery 2: from total mass balance
        Cb2 = Cao_in + Cbo + Cco - Ca_p - Cc_p

        out = torch.cat(
            [
                Ca_p / scale_Ca,
                Cb1 / scale_Cb,
                Cb2 / scale_Cb,
                Cc_p / scale_Cc,
                h_used / scale_h,
            ],
            dim=1
        )

        return out

    def reset_parameters(self):                            # reinitialize the weights of certain layers
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.fc_fixed1 and module is not self.fc_fixed2:
                module.reset_parameters()