import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
# from scaler_utils import scaler
from cubic import solve_cb_ca_g_from_fh


"""
same model as before, without weights, but returning both 3 and 5 outputs.

I wanna clean the code here, and delete extra stuff, and have just 3 outputs probably and calculate the violation based on that.
"""

# Constants
Cao = 1 #mol/L
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
    def __init__(self, input_dim, hidden_dim, hidden_num, z0_inner_dim, z0_dim, A, B, b, scaler):
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

    def forward(self, x):
        T_scaled = x                     # shape: [batch, 1]
        # recover original T (MaxAbsScaler: T_scaled = T_orig / max_abs)
        T = T_scaled * self.scaler.scale_[0]  # .scale_[0] is the max-abs for the temperature feature
        x0 = x                                             # input               
        for layer in self.layers[:-1]:
            x0 = F.relu(layer(x0))
        z0 = self.layers[-1](x0)                            # output after NN
        
        Ca, Cb, Cc = z0[:, 0:1], z0[:, 1:2], z0[:, 2:3]
        
        Ca_unscaled = z0[:, 0:1] * self.scaler.scale_[1]  # unscale Ca
        Cb_unscaled = z0[:, 1:2] * self.scaler.scale_[2]

        kf = Afo * torch.exp(-Eaf / (R * T))
        kr = Aro * torch.exp(-Ear / (R * T))
        
        if not torch.isfinite(Ca_unscaled).all() or not torch.isfinite(Cb_unscaled).all():
            print("Non-finite Ca/Cb before g/f/h")
            print("Ca_unscaled min/max:", Ca_unscaled.min().item(), Ca_unscaled.max().item())
            print("Cb_unscaled min/max:", Cb_unscaled.min().item(), Cb_unscaled.max().item())
            raise RuntimeError("NaN/Inf in Ca/Cb")


        g = (kf) * Ca_unscaled * (Cb_unscaled ** 2)
        h = (Ca_unscaled + tau * g)/ self.scaler.scale_[5]   # scaled h
        f = (kr / self.scaler.scale_[4]) * (Cao - Ca_unscaled + Cbo - Cb_unscaled + Cco)
        
        # keep f,h finite to prevent 0*inf -> nan in projection
        # f = torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
        # h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

        basis_outputs = torch.cat([Ca, Cb , Cc, f, h], dim=1)
        
        # basis_outputs = torch.nan_to_num(basis_outputs, nan=0.0, posinf=0.0, neginf=0.0)


        z5 = self.fc_fixed1(basis_outputs) + self.fc_fixed2(x)
        
        # delta = (z5 - basis_outputs).abs().mean(dim=0)
        # print("mean |delta| per component [Ca,Cb,Cc,f,g]:", delta.tolist())
        
        z3 = z5[:, :3]   # keep first 3 components
        
        f_s = z5[:, 3:4]
        h_s = z5[:, 4:5]
        
        Ca_s, Cb_s, Cc_s, Ca_un, Cb_un, g_un, ok = solve_cb_ca_g_from_fh(
            T_scaled=T_scaled,  # x, shape [N,1]
            z5=z5,              # shape [N,5]
            scaler=self.scaler
        )
        if not torch.isfinite(Ca_s).all() or not torch.isfinite(Cb_s).all():
            print("NON-FINITE in cubic outputs")
            print("f_s range:", f_s.min().item(), f_s.max().item())
            print("h_s range:", h_s.min().item(), h_s.max().item())
            print("kr range:", kr.min().item(), kr.max().item())
            print("T range:", T.min().item(), T.max().item())
            raise RuntimeError("NaN/Inf detected")

        # print("ok ratio:", ok.float().mean().item())
        # print("finite Ca_s:", torch.isfinite(Ca_s).float().mean().item(),
        #     "finite Cb_s:", torch.isfinite(Cb_s).float().mean().item())
        # print("kr min:", kr.min().item(), "kr max:", kr.max().item())


        z3 = torch.cat([Ca_s, Cb_s, Cc_s], dim=1)
        return z3, z5, z0

    def reset_parameters(self):                            # reinitialize the weights of certain layers
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.fc_fixed1 and module is not self.fc_fixed2:
                module.reset_parameters()