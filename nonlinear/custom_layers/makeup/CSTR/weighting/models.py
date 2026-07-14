import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
# from scaler_utils import scaler



# Constants
Cao = 1 #mol/L
Cbo = 0 #mol/L

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
        T_scaled = x                     # shape: [batch, 1]
        # recover original T (MaxAbsScaler: T_scaled = T_orig / max_abs)
        T = T_scaled * self.scaler.scale_[0]  # .scale_[0] is the max-abs for the temperature feature
        x0 = x                                             # input               
        for layer in self.layers[:-1]:
            x0 = F.relu(layer(x0))
        z0 = self.layers[-1](x0)                            # output after NN
        
        Ca, Cb = z0[:, 0:1], z0[:, 1:2]
        
        Ca_unscaled = z0[:, 0:1] * self.scaler.scale_[1]  # unscale Ca
        Cb_unscaled = z0[:, 1:2] * self.scaler.scale_[2]

        kf = Afo * torch.exp(-Eaf / (R * T))
        kr = Aro * torch.exp(-Ear / (R * T))

        # -----------------------------
        # Lifted outputs in physical space
        # -----------------------------
        kfCa = (kf * tau + 1) * (Ca_unscaled)
        krCb2 = kr * (Cb_unscaled**2)
        
        kfCa_scaled = kfCa / self.scaler.scale_[3]
        krCb2_scaled = krCb2 / self.scaler.scale_[4]
        
        basis_outputs = torch.cat([Ca, Cb , kfCa_scaled, krCb2_scaled], dim=1)
        
        if not self.use_kkt_projection:
            return basis_outputs

        z = self.fc_fixed1(basis_outputs) + self.fc_fixed2(x)
        
        # --------------------------------------------------
        # Recover Ca and Cb from the projected lifted outputs
        # z[:, 2] = projected kfCa
        # z[:, 3] = projected krCb2
        # --------------------------------------------------
        # eps = 1e-20

        # kfCa_projected = z[:, 2:3] * self.scaler.scale_[3]
        # krCb2_projected = z[:, 3:4] * self.scaler.scale_[4]
        
        # # Recover physical Cb and Ca
       
        # Ca_recovered = kfCa_projected / (kf * tau + 1)
        # Cb_recovered = torch.sqrt(torch.clamp(krCb2_projected / kr, min=eps))
  
        
        # lifted_residual = - 1 * kfCa_projected + tau * krCb2_projected + Cao

        # recovered_residual = (
        #     -(kf * tau + 1) * (Ca_recovered)
        #     + tau * kr * (Cb_recovered ** 2)
        #     + Cao
        # )
        
        # scaled_kkt_residual = (
        #     torch.mm(self.B, z.T)
        #     + torch.mm(self.A, x.T)
        #     - self.b.repeat(1, x.shape[0])
        # )

        # # Scale recovered Ca and Cb back to scaled space
        # Ca_recovered_scaled = Ca_recovered / self.scaler.scale_[1]
        # Cb_recovered_scaled = Cb_recovered / self.scaler.scale_[2]

        # # Final output:
        # # [recovered Ca, recovered Cb, projected kfCa, projected krCb2]
        # z = torch.cat(
        #     [
        #         Ca_recovered_scaled,
        #         Cb_recovered_scaled,
        #         z[:, 2:3],
        #         z[:, 3:4]
        #     ],
        #     dim=1
        # )

        # return z
        
        # --------------------------------------------------
        # Positive-z4 architecture
        # z4 = kr*Cb^2 is forced positive by construction.
        # Then z3 is reconstructed so the lifted equality is exact:
        # -z3 + tau*z4 + Cao = 0
        # --------------------------------------------------

        eps = torch.as_tensor(1e-12, device=x.device, dtype=x.dtype)

        tau_t = torch.as_tensor(tau, device=x.device, dtype=x.dtype)
        Cao_t = torch.as_tensor(Cao, device=x.device, dtype=x.dtype)

        scale_Ca = torch.as_tensor(self.scaler.scale_[1], device=x.device, dtype=x.dtype)
        scale_Cb = torch.as_tensor(self.scaler.scale_[2], device=x.device, dtype=x.dtype)
        scale_z3 = torch.as_tensor(self.scaler.scale_[3], device=x.device, dtype=x.dtype)
        scale_z4 = torch.as_tensor(self.scaler.scale_[4], device=x.device, dtype=x.dtype)

        # Use the projected z4 as a raw trainable value.
        # This value may be negative, but it is NOT used directly as z4.
        raw_z4_phys = z[:, 3:4] * scale_z4

        # Force physical z4 > 0.
        # beta controls how sharp softplus is.
      
        # z4_phys = F.softplus(raw_z4_phys, beta=100.0, threshold=20.0) + eps
        # z4_phys = F.relu(raw_z4_phys)
        if self.z4_activation == "softplus_eps":
            z4_phys = F.softplus(raw_z4_phys, beta=100.0, threshold=20.0) + eps
        elif self.z4_activation == "relu":
            z4_phys = F.relu(raw_z4_phys) 
        else:
            raise ValueError(f"Unknown z4_activation: {self.z4_activation}")

        # Reconstruct z3 so the lifted equality is exactly satisfied:
        # -z3 + tau*z4 + Cao = 0  ->  z3 = tau*z4 + Cao
        z3_phys = tau_t * z4_phys + Cao_t

        # Recover physical Ca and Cb
        Ca_recovered = z3_phys / (kf * tau_t + 1.0)
        Cb_recovered = torch.sqrt(z4_phys / kr)

        # Optional diagnostics
        lifted_residual = -z3_phys + tau_t * z4_phys + Cao_t

        recovered_residual = (
            -(kf * tau_t + 1.0) * Ca_recovered
            + tau_t * kr * (Cb_recovered ** 2)
            + Cao_t
        )

        # Scale recovered outputs back to scaled space
        Ca_recovered_scaled = Ca_recovered / scale_Ca
        Cb_recovered_scaled = Cb_recovered / scale_Cb
        z3_scaled = z3_phys / scale_z3
        z4_scaled = z4_phys / scale_z4

        # Final output:
        # [Ca, Cb, z3, z4] in scaled space
        z = torch.cat(
            [
                Ca_recovered_scaled,
                Cb_recovered_scaled,
                z3_scaled,
                z4_scaled
            ],
            dim=1
        )

        return z

    def reset_parameters(self):                            # reinitialize the weights of certain layers
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.fc_fixed1 and module is not self.fc_fixed2:
                module.reset_parameters()