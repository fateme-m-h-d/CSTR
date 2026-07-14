import torch
import torch.nn as nn
import torch.nn.functional as F

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"


class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, z0_dim):
        super(NN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(hidden_num - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, z0_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()


class NNOPT(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, z0_dim, A_list, B_list, b_list, C_edges):
        super(NNOPT, self).__init__()
        self.register_buffer("C_edges", torch.as_tensor(C_edges))

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(hidden_num - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, z0_dim))

        self.fc1_list = nn.ModuleList()
        self.fc2_list = nn.ModuleList()

        for A, B, b in zip(A_list, B_list, b_list):
            chunk = B.t() @ torch.inverse(B @ B.t())
            Astar = -chunk @ A
            Bstar = torch.eye(z0_dim, dtype=B.dtype, device=B.device) - chunk @ B
            bstar = (chunk @ b.view(-1, 1)).squeeze(-1)

            fc1 = nn.Linear(z0_dim, z0_dim, bias=False)
            fc1.weight = nn.Parameter(Bstar, requires_grad=False)

            fc2 = nn.Linear(input_dim, z0_dim, bias=True)
            fc2.weight = nn.Parameter(Astar, requires_grad=False)
            fc2.bias = nn.Parameter(bstar, requires_grad=False)

            self.fc1_list.append(fc1)
            self.fc2_list.append(fc2)

    def get_masks_1d(self, x):
        """Hard Heaviside/indicator masks for Cao regions. x is scaled Cao, shape (batch, 1)."""
        C_edges = self.C_edges.to(dtype=x.dtype, device=x.device)
        n_regions = len(C_edges) - 1

        if n_regions == 1:
            return torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)

        c = x[:, 0]
        masks = torch.zeros((x.shape[0], n_regions), dtype=x.dtype, device=x.device)

        for i in range(n_regions):
            lo = C_edges[i]
            hi = C_edges[i + 1]
            if i < n_regions - 1:
                mask_i = (c >= lo) & (c < hi)
            else:
                mask_i = (c >= lo) & (c <= hi)
            masks[mask_i, i] = 1.0

        masks[c < C_edges[0], 0] = 1.0
        masks[c > C_edges[-1], -1] = 1.0
        return masks

    def forward(self, x):
        x0 = x
        for layer in self.layers[:-1]:
            x0 = F.relu(layer(x0))
        z0 = self.layers[-1](x0)

        masks = self.get_masks_1d(x)
        out = torch.zeros_like(z0)
        for r, (fc1, fc2) in enumerate(zip(self.fc1_list, self.fc2_list)):
            z_fixed = fc1(z0) + fc2(x)
            out = out + z_fixed * masks[:, r:r+1]
        return out

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) and module not in self.fc1_list and module not in self.fc2_list:
                module.reset_parameters()
