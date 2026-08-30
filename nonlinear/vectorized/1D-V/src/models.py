import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(
        self,
        input_dim,
        hidden_dim,
        hidden_num,
        z0_dim,
        A_list,
        B_list,
        b_list,
        C_edges,
    ):
        super().__init__()

        self.register_buffer("C_edges", torch.as_tensor(C_edges))

        # Normal neural network
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        for _ in range(hidden_num - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, z0_dim))

        # -------------------------------------------------
        # Stack ALL region matrices
        # -------------------------------------------------
        A = torch.stack(A_list, dim=0)   # [R, m, input_dim]
        B = torch.stack(B_list, dim=0)   # [R, m, z0_dim]
        b = torch.stack(b_list, dim=0)   # [R, m]

        # -------------------------------------------------
        # Batched KKT projection matrices
        # -------------------------------------------------
        BBt = B @ B.transpose(-1, -2)   # [R, m, m]

        chunk = (
            B.transpose(-1, -2)
            @ torch.linalg.inv(BBt)
        )                                # [R, z0_dim, m]

        Astar = -chunk @ A               # [R, z0_dim, input_dim]

        I = torch.eye(
            z0_dim,
            dtype=B.dtype,
            device=B.device,
        ).unsqueeze(0)

        Bstar = I - chunk @ B             # [R, z0_dim, z0_dim]

        bstar = (
            chunk @ b.unsqueeze(-1)
        ).squeeze(-1)                     # [R, z0_dim]

        # Buffers automatically move CPU <-> GPU with model.to(...)
        self.register_buffer("Astar", Astar)
        self.register_buffer("Bstar", Bstar)
        self.register_buffer("bstar", bstar)

    def get_region_index(self, x):
        edges = self.C_edges.to(
            dtype=x.dtype,
            device=x.device,
        )

        # right=True reproduces:
        # [edge_i, edge_i+1)
        # except last region
        region = torch.bucketize(
            x[:, 0].contiguous(),
            edges[1:-1].contiguous(),
            right=True,
        )

        return region

    def forward(self, x):
        # NN forward
        hidden = x

        for layer in self.layers[:-1]:
            hidden = F.relu(layer(hidden))

        z0 = self.layers[-1](hidden)

        # One region index for every sample simultaneously
        region = self.get_region_index(x)

        # Gather appropriate projection for every sample
        Bstar = self.Bstar[region]   # [batch, z, z]
        Astar = self.Astar[region]   # [batch, z, xdim]
        bstar = self.bstar[region]   # [batch, z]

        # Batched projection
        output = (
            torch.bmm(
                Bstar,
                z0.unsqueeze(-1),
            ).squeeze(-1)
            +
            torch.bmm(
                Astar,
                x.unsqueeze(-1),
            ).squeeze(-1)
            +
            bstar
        )

        return output

    def reset_parameters(self):
        for module in self.layers:
            if isinstance(module, nn.Linear):
                module.reset_parameters()