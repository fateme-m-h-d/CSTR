import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, z0_dim):
        super().__init__()
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
        T_edges,
        C_edges,
    ):
        super().__init__()

        self.register_buffer(
            "T_edges",
            torch.as_tensor(T_edges),
        )

        self.register_buffer(
            "C_edges",
            torch.as_tensor(C_edges),
        )

        # NN
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Linear(input_dim, hidden_dim)
        )

        for _ in range(hidden_num - 1):
            self.layers.append(
                nn.Linear(hidden_dim, hidden_dim)
            )

        self.layers.append(
            nn.Linear(hidden_dim, z0_dim)
        )

        # ---------------------------------------------
        # Stack all PL regions
        # ---------------------------------------------
        A = torch.stack(A_list, dim=0)
        B = torch.stack(B_list, dim=0)
        b = torch.stack(b_list, dim=0)

        BBt = B @ B.transpose(-1, -2)

        chunk = (
            B.transpose(-1, -2)
            @ torch.linalg.inv(BBt)
        )

        Astar = -chunk @ A

        I = torch.eye(
            z0_dim,
            dtype=B.dtype,
            device=B.device,
        ).unsqueeze(0)

        Bstar = I - chunk @ B

        bstar = (
            chunk @ b.unsqueeze(-1)
        ).squeeze(-1)

        self.register_buffer("Astar", Astar)
        self.register_buffer("Bstar", Bstar)
        self.register_buffer("bstar", bstar)

    def get_region_index(self, x):
        T_edges = self.T_edges.to(
            dtype=x.dtype,
            device=x.device,
        )

        C_edges = self.C_edges.to(
            dtype=x.dtype,
            device=x.device,
        )

        # x[:,0] = T
        T_region = torch.bucketize(
            x[:, 0].contiguous(),
            T_edges[1:-1].contiguous(),
            right=True,
        )

        # x[:,1] = Cao
        C_region = torch.bucketize(
            x[:, 1].contiguous(),
            C_edges[1:-1].contiguous(),
            right=True,
        )

        nC = len(C_edges) - 1

        # Important: matches your linearization.py ordering
        region = T_region * nC + C_region

        return region

    def forward(self, x):
        hidden = x

        for layer in self.layers[:-1]:
            hidden = F.relu(layer(hidden))

        z0 = self.layers[-1](hidden)

        region = self.get_region_index(x)

        Bstar = self.Bstar[region]
        Astar = self.Astar[region]
        bstar = self.bstar[region]

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