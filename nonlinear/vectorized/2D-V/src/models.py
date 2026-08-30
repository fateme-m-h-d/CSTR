import torch
import torch.nn as nn
import torch.nn.functional as F


def get_masks_1d(values, edges):
    edges = torch.as_tensor(edges, dtype=values.dtype, device=values.device)
    n_regions = len(edges) - 1
    if n_regions == 1:
        return torch.ones(
            (values.shape[0], 1), dtype=values.dtype, device=values.device
        )

    masks = torch.zeros(
        (values.shape[0], n_regions),
        dtype=values.dtype,
        device=values.device,
    )
    for i in range(n_regions):
        lower = edges[i]
        upper = edges[i + 1]
        if i < n_regions - 1:
            in_region = (values >= lower) & (values < upper)
        else:
            in_region = (values >= lower) & (values <= upper)
        masks[in_region, i] = 1.0

    masks[values < edges[0], 0] = 1.0
    masks[values > edges[-1], -1] = 1.0
    return masks


def get_masks_2d(X, T_edges, C_edges):
    region = get_region_index_2d(
        X,
        T_edges,
        C_edges,
    )

    nT = len(T_edges) - 1
    nC = len(C_edges) - 1
    n_regions = nT * nC

    return F.one_hot(
        region,
        num_classes=n_regions,
    ).to(dtype=X.dtype)


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