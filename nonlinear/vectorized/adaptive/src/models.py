import torch
import torch.nn as nn
import torch.nn.functional as F


def get_rectangle_index(X, region_bounds):
    """Select one adaptive rectangle per sample without a region loop.

    Bounds are [T_low, T_high, Cao_low, Cao_high] in the same scaled
    coordinates as X. Internal upper edges are open; domain upper edges
    are closed. Points outside the domain use the nearest rectangle.
    """
    bounds = torch.as_tensor(region_bounds, dtype=X.dtype, device=X.device)
    T = X[:, 0:1]
    C = X[:, 1:2]
    T_low = bounds[:, 0].unsqueeze(0)
    T_high = bounds[:, 1].unsqueeze(0)
    C_low = bounds[:, 2].unsqueeze(0)
    C_high = bounds[:, 3].unsqueeze(0)

    touches_T_max = torch.isclose(bounds[:, 1], bounds[:, 1].max()).unsqueeze(0)
    touches_C_max = torch.isclose(bounds[:, 3], bounds[:, 3].max()).unsqueeze(0)
    T_upper_ok = (T < T_high) | (touches_T_max & (T <= T_high))
    C_upper_ok = (C < C_high) | (touches_C_max & (C <= C_high))
    inside = (T >= T_low) & T_upper_ok & (C >= C_low) & C_upper_ok

    region = inside.to(dtype=torch.long).argmax(dim=1)
    dT = torch.relu(T_low - T) + torch.relu(T - T_high)
    dC = torch.relu(C_low - C) + torch.relu(C - C_high)
    nearest = (dT.square() + dC.square()).argmin(dim=1)
    return torch.where(inside.any(dim=1), region, nearest)


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
        region_bounds,
    ):
        super().__init__()

        self.register_buffer(
            "region_bounds",
            torch.as_tensor(region_bounds),
        )
        if self.region_bounds.ndim != 2 or self.region_bounds.shape[1] != 4:
            raise ValueError("region_bounds must have shape (n_regions, 4)")
        n_regions = self.region_bounds.shape[0]
        if n_regions == 0 or not (n_regions == len(A_list) == len(B_list) == len(b_list)):
            raise ValueError("Region bounds and A/B/b lists must have the same nonzero length")

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
        return get_rectangle_index(x, self.region_bounds)

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
