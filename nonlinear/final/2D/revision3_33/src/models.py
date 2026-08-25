import torch
import torch.nn as nn
import torch.nn.functional as F


def get_rectangle_masks(X, region_bounds):
    """Hard one-hot membership for arbitrary non-overlapping rectangles.

    ``region_bounds`` has rows [T_low, T_high, C_low, C_high] in the same
    (possibly scaled) coordinates as X.

    Shared internal boundaries are handled as half-open intervals.  Rectangles
    touching the global upper domain edge include that upper edge.  If an input
    lies slightly outside the domain due to numerical/extrapolation effects, it
    is assigned to the nearest rectangle rather than producing an all-zero mask.
    """

    bounds = torch.as_tensor(
        region_bounds, dtype=X.dtype, device=X.device
    )
    if bounds.ndim != 2 or bounds.shape[1] != 4:
        raise ValueError(
            "region_bounds must have shape (n_regions, 4) with "
            "[T_low, T_high, C_low, C_high]"
        )

    T = X[:, 0:1]
    C = X[:, 1:2]
    T_low = bounds[:, 0].view(1, -1)
    T_high = bounds[:, 1].view(1, -1)
    C_low = bounds[:, 2].view(1, -1)
    C_high = bounds[:, 3].view(1, -1)

    global_T_high = torch.max(bounds[:, 1])
    global_C_high = torch.max(bounds[:, 3])
    touches_T_max = torch.isclose(bounds[:, 1], global_T_high).view(1, -1)
    touches_C_max = torch.isclose(bounds[:, 3], global_C_high).view(1, -1)

    T_upper_ok = (T < T_high) | (touches_T_max & (T <= T_high))
    C_upper_ok = (C < C_high) | (touches_C_max & (C <= C_high))
    inside = (T >= T_low) & T_upper_ok & (C >= C_low) & C_upper_ok

    # Select one rectangle deterministically. For a valid partition each in-domain
    # point already has exactly one True entry. If numerical overlap occurs at a
    # boundary, argmax chooses the first region.
    has_match = inside.any(dim=1)
    region_index = inside.to(dtype=X.dtype).argmax(dim=1)

    # Robust fallback for points outside the partitioned domain: nearest box.
    if not bool(torch.all(has_match)):
        dT = torch.relu(T_low - T) + torch.relu(T - T_high)
        dC = torch.relu(C_low - C) + torch.relu(C - C_high)
        distance2 = dT**2 + dC**2
        nearest = distance2.argmin(dim=1)
        region_index = torch.where(has_match, region_index, nearest)

    return F.one_hot(region_index, num_classes=bounds.shape[0]).to(dtype=X.dtype)


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
        self.register_buffer("region_bounds", torch.as_tensor(region_bounds))

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
            fc2 = nn.Linear(input_dim, z0_dim, bias=False)
            fc2.weight = nn.Parameter(Astar, requires_grad=False)
            fc2.bias = nn.Parameter(bstar, requires_grad=False)
            self.fc1_list.append(fc1)
            self.fc2_list.append(fc2)

    def forward(self, x):
        hidden = x
        for layer in self.layers[:-1]:
            hidden = F.relu(layer(hidden))
        z0 = self.layers[-1](hidden)

        masks = get_rectangle_masks(x, self.region_bounds)
        output = torch.zeros_like(z0)
        for region, (fc1, fc2) in enumerate(
            zip(self.fc1_list, self.fc2_list)
        ):
            fixed_output = fc1(z0) + fc2(x)
            output = output + fixed_output * masks[:, region:region + 1]
        return output

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
