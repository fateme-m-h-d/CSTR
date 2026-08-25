import torch
import torch.nn as nn
import torch.nn.functional as F


def get_box_masks(X, region_lows, region_highs):
    """Return one hard mask per adaptive hyper-rectangular leaf.

    Regions are lower-inclusive and upper-exclusive. A region touching the
    global upper domain boundary is upper-inclusive in that dimension. This
    gives each in-domain point exactly one active region, including points on
    shared split boundaries.
    """

    lows = torch.as_tensor(region_lows, dtype=X.dtype, device=X.device)
    highs = torch.as_tensor(region_highs, dtype=X.dtype, device=X.device)
    if lows.ndim != 2 or highs.ndim != 2 or lows.shape != highs.shape:
        raise ValueError("region_lows and region_highs must have shape [R, d].")
    if X.shape[1] != lows.shape[1]:
        raise ValueError(
            f"Input dimension {X.shape[1]} does not match region dimension {lows.shape[1]}."
        )

    global_low = torch.min(lows, dim=0).values
    global_high = torch.max(highs, dim=0).values
    scale = torch.maximum(global_high - global_low, torch.ones_like(global_high))
    tolerance = 10.0 * torch.finfo(X.dtype).eps * scale

    X_expanded = X[:, None, :]
    lower_ok = X_expanded >= (lows[None, :, :] - tolerance[None, None, :])

    touches_global_high = (
        torch.abs(highs - global_high[None, :]) <= tolerance[None, :]
    )
    upper_strict = X_expanded < highs[None, :, :]
    upper_global = (
        touches_global_high[None, :, :]
        & (X_expanded <= highs[None, :, :] + tolerance[None, None, :])
    )
    upper_ok = upper_strict | upper_global

    masks = (lower_ok & upper_ok).all(dim=2).to(dtype=X.dtype)

    # For an out-of-domain point, follow the old repository behavior in spirit:
    # assign the nearest leaf instead of returning an all-zero prediction.
    unmatched = masks.sum(dim=1) == 0
    if torch.any(unmatched):
        centers = 0.5 * (lows + highs)
        normalized_distance = (
            (X[unmatched, None, :] - centers[None, :, :]) / scale[None, None, :]
        ) ** 2
        nearest = torch.argmin(normalized_distance.sum(dim=2), dim=1)
        masks[unmatched] = F.one_hot(
            nearest, num_classes=lows.shape[0]
        ).to(dtype=X.dtype)

    # This should not occur for a valid binary box partition. Keep the first
    # matching region deterministically if floating-point tolerances overlap.
    multiple = masks.sum(dim=1) > 1
    if torch.any(multiple):
        first = torch.argmax(masks[multiple], dim=1)
        masks[multiple] = F.one_hot(
            first, num_classes=lows.shape[0]
        ).to(dtype=X.dtype)

    return masks


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
        region_lows,
        region_highs,
    ):
        super().__init__()
        self.register_buffer("region_lows", torch.as_tensor(region_lows))
        self.register_buffer("region_highs", torch.as_tensor(region_highs))

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(hidden_num - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, z0_dim))

        self.fc1_list = nn.ModuleList()
        self.fc2_list = nn.ModuleList()
        for A, B, b in zip(A_list, B_list, b_list):
            # Keep the repository's exact closed-form projection.
            gram_inverse = torch.linalg.inv(B @ B.t())
            chunk = B.t() @ gram_inverse
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

        masks = get_box_masks(x, self.region_lows, self.region_highs)
        output = torch.zeros_like(z0)
        for region, (fc1, fc2) in enumerate(zip(self.fc1_list, self.fc2_list)):
            fixed_output = fc1(z0) + fc2(x)
            output = output + fixed_output * masks[:, region : region + 1]
        return output

    def reset_parameters(self):
        # Projection layers are fixed and must not be reinitialized.
        for layer in self.layers:
            layer.reset_parameters()
