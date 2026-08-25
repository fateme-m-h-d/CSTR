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
    T_masks = get_masks_1d(X[:, 0], T_edges)
    C_masks = get_masks_1d(X[:, 1], C_edges)
    masks = [
        T_masks[:, i:i + 1] * C_masks[:, j:j + 1]
        for i in range(T_masks.shape[1])
        for j in range(C_masks.shape[1])
    ]
    return torch.cat(masks, dim=1)


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
        self.register_buffer("T_edges", torch.as_tensor(T_edges))
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
            Bstar = torch.eye(z0_dim) - chunk @ B
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

        masks = get_masks_2d(x, self.T_edges, self.C_edges)
        output = torch.zeros_like(z0)
        for region, (fc1, fc2) in enumerate(
            zip(self.fc1_list, self.fc2_list)
        ):
            fixed_output = fc1(z0) + fc2(x)
            output = output + fixed_output * masks[:, region:region + 1]
        return output

    def reset_parameters(self):
        for module in self.modules():
            if (
                isinstance(module, nn.Linear)
                and module is not self.fc1_list
                and module is not self.fc2_list
            ):
                module.reset_parameters()
