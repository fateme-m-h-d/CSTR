import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

from models import NN, NNOPT, get_box_masks


device = "cpu"


def _legacy_grid_to_boxes(T_edges, C_edges):
    lows = []
    highs = []
    for i in range(len(T_edges) - 1):
        for j in range(len(C_edges) - 1):
            lows.append([T_edges[i], C_edges[j]])
            highs.append([T_edges[i + 1], C_edges[j + 1]])
    return np.asarray(lows, dtype=float), np.asarray(highs, dtype=float)


def load_region_bounds(
    bounds_path="region_bounds.npz",
    legacy_edges_path="region_edges.npz",
):
    """Load adaptive boxes, with backward compatibility for the old grid."""

    if os.path.exists(bounds_path):
        bounds = np.load(bounds_path)
        lows = bounds["lows"].astype(float)
        highs = bounds["highs"].astype(float)
        if lows.shape != highs.shape or lows.ndim != 2:
            raise ValueError("Invalid region_bounds.npz: lows/highs must be [R, d].")
        return lows, highs

    if os.path.exists(legacy_edges_path):
        edges = np.load(legacy_edges_path)
        return _legacy_grid_to_boxes(
            edges["T_edges"].astype(float),
            edges["C_edges"].astype(float),
        )

    raise FileNotFoundError(
        "Neither region_bounds.npz nor legacy region_edges.npz was found. "
        "Run src.linearization first."
    )


def LoadData(args):
    if args.dataset_type != "cstr":
        raise ValueError("Dataset not supported!")

    dataset_arr, scaler = load_data(args.dataset_path)
    region_lows_raw, region_highs_raw = load_region_bounds()

    input_scale = np.asarray(scaler.scale_[: args.input_dim], dtype=float)
    region_lows = region_lows_raw / input_scale[None, :]
    region_highs = region_highs_raw / input_scale[None, :]
    n_regions = region_lows.shape[0]

    if args.dtype == 32:
        dataset_arr = dataset_arr.astype(np.float32)

    dataset = Data_cstr(dataset_arr)
    dataset.resplit_data(args.val_ratio)

    loader_args = {"batch_size": args.batch_size, "shuffle": True}
    train_loader = data.DataLoader(dataset.train_set, **loader_args)
    val_loader = data.DataLoader(dataset.val_set, **loader_args)
    test_loader = data.DataLoader(dataset.test_set, **loader_args)

    A_list, B_list, b_list = get_scaledABb_list(
        dataset.A_list, dataset.B_list, dataset.b_list, scaler
    )
    assert n_regions == len(A_list), (
        f"{n_regions=} but {len(A_list)=}. Fix region_bounds.npz or "
        "ABb_matrices.csv."
    )

    if args.dtype == 32:
        A_list = [A.float() for A in A_list]
        B_list = [B.float() for B in B_list]
        b_list = [b.float() for b in b_list]
        region_lows = region_lows.astype(np.float32)
        region_highs = region_highs.astype(np.float32)
    else:
        A_list = [A.double() for A in A_list]
        B_list = [B.double() for B in B_list]
        b_list = [b.double() for b in b_list]
        region_lows = region_lows.astype(np.float64)
        region_highs = region_highs.astype(np.float64)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "dataset": dataset,
        "A_list": A_list,
        "B_list": B_list,
        "b_list": b_list,
        "region_lows": region_lows,
        "region_highs": region_highs,
        "scaler": scaler,
    }


def LoadModel(args, data):
    if args.model == "NN":
        model = NN(args.input_dim, args.hidden_dim, args.hidden_num, args.z0_dim)
    elif args.model == "KKThPINN":
        model = NNOPT(
            args.input_dim,
            args.hidden_dim,
            args.hidden_num,
            args.z0_dim,
            data["A_list"],
            data["B_list"],
            data["b_list"],
            data["region_lows"],
            data["region_highs"],
        )
    else:
        raise ValueError("Model not supported!")

    if args.dtype == 32:
        return model.to(device)
    return model.double().to(device)


def get_optimizer(args, model):
    if args.optimizer == "adam":
        return optim.Adam(model.parameters(), lr=args.lr)
    if args.optimizer == "SGD":
        return optim.SGD(model.parameters(), lr=args.lr)
    raise ValueError("Invalid optimizer")


def get_loss_func(args, data):
    if args.loss_type == "MSE":
        return nn.MSELoss()
    if args.loss_type == "PINN":
        return PINNLoss(data["A"], data["B"], data["b"], args.mu)
    if args.loss_type == "ALM":
        return ALMLoss(data["A"], data["B"], data["b"])
    raise ValueError("Loss function not supported!")


def load_data(dataset_path):
    dataset = np.asarray(pd.read_csv(dataset_path).values)
    scaler = MaxAbsScaler()
    dataset_scaled = scaler.fit_transform(dataset)
    return dataset_scaled, scaler


def get_ScaleAndMean(scaler, x_dim, z_dim):
    xscale = [scaler.scale_[idx] for idx in range(x_dim)]
    zscale = [scaler.scale_[idx + x_dim] for idx in range(z_dim)]
    return xscale, zscale


def get_scaledABb(A, B, b, scaler):
    xscale, zscale = get_ScaleAndMean(scaler, A.shape[1], B.shape[1])
    xscale = torch.tensor(xscale)
    zscale = torch.tensor(zscale)
    A_scaled = A * torch.ones_like(A) * xscale
    B_scaled = B * torch.ones_like(B) * zscale
    return A_scaled, B_scaled, b


def get_scaledABb_list(A_list, B_list, b_list, scaler):
    scaled = [
        get_scaledABb(A, B, b, scaler)
        for A, B, b in zip(A_list, B_list, b_list)
    ]
    return tuple(map(list, zip(*scaled)))


def load_ABb_from_csv(csv_path="ABb_matrices.csv"):
    frame = (
        pd.read_csv(csv_path)
        .sort_values(["region_id", "constraint_order"])
        .reset_index(drop=True)
    )
    A_list, B_list, b_list = [], [], []
    for _, region in frame.groupby("region_id", sort=True):
        A_list.append(
            torch.tensor(
                region[["A_T", "A_Cao"]].to_numpy(dtype=float),
                dtype=torch.float32,
            )
        )
        B_list.append(
            torch.tensor(
                region[["B_Ca", "B_Cb", "B_Cc"]].to_numpy(dtype=float),
                dtype=torch.float32,
            )
        )
        b_list.append(
            torch.tensor(region["b"].to_numpy(dtype=float), dtype=torch.float32)
        )

    return A_list, B_list, b_list


class Data_cstr(data.Dataset):
    def __init__(self, dataset):
        self.dataset_tensor = torch.from_numpy(dataset)
        self.X = self.dataset_tensor[:, :2]
        self.Y = self.dataset_tensor[:, 2:]
        self.train_set, self.val_set, self.test_set = self.split_data(0.2)
        self.A_list, self.B_list, self.b_list = load_ABb_from_csv()
        self.constrained_indexes = []
        self.unconstrained_indexes = []

    def __len__(self):
        return len(self.dataset_tensor)

    def __getitem__(self, idx):
        return self.dataset_tensor[idx, :]

    def split_data(self, val_ratio, test_ratio=0.2):
        samples = data.TensorDataset(self.X, self.Y)
        samples = shuffle(samples, random_state=42)
        n_samples = len(samples)
        n_val = int(val_ratio * n_samples)
        n_test = int(test_ratio * n_samples)
        n_train = n_samples - n_val - n_test
        train_set = data.Subset(samples, range(0, n_train))
        val_set = data.Subset(samples, range(n_train, n_train + n_val))
        test_set = data.Subset(samples, range(n_train + n_val, n_samples))
        return train_set, val_set, test_set

    def resplit_data(self, val_ratio, test_ratio=0.2):
        self.train_set, self.val_set, self.test_set = self.split_data(
            val_ratio, test_ratio
        )


class PINNLoss(nn.Module):
    def __init__(self, A, B, b, mu, reduction="mean"):
        super().__init__()
        self.A = A
        self.B = B
        self.b = b
        self.mu = mu
        self.reduction = reduction

    def forward(self, X, pred, target):
        mse_loss = F.mse_loss(pred, target, reduction=self.reduction)
        residual = torch.mm(self.B, pred.T) + torch.mm(self.A, X.T)
        pinn_loss = torch.mean(self.mu * (residual - self.b.unsqueeze(1)) ** 2)
        return mse_loss, pinn_loss


class ALMLoss(nn.Module):
    def __init__(self, A, B, b, reduction="mean"):
        super().__init__()
        self.A = A
        self.B = B
        self.b = b
        self.reduction = reduction

    def forward(self, X, pred, target, lambda_k, mu_k):
        mse_loss = F.mse_loss(pred, target, reduction=self.reduction)
        c = (
            torch.mm(self.A, X.T)
            + torch.mm(self.B, pred.T)
            - self.b.repeat(1, X.T.shape[1])
        )
        lambda_c = torch.mm(lambda_k.unsqueeze(0), c).mean()
        mu_c = mu_k / 2 * c.pow(2).mean()
        return mse_loss, lambda_c + mu_c


def get_violation(args, data, X, pred):
    masks = get_box_masks(X, data["region_lows"], data["region_highs"])
    assert masks.shape[1] == len(data["A_list"]), (
        f"Mask count {masks.shape[1]} != number of regions "
        f"{len(data['A_list'])}"
    )

    n_constraints = data["b_list"][0].numel()
    active_violation = torch.zeros(
        (X.shape[0], n_constraints), dtype=X.dtype, device=X.device
    )
    for region, (A, B, b) in enumerate(
        zip(data["A_list"], data["B_list"], data["b_list"])
    ):
        A = A.to(dtype=X.dtype, device=X.device)
        B = B.to(dtype=X.dtype, device=X.device)
        b = b.to(dtype=X.dtype, device=X.device).view(1, -1)
        residual = X @ A.T + pred @ B.T - b
        active_violation += residual * masks[:, region : region + 1]

    return active_violation


def compute_violation_original_nonlinear(
    X_scaled, Ypred_scaled, scaler, device="cpu"
):
    with torch.no_grad():
        torch_dtype = X_scaled.dtype
        scaled = torch.cat([X_scaled, Ypred_scaled], dim=1).cpu().numpy()
        unscaled = scaler.inverse_transform(scaled)

        T = unscaled[:, 0]
        Cao = unscaled[:, 1]
        Ca = unscaled[:, 2]
        Cb = unscaled[:, 3]
        Cc = unscaled[:, 4]

        tau = 10.0
        kf = 1e13 * np.exp(-90000.0 / (8.314 * T))
        kr = 1e11 * np.exp(-80000.0 / (8.314 * T))

        reaction = Cao - Ca - kf * Ca * Cb**2 * tau + kr * Cc * tau
        mass_balance = -Cao + Ca + Cb + Cc - 2.0
        violations = np.stack(
            [np.abs(reaction), np.abs(mass_balance)], axis=1
        )
        return torch.tensor(violations, dtype=torch_dtype, device=device)
