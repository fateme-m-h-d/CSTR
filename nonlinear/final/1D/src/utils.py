import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

from .models import NN, NNOPT
from .generate_data import Cbo, Cco, tau, kf_const, kr_const

device = "cpu"


def load_region_edges(npz_path="region_edges.npz"):
    arr = np.load(npz_path)
    return arr["C_edges"].astype(float)


def LoadData(args):
    if args.dataset_type != "cstr":
        raise ValueError("Dataset not supported!")

    dataset_arr, scaler = load_data(args.dataset_path)

    C_edges_raw = load_region_edges("region_edges.npz")
    C_edges = C_edges_raw / scaler.scale_[0]
    n_regions = len(C_edges) - 1

    if args.dtype == 32:
        dataset_arr = dataset_arr.astype(np.float32)

    dataset = Data_cstr(dataset_arr)

    if args.job == "projection_check":
        dataset.train_set = data.TensorDataset(dataset.X, dataset.Y)
        dataset.val_set = data.TensorDataset(dataset.X, dataset.Y)
        dataset.test_set = data.TensorDataset(dataset.X, dataset.Y)
    else:
        dataset.resplit_data(args.val_ratio)

    params = {
        "batch_size": args.batch_size,
        "shuffle": False if args.job == "projection_check" else True,
    }
    train_loader = data.DataLoader(dataset.train_set, **params)
    val_loader = data.DataLoader(dataset.val_set, **params)
    test_loader = data.DataLoader(dataset.test_set, **params)

    A_list, B_list, b_list = get_scaledABb_list(dataset.A_list, dataset.B_list, dataset.b_list, scaler)
    assert n_regions == len(A_list), f"{n_regions=} but {len(A_list)=}. Fix C_edges or ABb_matrices.csv."

    if args.dtype == 32:
        A_list = [A_i.float() for A_i in A_list]
        B_list = [B_i.float() for B_i in B_list]
        b_list = [b_i.float() for b_i in b_list]
    else:
        A_list = [A_i.double() for A_i in A_list]
        B_list = [B_i.double() for B_i in B_list]
        b_list = [b_i.double() for b_i in b_list]

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "dataset": dataset,
        "A_list": A_list,
        "B_list": B_list,
        "b_list": b_list,
        "C_edges": C_edges,
        "scaler": scaler,
    }


def LoadModel(args, data):
    if args.model == "NN":
        model = NN(args.input_dim, args.hidden_dim, args.hidden_num, args.z0_dim)
    elif args.model == "KKThPINN":
        model = NNOPT(args.input_dim, args.hidden_dim, args.hidden_num, args.z0_dim,
                      data["A_list"], data["B_list"], data["b_list"], data["C_edges"])
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


def load_data(dataset_path, save_scaler=False, scaler_path="results/scaler.pkl"):
    dataset = np.array(pd.read_csv(dataset_path).values)

    scaler = MaxAbsScaler()
    scaler.fit(dataset)
    dataset_scaled = scaler.transform(dataset)

    if save_scaler:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

    return dataset_scaled, scaler


def get_ScaleAndMean(scaler, x_dim, z_dim):
    xscale = [scaler.scale_[idx] for idx in range(x_dim)]
    zscale = [scaler.scale_[idx + x_dim] for idx in range(z_dim)]
    return xscale, zscale


def get_scaledABb(A, B, b, scaler):
    x_dim = A.shape[1]
    z_dim = B.shape[1]
    xscale, zscale = get_ScaleAndMean(scaler, x_dim, z_dim)
    xscale = torch.tensor(xscale)
    zscale = torch.tensor(zscale)
    A_scaled = A * torch.ones_like(A) * xscale
    B_scaled = B * torch.ones_like(B) * zscale
    return A_scaled, B_scaled, b


def get_scaledABb_list(A_list, B_list, b_list, scaler):
    scaled_As, scaled_Bs, scaled_bs = [], [], []
    for A, B, b in zip(A_list, B_list, b_list):
        A_s, B_s, b_s = get_scaledABb(A, B, b, scaler)
        scaled_As.append(A_s)
        scaled_Bs.append(B_s)
        scaled_bs.append(b_s)
    return scaled_As, scaled_Bs, scaled_bs


def load_ABb_from_csv(csv_path="ABb_matrices.csv"):
    df = (
        pd.read_csv(csv_path)
        .sort_values(["region_id", "constraint_order"])
        .reset_index(drop=True)
    )

    A_list, B_list, b_list = [], [], []

    for _, g in df.groupby("region_id", sort=True):
        A = torch.tensor(
            g[["A_Cao"]].to_numpy(dtype=float),
            dtype=torch.float32
        )

        B = torch.tensor(
            g[["B_Ca", "B_Cb", "B_Cc"]].to_numpy(dtype=float),
            dtype=torch.float32
        )

        b = torch.tensor(
            g["b"].to_numpy(dtype=float),
            dtype=torch.float32
        )

        A_list.append(A)
        B_list.append(B)
        b_list.append(b)

    return A_list, B_list, b_list


class Data_cstr(data.Dataset):
    def __init__(self, dataset):
        self.dataset_tensor = torch.from_numpy(dataset)
        self.X = self.dataset_tensor[:, :1]
        self.Y = self.dataset_tensor[:, 1:]
        self.train_set, self.val_set, self.test_set = self.split_data(0.2)
        self.A_list, self.B_list, self.b_list = load_ABb_from_csv("ABb_matrices.csv")
        self.constrained_indexes = []
        self.unconstrained_indexes = []

    def __len__(self):
        return len(self.dataset_tensor)

    def __getitem__(self, idx):
        return self.dataset_tensor[idx, :]

    def split_data(self, val_ratio, test_ratio=0.2):
        XY = data.TensorDataset(self.X, self.Y)
        XY = shuffle(XY, random_state=42)
        n_samples = len(XY)
        n_val = int(val_ratio * n_samples)
        n_test = int(test_ratio * n_samples)
        n_train = n_samples - n_val - n_test
        train_set = data.Subset(XY, range(0, n_train))
        val_set = data.Subset(XY, range(n_train, n_train + n_val))
        test_set = data.Subset(XY, range(n_train + n_val, n_samples))
        return train_set, val_set, test_set

    def resplit_data(self, val_ratio, test_ratio=0.2):
        self.train_set, self.val_set, self.test_set = self.split_data(val_ratio, test_ratio)


class PINNLoss(nn.Module):
    def __init__(self, A, B, b, mu, reduction="mean"):
        super(PINNLoss, self).__init__()
        self.A = A
        self.B = B
        self.b = b
        self.mu = mu
        self.reduction = reduction

    def forward(self, X, pred, target):
        mse_loss = F.mse_loss(pred, target, reduction=self.reduction)
        pinn_loss = torch.mean(self.mu * (torch.mm(self.B, pred.T) + torch.mm(self.A, X.T) - self.b.unsqueeze(1)) ** 2)
        return mse_loss, pinn_loss


class ALMLoss(nn.Module):
    def __init__(self, A, B, b, reduction="mean"):
        super(ALMLoss, self).__init__()
        self.A = A
        self.B = B
        self.b = b
        self.reduction = reduction

    def forward(self, X, pred, target, lambda_k, mu_k):
        mse_loss = F.mse_loss(pred, target, reduction=self.reduction)
        c = torch.mm(self.A, X.T) + torch.mm(self.B, pred.T) - self.b.repeat(1, X.T.shape[1])
        lambda_c = torch.mm(lambda_k.unsqueeze(0), c).mean()
        mu_c = mu_k / 2 * c.pow(2).mean()
        return mse_loss, lambda_c + mu_c


def get_masks_1d(X, C_edges):
    C_edges = torch.as_tensor(C_edges, dtype=X.dtype, device=X.device)
    n_regions = len(C_edges) - 1
    if n_regions == 1:
        return torch.ones((X.shape[0], 1), dtype=X.dtype, device=X.device)

    c = X[:, 0]
    masks = torch.zeros((X.shape[0], n_regions), dtype=X.dtype, device=X.device)
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


def get_violation(args, data, X, pred):
    A_list, B_list, b_list = data["A_list"], data["B_list"], data["b_list"]
    C_edges = torch.as_tensor(data["C_edges"], dtype=X.dtype, device=X.device)
    masks = get_masks_1d(X, C_edges)

    n_constraints = b_list[0].numel()
    active_violation = torch.zeros(
        (X.shape[0], n_constraints),
        dtype=X.dtype,
        device=X.device,
    )

    for r, (Ai, Bi, bi) in enumerate(zip(A_list, B_list, b_list)):
        Ai = Ai.to(dtype=X.dtype, device=X.device)
        Bi = Bi.to(dtype=X.dtype, device=X.device)
        bi = bi.to(dtype=X.dtype, device=X.device).view(1, -1)

        v = X @ Ai.T + pred @ Bi.T - bi
        active_violation = active_violation + v * masks[:, r:r+1]

    return active_violation


def compute_violation_original_nonlinear(X_scaled, Ypred_scaled, scaler, device="cpu"):
    with torch.no_grad():
        torch_dtype = X_scaled.dtype

        XYs = torch.cat([X_scaled, Ypred_scaled], dim=1).detach().cpu().numpy()
        XY = scaler.inverse_transform(XYs)

        Cao = XY[:, 0]
        Ca = XY[:, 1]
        Cb = XY[:, 2]
        Cc = XY[:, 3]

        eq_rxn = (
            Cao - Ca
            - kf_const * Ca * (Cb ** 2) * tau
            + kr_const * Cc * tau
        )

        eq_mb = (
            -Cao + Ca + Cb + Cc
            - (Cbo + Cco)
        )

        viol = np.stack([np.abs(eq_rxn), np.abs(eq_mb)], axis=1)

        return torch.tensor(viol, dtype=torch_dtype, device=device)
