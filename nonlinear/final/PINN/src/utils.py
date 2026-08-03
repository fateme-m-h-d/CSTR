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

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"


def load_region_edges(npz_path="region_edges.npz"):
    arr = np.load(npz_path)
    return arr["C_edges"].astype(float)


def LoadData(args):
    if args.dataset_type != "cstr":
        raise ValueError("Dataset not supported!")

    dataset_arr, scaler = load_data(args.dataset_path)

    # C_edges_raw = load_region_edges("region_edges.npz")
    # C_edges = C_edges_raw / scaler.scale_[0]  # feature 0 = Cao
    # n_regions = len(C_edges) - 1
    if args.dtype == 32:
        dataset_arr = dataset_arr.astype(np.float32)
        
    load_kkt = args.model == "KKThPINN"
    dataset = Data_cstr(dataset_arr, load_kkt=load_kkt)

    # dataset = Data_cstr(dataset_arr)
    # dataset.resplit_data(args.val_ratio)

    # params = {"batch_size": args.batch_size, "shuffle": True}

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


    data_dict = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "dataset": dataset,
        "scaler": scaler,
    }

    if args.model == "KKThPINN":
        C_edges_raw = load_region_edges("region_edges.npz")
        C_edges = C_edges_raw / scaler.scale_[0]
        n_regions = len(C_edges) - 1

        A_list, B_list, b_list = get_scaledABb_list(
            dataset.A_list, dataset.B_list, dataset.b_list, scaler
        )

        assert n_regions == len(A_list), (
            f"{n_regions=} but {len(A_list)=}. Fix C_edges or ABb_matrices.csv."
        )

        if args.dtype == 32:
            A_list = [A_i.float() for A_i in A_list]
            B_list = [B_i.float() for B_i in B_list]
            b_list = [b_i.float() for b_i in b_list]
        else:
            A_list = [A_i.double() for A_i in A_list]
            B_list = [B_i.double() for B_i in B_list]
            b_list = [b_i.double() for b_i in b_list]

        data_dict.update({
            "A_list": A_list,
            "B_list": B_list,
            "b_list": b_list,
            "C_edges": C_edges,
        })

    return data_dict

def LoadModel(args, data):
    if args.model == "NN":
        model = NN(args.input_dim, args.hidden_dim, args.hidden_num, args.z0_dim)
    elif args.model == 'PINN':
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
        return PINNLoss(data["scaler"], args.mu_rxn, args.mu_mb)
    if args.loss_type == "ALM":
        print("ALM loss function is used!")
        return ALMLoss(data["A"], data["B"], data["b"])
    raise ValueError("Loss function not supported!")


def load_data(dataset_path):
    dataset = np.array(pd.read_csv(dataset_path).values)
    scaler = MaxAbsScaler()
    scaler.fit(dataset)
    dataset = scaler.transform(dataset)
    return dataset, scaler


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
    df = pd.read_csv(csv_path).sort_values(["region_id"]).reset_index(drop=True)

    A_list, B_list, b_list = [], [], []

    for region_id, group in df.groupby("region_id", sort=True):
        A = torch.tensor(
            group[["A_Cao"]].to_numpy(),
            dtype=torch.float32,
        )

        B = torch.tensor(
            group[["B_Ca", "B_Cb", "B_Cc"]].to_numpy(),
            dtype=torch.float32,
        )

        b = torch.tensor(
            group["b"].to_numpy(),
            dtype=torch.float32,
        )

        A_list.append(A)
        B_list.append(B)
        b_list.append(b)

    return A_list, B_list, b_list


class Data_cstr(data.Dataset):
    def __init__(self, dataset, load_kkt=False):
        self.dataset_tensor = torch.from_numpy(dataset)
        self.X = self.dataset_tensor[:, :1]   # Cao only
        self.Y = self.dataset_tensor[:, 1:]   # Ca, Cb, Cc
        self.train_set, self.val_set, self.test_set = self.split_data(0.2)
        self.A_list, self.B_list, self.b_list = None, None, None
        if load_kkt:
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
    def __init__(self, scaler, mu_rxn, mu_mb, reduction="mean"):
        super(PINNLoss, self).__init__()
        self.scale_np = scaler.scale_.copy()
        self.mu_rxn = mu_rxn
        self.mu_mb = mu_mb
        self.reduction = reduction

    def forward(self, X, pred, target):
        mse_loss = F.mse_loss(pred, target, reduction=self.reduction)
        # pinn_loss = torch.mean(self.mu * (torch.mm(self.B, pred.T) + torch.mm(self.A, X.T) - self.b.unsqueeze(1)) ** 2)
        
        # convert scaled variables back to original physical units
        scale = torch.as_tensor(self.scale_np, dtype=pred.dtype, device=pred.device)

        Cao = X[:, 0] * scale[0]
        Ca = pred[:, 0] * scale[1]
        Cb = pred[:, 1] * scale[2]
        Cc = pred[:, 2] * scale[3]

        # nonlinear physical residual:
        # eq1 = Cao - Ca - kf_const * Ca * Cb^2 * tau + kr_const * Cc * tau
        # residual = (
        #     Cao
        #     - Ca
        #     - kf_const * Ca * (Cb ** 2) * tau
        #     + kr_const * (Cc) * tau
        # )
        r_rxn = (
            Cao
            - Ca
            - kf_const * Ca * (Cb ** 2) * tau
            + kr_const * Cc * tau
            )

        r_mb = (
            Cc
            - Cao
            + Ca
            - Cbo
            + Cb
            - Cco
        )

        pinn_loss = (
            self.mu_rxn * torch.mean(r_rxn ** 2)
            + self.mu_mb * torch.mean(r_mb ** 2)
        )
        
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





def get_violation(args, data, X, pred):
    """
    Returns the original nonlinear residual in physical units:

    eq1 = Cao - Ca - kf_const * Ca * Cb^2 * tau + kr_const * Cc * tau

    X and pred are scaled, so we first convert them back to original units.
    Output shape: (batch, 1)
    """

    scale = torch.as_tensor(
        data["scaler"].scale_,
        dtype=X.dtype,
        device=X.device
    )

    # Unscale inputs and predictions
    Cao = X[:, 0] * scale[0]

    Ca = pred[:, 0] * scale[1]
    Cb = pred[:, 1] * scale[2]
    Cc = pred[:, 2] * scale[3]

    r_rxn = (
        Cao
        - Ca
        - kf_const * Ca * (Cb ** 2) * tau
        + kr_const * Cc * tau
    )

    r_mb = (
        Cc
        - Cao
        + Ca
        - Cbo
        + Cb
        - Cco
    )

    return torch.stack([r_rxn, r_mb], dim=1)
