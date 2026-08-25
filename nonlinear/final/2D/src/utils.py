import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

from models import NN, NNOPT, get_masks_2d

device = "cpu"

def get_torch_dtype(args):
    if args.dtype == 32:
        return torch.float32
    if args.dtype == 64:
        return torch.float64
    raise ValueError("--dtype must be 32 or 64")


def get_numpy_dtype(args):
    if args.dtype == 32:
        return np.float32
    if args.dtype == 64:
        return np.float64
    raise ValueError("--dtype must be 32 or 64")


def load_region_edges(
    npz_path="region_edges.npz",
    np_dtype=np.float64,
):
    edges = np.load(npz_path)

    T_edges = np.asarray(
        edges["T_edges"],
        dtype=np_dtype,
    )

    C_edges = np.asarray(
        edges["C_edges"],
        dtype=np_dtype,
    )

    return T_edges, C_edges


def LoadData(args):
    if args.dataset_type != "cstr":
        raise ValueError("Dataset not supported!")

    torch_dtype = get_torch_dtype(args)
    np_dtype = get_numpy_dtype(args)

    dataset_arr, scaler = load_data(args.dataset_path)
    dataset_arr = dataset_arr.astype(
        np_dtype,
        copy=False,
    )

    T_edges_raw, C_edges_raw = load_region_edges(
        "region_edges.npz",
        np_dtype=np_dtype,
    )

    T_edges = (
        T_edges_raw / np_dtype(scaler.scale_[0])
    ).astype(np_dtype, copy=False)

    C_edges = (
        C_edges_raw / np_dtype(scaler.scale_[1])
    ).astype(np_dtype, copy=False)

    n_regions = (
        (len(T_edges) - 1)
        * (len(C_edges) - 1)
    )

    dataset = Data_cstr(
        dataset_arr,
        dtype=torch_dtype,
    )

    # Projection diagnostic should use every available point.
    if args.job == "projection_check":
        full_dataset = data.TensorDataset(
            dataset.X,
            dataset.Y,
        )

        dataset.train_set = full_dataset
        dataset.val_set = full_dataset
        dataset.test_set = full_dataset

        shuffle_loader = False
    else:
        dataset.resplit_data(args.val_ratio)
        
        # Add noise only during training
        if args.job == "train" and args.noise_level > 0.0:
            dataset.add_gaussian_noise_to_train(
                noise_level=args.noise_level,
                noise_seed=args.noise_seed,
            )

            print(
                f"Gaussian training noise: "
                f"level={args.noise_level}, "
                f"seed={args.noise_seed}"
            )
        shuffle_loader = True

    loader_args = {
        "batch_size": args.batch_size,
        "shuffle": shuffle_loader,
    }

    train_loader = data.DataLoader(
        dataset.train_set,
        **loader_args,
    )

    val_loader = data.DataLoader(
        dataset.val_set,
        **loader_args,
    )

    test_loader = data.DataLoader(
        dataset.test_set,
        **loader_args,
    )

    A_list, B_list, b_list = get_scaledABb_list(
        dataset.A_list,
        dataset.B_list,
        dataset.b_list,
        scaler,
    )

    assert n_regions == len(A_list), (
        f"{n_regions=} but {len(A_list)=}. "
        "Fix edges or ABb_matrices.csv."
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "dataset": dataset,
        "A_list": A_list,
        "B_list": B_list,
        "b_list": b_list,
        "T_edges": T_edges,
        "C_edges": C_edges,
        "scaler": scaler,
    }


def LoadModel(args, data):
    torch_dtype = get_torch_dtype(args)
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
            data["T_edges"],
            data["C_edges"],
        )
    else:
        raise ValueError("Model not supported!")

    return model.to(
        device=device,
        dtype=torch_dtype,
    )


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
    xscale, zscale = get_ScaleAndMean(
        scaler,
        A.shape[1],
        B.shape[1],
    )

    xscale = torch.as_tensor(
        xscale,
        dtype=A.dtype,
        device=A.device,
    )

    zscale = torch.as_tensor(
        zscale,
        dtype=B.dtype,
        device=B.device,
    )

    A_scaled = A * xscale
    B_scaled = B * zscale

    return A_scaled, B_scaled, b


def get_scaledABb_list(A_list, B_list, b_list, scaler):
    scaled = [
        get_scaledABb(A, B, b, scaler)
        for A, B, b in zip(A_list, B_list, b_list)
    ]
    return tuple(map(list, zip(*scaled)))


def load_ABb_from_csv(
    csv_path="ABb_matrices.csv",
    dtype=torch.float64,
):
    frame = (
        pd.read_csv(csv_path)
        .sort_values(["region_id", "constraint_order"])
        .reset_index(drop=True)
    )

    np_dtype = (
        np.float32
        if dtype == torch.float32
        else np.float64
    )

    A_list, B_list, b_list = [], [], []

    for _, region in frame.groupby("region_id", sort=True):
        A = torch.as_tensor(
            region[["A_T", "A_Cao"]].to_numpy(
                dtype=np_dtype
            ),
            dtype=dtype,
        )

        B = torch.as_tensor(
            region[["B_Ca", "B_Cb", "B_Cc"]].to_numpy(
                dtype=np_dtype
            ),
            dtype=dtype,
        )

        b = torch.as_tensor(
            region["b"].to_numpy(dtype=np_dtype),
            dtype=dtype,
        )

        A_list.append(A)
        B_list.append(B)
        b_list.append(b)

    return A_list, B_list, b_list


class Data_cstr(data.Dataset):
    def __init__(self, dataset, dtype=torch.float64):
        self.dataset_tensor = torch.as_tensor(dataset, dtype=dtype)
        self.X = self.dataset_tensor[:, :2]
        self.Y = self.dataset_tensor[:, 2:]
        self.train_set, self.val_set, self.test_set = self.split_data(0.2)
        self.A_list, self.B_list, self.b_list = load_ABb_from_csv("ABb_matrices.csv", dtype=dtype)
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
    def add_gaussian_noise_to_train(self, noise_level, noise_seed):
        """
        Add Gaussian noise ONLY to training targets Y.

        noise_level = 0.05 means:
        sigma_noise = 0.05 * std(Y_train)
        separately for each output.
        """

        if noise_level <= 0.0:
            return

        # Extract current CLEAN training data
        X_train = torch.stack([
            self.train_set[i][0]
            for i in range(len(self.train_set))
        ])

        Y_train = torch.stack([
            self.train_set[i][1]
            for i in range(len(self.train_set))
        ])

        # Per-output standard deviation
        Y_std = torch.std(Y_train, dim=0, unbiased=False)

        # Reproducible noise
        generator = torch.Generator()
        generator.manual_seed(noise_seed)

        noise = torch.randn(
            Y_train.shape,
            generator=generator,
            dtype=Y_train.dtype,
        )

        # Gaussian noisy observations
        Y_train_noisy = Y_train + noise_level * Y_std * noise

        # Replace ONLY training set
        # Validation and test sets stay clean
        self.train_set = data.TensorDataset(
            X_train,
            Y_train_noisy
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
    masks = get_masks_2d(X, data["T_edges"], data["C_edges"])
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
        active_violation += residual * masks[:, region:region + 1]

    return active_violation


# def compute_violation_original_nonlinear(
#     X_scaled, Ypred_scaled, scaler, device="cpu"
# ):
#     with torch.no_grad():
#         torch_dtype = X_scaled.dtype
#         scaled = torch.cat([X_scaled, Ypred_scaled], dim=1).cpu().numpy()
#         unscaled = scaler.inverse_transform(scaled)

#         T = unscaled[:, 0]
#         Cao = unscaled[:, 1]
#         Ca = unscaled[:, 2]
#         Cb = unscaled[:, 3]
#         Cc = unscaled[:, 4]

#         tau = 10.0
#         kf = 1e13 * np.exp(-90000.0 / (8.314 * T))
#         kr = 1e11 * np.exp(-80000.0 / (8.314 * T))

#         reaction = Cao - Ca - kf * Ca * Cb**2 * tau + kr * Cc * tau
#         mass_balance = -Cao + Ca + Cb + Cc - 2.0
#         violations = np.stack(
#             [np.abs(reaction), np.abs(mass_balance)], axis=1
#         )
#         return torch.tensor(violations, dtype=torch_dtype, device=device)

def compute_violation_original_nonlinear(
    X_scaled,
    Ypred_scaled,
    scaler,
    device="cpu",
):
    with torch.no_grad():
        torch_dtype = X_scaled.dtype

        scaled = torch.cat(
            [X_scaled, Ypred_scaled],
            dim=1,
        ).detach().cpu().numpy()

        unscaled = scaler.inverse_transform(scaled)

        T = unscaled[:, 0]
        Cao = unscaled[:, 1]
        Ca = unscaled[:, 2]
        Cb = unscaled[:, 3]
        Cc = unscaled[:, 4]

        tau_local = 10.0
        kf = 1.0e13 * np.exp(
            -90000.0 / (8.314 * T)
        )
        kr = 1.0e11 * np.exp(
            -80000.0 / (8.314 * T)
        )

        reaction = (
            Cao
            - Ca
            - kf * Ca * Cb**2 * tau_local
            + kr * Cc * tau_local
        )

        mass_balance = (
            -Cao + Ca + Cb + Cc - 2.0
        )

        violations = np.stack(
            [
                np.abs(reaction),
                np.abs(mass_balance),
            ],
            axis=1,
        )

        return torch.tensor(
            violations,
            dtype=torch_dtype,
            device=device,
        )