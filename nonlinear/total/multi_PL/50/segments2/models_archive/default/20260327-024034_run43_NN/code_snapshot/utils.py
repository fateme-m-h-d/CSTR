import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from models import NN, NNOPT
from sklearn.utils import shuffle

# device = "cuda" if torch.cuda.is_available() else "cpu"    #GPU OR CPU
device = "cpu"
print("Using device:", device)

def load_region_edges(npz_path="region_edges.npz"):
    arr = np.load(npz_path)
    return arr["T_edges"].astype(float), arr["C_edges"].astype(float)

def LoadData(args):
    if args.dataset_type == 'cstr':
        dataset_arr, scaler = load_data(args.dataset_path)
        # raw edges in ORIGINAL units
        # T_edges_raw = np.array([280, 300, 340, 360, 400, 420, 440, 460], dtype=float)
        # C_edges_raw = np.linspace(0.8, 1.2, 4)  # example: 3 Cao-bins
        T_edges_raw, C_edges_raw = load_region_edges("region_edges.npz")

        # scale them using the same scaler used on the dataset
        T_edges = T_edges_raw / scaler.scale_[0]   # feature 0 = T
        C_edges = C_edges_raw / scaler.scale_[1]   # feature 1 = Cao

        # sanity: region count must match constraints
        n_regions = (len(T_edges) - 1) * (len(C_edges) - 1)
        Data_class = Data_cstr
    else:
        raise ValueError('Dataset not supported!')

    if args.dtype == 32:
        dataset_arr = dataset_arr.astype(np.float32)
    dataset = Data_class(dataset_arr)
    dataset.resplit_data(args.val_ratio)

    # A, B, b = get_scaledABb(dataset.A, dataset.B, dataset.b, scaler)

    # if args.dtype == 32:
    #     A, B, b = A.float(), B.float(), b.float()
    # else:
    #     A, B, b = A.double(), B.double(), b.double()
    # print(f'type of A: {A.dtype}, type of B: {B.dtype}, type of b: {b.dtype}')


    params = {'batch_size': args.batch_size,
              'shuffle': True}
    train_loader = data.DataLoader(dataset.train_set, **params)
    val_loader = data.DataLoader(dataset.val_set, **params)
    test_loader = data.DataLoader(dataset.test_set, **params)

    print(f'train set size: {len(dataset.train_set)}, val set size: {len(dataset.val_set)}, test set size: {len(dataset.test_set)}')
    
    A_list, B_list, b_list = get_scaledABb_list(dataset.A_list, dataset.B_list,
                                       dataset.b_list, scaler)
    
    assert n_regions == len(A_list), f"{n_regions=} but {len(A_list)=}. Fix edges or A_list order."
    
    
        # --- cast each fixed‐layer tensor to the same dtype as A,B,b above ---
    if args.dtype == 32:
        A_list = [A_i.float() for A_i in A_list]
        B_list = [B_i.float() for B_i in B_list]
        b_list = [b_i.float() for b_i in b_list]
    else:
        A_list = [A_i.double() for A_i in A_list]
        B_list = [B_i.double() for B_i in B_list]
        b_list = [b_i.double() for b_i in b_list]
    # Debug check
    # print("Dtypes after casting:")
    # print([A_i.dtype for A_i in A_list])
    # print([B_i.dtype for B_i in B_list])
    # print([b_i.dtype for b_i in b_list])


    data_dict = {'train_loader': train_loader, 'val_loader': val_loader, 'test_loader': test_loader,
                 'dataset': dataset, 'A_list': A_list, 'B_list': B_list, 'b_list': b_list, "T_edges": T_edges,
    "C_edges": C_edges,
                }
    return data_dict


def LoadModel(args, data):
    if args.model == 'NN':
        model = NN(args.input_dim, args.hidden_dim, args.hidden_num, args.z0_dim)
    elif args.model == 'KKThPINN':
        model = NNOPT(args.input_dim, args.hidden_dim, args.hidden_num, args.z0_dim,
                      data['A_list'], data['B_list'], data['b_list'], data['T_edges'], data['C_edges'])
    else:
        raise ValueError('Model not supported!')

    if args.dtype == 32:
        model = model.to(device)
    else:
        model = model.double().to(device)
    return model


def get_optimizer(args, model):
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else:
        raise ValueError('Invalid optimizer')

    return optimizer


def get_loss_func(args, data):
    if args.loss_type == 'MSE':
        loss_func = nn.MSELoss()
    elif args.loss_type == 'PINN':
        loss_func = PINNLoss(data['A'], data['B'], data['b'], args.mu)
    elif args.loss_type == 'ALM':
        loss_func = ALMLoss(data['A'], data['B'], data['b'])
        print('ALM loss function is used!')
    else:
        raise ValueError('Loss function not supported!')
    return loss_func

import pickle
import os
def load_data(dataset_path):
    dataset = np.array(pd.read_csv(dataset_path).values)
    scaler = MaxAbsScaler()
    scaler.fit(dataset)
    # Manually set a higher max value for temperature input
    # scaler.scale_[0] = max(scaler.scale_[0], 800)  # Assuming temperature is the first feature
    
    print("Scaler Factors:", scaler.scale_)

    dataset = scaler.transform(dataset)
    
    # Save the scaler to use it later during predictions
    base_dir = os.getcwd()  #new line of the code. need to change it.
    scaler_path = os.path.join(base_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved at {scaler_path}")
    
    # # Load the scaler from the .pkl file
    # with open(scaler_path, 'rb') as f:
    #     scaler = pickle.load(f)

    # # Inspect the scaler object
    # print("Scaler Type:", type(scaler))
    # if hasattr(scaler, 'scale_'):
    #     print("Scaler Factors (scale_):", scaler.scale_)
    # if hasattr(scaler, 'min_'):
    #     print("Scaler Min Values (min_):", scaler.min_)
    # if hasattr(scaler, 'data_min_'):
    #     print("Data Min Values (data_min_):", scaler.data_min_)
    # if hasattr(scaler, 'data_max_'):
    #     print("Data Max Values (data_max_):", scaler.data_max_)
    # if hasattr(scaler, 'n_features_in_'):
    #     print("Number of Features:", scaler.n_features_in_)

    # Save the transformed dataset to a CSV file
    scaled_dataset_path = os.path.join(os.getcwd(), 'scaled_data.csv')
    dataset_scaled_df = pd.DataFrame(dataset)
    dataset_scaled_df.to_csv(scaled_dataset_path, index=False)
    print(f"Scaled dataset saved at {scaled_dataset_path}")
    return dataset, scaler


# load_data("./data.csv")
dataset, scaler = load_data("./data.csv")
# Print scaler factors outside the function
print("Scaler Factors outside function:", scaler.scale_)


def get_ScaleAndMean(scaler, x_dim, z_dim):
    xscale = []
    zscale = []
    for idx in range(x_dim):
        xscale.append(scaler.scale_[idx])
    for idx in range(z_dim):
        zscale.append((scaler.scale_[idx+x_dim]))
    return xscale, zscale


def get_scaledABb(A, B, b, scaler):
    x_dim = A.shape[1]
    z_dim = B.shape[1]
    xscale, zscale = get_ScaleAndMean(scaler, x_dim, z_dim)
    xscale, zscale = torch.tensor(xscale), torch.tensor(zscale)
    A_scale = torch.ones_like(A) * xscale
    B_scale = torch.ones_like(B) * zscale
    A_scaled = A * A_scale
    B_scaled = B * B_scale
    b_scaled = b
    return A_scaled, B_scaled, b_scaled

def get_scaledABb_list(A_list, B_list, b_list, scaler):
    """
    Scale each (A, B, b) in your lists using your existing
    get_ScaleAndMean and get_scaledABb functions.
    Returns three lists: scaled_As, scaled_Bs, scaled_bs.
    """
    scaled_As, scaled_Bs, scaled_bs = [], [], []

    for A, B, b in zip(A_list, B_list, b_list):
        # reuse your single‐triplet scaler
        A_s, B_s, b_s = get_scaledABb(A, B, b, scaler)
        scaled_As.append(A_s)
        scaled_Bs.append(B_s)
        scaled_bs.append(b_s)

    return scaled_As, scaled_Bs, scaled_bs


def load_ABb_from_csv(csv_path="ABb_matrices.csv"):
    df = pd.read_csv(csv_path).sort_values("region_id").reset_index(drop=True)

    A_list = []
    B_list = []
    b_list = []

    for _, row in df.iterrows():
        A_list.append(torch.tensor([[row["A_T"], row["A_Cao"]]], dtype=torch.float32))
        B_list.append(torch.tensor([[row["B_Ca"], row["B_Cb"], row["B_Cc"]]], dtype=torch.float32))
        b_list.append(torch.tensor([[row["b"]]], dtype=torch.float32))

    return A_list, B_list, b_list

class Data_cstr(data.Dataset):
    def __init__(self, dataset):
        self.dataset_tensor = torch.from_numpy(dataset)
        self.X = self.dataset_tensor[:, :2]
        self.Y = self.dataset_tensor[:, 2:]  # assuming Y is the remaining columns
        self.train_set, self.val_set, self.test_set = self.split_data(0.2)  # initial val_ratio -> 0.2

        self.A_list, self.B_list, self.b_list = load_ABb_from_csv("ABb_matrices.csv")
        # self.constrained_indexes = list(set([index for index in torch.nonzero(self.B)[:, -1].tolist()]))
        # self.unconstrained_indexes = [item for item in range(self.B.shape[1]) if item not in self.constrained_indexes]

        self.constrained_indexes = []
        self.unconstrained_indexes = []

    def __len__(self):
        return len(self.dataset_tensor)

    def __getitem__(self, idx):
        return self.dataset_tensor[idx, :]

    def split_data(self, val_ratio, test_ratio=0.2):
        XY = data.TensorDataset(self.X, self.Y)
        XY = shuffle(XY, random_state=42)  # Shuffle the data (you can set random_state for reproducibility)
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
    def __init__(self, A, B, b, mu, reduction='mean'):
        super(PINNLoss, self).__init__()
        self.A = A
        self.B = B
        self.b = b
        self.mu = mu
        self.reduction = reduction

    def forward(self, X, pred, target):
        mse_loss = F.mse_loss(pred, target, reduction=self.reduction)
        pinn_loss = torch.mean(self.mu * (torch.mm(self.B, pred.T) + torch.mm(self.A, X.T) - self.b.unsqueeze(1))**2)
        return mse_loss, pinn_loss


class ALMLoss(nn.Module):
    def __init__(self, A, B, b, reduction='mean'):
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



def custom_sigmoid(x, transition_point, steepness):
    transition_width = 100.0 / steepness
    w = (x - transition_point) / transition_width
    return torch.sigmoid(w)

def window_1d(x, L, U, steepness):
    left  = custom_sigmoid(x, L, steepness)
    right = 1.0 - custom_sigmoid(x, U, steepness)
    return left * right

def get_masks_2d(X, T_edges, C_edges, steepT=8e5, steepC=8e5, normalize=True):
    # X is already scaled
    T = X[:, 0:1]
    C = X[:, 1:2]

    masks = []
    for i in range(len(T_edges) - 1):
        wT = window_1d(T, T_edges[i], T_edges[i+1], steepT)
        for j in range(len(C_edges) - 1):
            wC = window_1d(C, C_edges[j], C_edges[j+1], steepC)
            masks.append(wT * wC)

    masks = torch.cat(masks, dim=1)  # (batch, n_regions)
    if normalize:
        masks = masks / (masks.sum(dim=1, keepdim=True) + 1e-12)
    return masks

def get_violation(args, data, X, pred, steepT=8e5, steepC=8e5):
    A_list, B_list, b_list = data['A_list'], data['B_list'], data['b_list']

    # edges stored in data_dict (already scaled)
    T_edges = torch.tensor(data["T_edges"], dtype=X.dtype, device=X.device)
    C_edges = torch.tensor(data["C_edges"], dtype=X.dtype, device=X.device)

    masks = get_masks_2d(X, T_edges, C_edges, steepT=steepT, steepC=steepC, normalize=True)

    assert masks.shape[1] == len(A_list), f"Mask count {masks.shape[1]} != #constraints {len(A_list)}"

    violations = []
    for r, (Ai, Bi, bi) in enumerate(zip(A_list, B_list, b_list)):
        Ai = Ai.to(dtype=X.dtype, device=X.device)
        Bi = Bi.to(dtype=X.dtype, device=X.device)
        bi = bi.to(dtype=X.dtype, device=X.device)

        v = (X @ Ai.T + pred @ Bi.T - bi)  # (batch,1)
        v = v * masks[:, r:r+1]           # apply region weight
        violations.append(v)

    return torch.cat(violations, dim=1)  # (batch, n_regions)


# --- NEW: original nonlinear violation (per-sample, unscaled space) ---

def compute_violation_original_nonlinear(
    X_scaled: torch.Tensor,
    Ypred_scaled: torch.Tensor,
    scaler,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Returns |eq1| per sample as a 1D torch tensor (length = batch size),
    evaluated in ORIGINAL (unscaled) units.

    Assumes:
        X_scaled     = [T, Cao]          with shape (batch, 2)
        Ypred_scaled = [Ca, Cb, Cc]      with shape (batch, 3)

    The scaler is assumed to be fit on columns:
        [T, Cao, Ca, Cb, Cc]
    """

    with torch.no_grad():
        # Keep dtype consistent with input tensor
        torch_dtype = X_scaled.dtype

        # Move to CPU numpy for sklearn inverse_transform
        XYs = torch.cat([X_scaled, Ypred_scaled], dim=1).detach().cpu().numpy()   # (batch, 5)

        n_features = scaler.n_features_in_

        # Pad if needed
        if XYs.shape[1] < n_features:
            pad = np.zeros((XYs.shape[0], n_features - XYs.shape[1]), dtype=XYs.dtype)
            XYs_full = np.hstack([XYs, pad])
        else:
            XYs_full = XYs

        # Inverse transform to original units
        XY = scaler.inverse_transform(XYs_full)

        T   = XY[:, 0]   # original temperature
        Cao = XY[:, 1]   # original Cao
        Ca  = XY[:, 2]   # original Ca
        Cb  = XY[:, 3]   # original Cb
        Cc  = XY[:, 4]   # original Cc (not used in eq1, but reconstructed)

        # Constants
        Cbo = 2.0
        V   = 10.0
        Q   = 1.0
        tau = V / Q

        Afo = 1e13
        Eaf = 90000.0
        Aro = 1e11
        Ear = 80000.0
        R   = 8.314

        # Kinetics
        kf = Afo * np.exp(-Eaf / (R * T))
        kr = Aro * np.exp(-Ear / (R * T))

        # Nonlinear equality residual in original units
        eq1 = (Cao - Ca) - (kf * Ca * (Cb ** 2) * tau) + (kr * (Cao - Ca + Cbo - Cb) * tau)

        v = np.abs(eq1)

        # Return tensor on desired device and same dtype as model tensors
        return torch.tensor(v, dtype=torch_dtype, device=device)