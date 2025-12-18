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
from models import (
    Cao, Cbo, Cco,
    V, Q, Afo, Eaf, Aro, Ear, R
)


# device = "cuda" if torch.cuda.is_available() else "cpu"    #GPU OR CPU
device = "cpu"

def LoadData(args):
    if args.dataset_type == 'cstr':
        dataset_arr, scaler = load_data(args.dataset_path)
        Data_class = Data_cstr
    else:
        raise ValueError('Dataset not supported!')

    if args.dtype == 32:
        dataset_arr = dataset_arr.astype(np.float32)
    dataset = Data_class(dataset_arr)
    dataset.resplit_data(args.val_ratio)

    A, B, b = get_scaledABb(dataset.A, dataset.B, dataset.b, scaler)

    if args.dtype == 32:
        A, B, b = A.float(), B.float(), b.float()
    else:
        A, B, b = A.double(), B.double(), b.double()
    print(f'type of A: {A.dtype}, type of B: {B.dtype}, type of b: {b.dtype}')


    params = {'batch_size': args.batch_size,
              'shuffle': True}
    train_loader = data.DataLoader(dataset.train_set, **params)
    val_loader = data.DataLoader(dataset.val_set, **params)
    test_loader = data.DataLoader(dataset.test_set, **params)

    print(f'train set size: {len(dataset.train_set)}, val set size: {len(dataset.val_set)}, test set size: {len(dataset.test_set)}')

    data_dict = {'train_loader': train_loader, 'val_loader': val_loader, 'test_loader': test_loader,
                 'dataset': dataset, 'A': A, 'B': B, 'b': b.unsqueeze(1), 'scaler': scaler
                 }
    return data_dict


def LoadModel(args, data):
    if args.model == 'NN':
        model = NN(args.input_dim, args.hidden_dim, args.hidden_num, args.z0_inner_dim)
    elif args.model == 'KKThPINN':
        model = NNOPT(args.input_dim, args.hidden_dim, args.hidden_num, args.z0_inner_dim, args.z0_dim,
                      data['A'], data['B'], data['b'], data['scaler'])
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
    return loss_func

import pickle
import os
def load_data(dataset_path):
    dataset = np.array(pd.read_csv(dataset_path).values)
    scaler = MaxAbsScaler()
    scaler.fit(dataset)
    # Manually set a higher max value for temperature input
    scaler.scale_[0] = max(scaler.scale_[0], 800)  # Assuming temperature is the first feature
    
    print("Scaler Factors:", scaler.scale_)

    dataset = scaler.transform(dataset)
    
    # Save the scaler to use it later during predictions
    base_dir = os.getcwd()    #new line of the code. need to change it.
    scaler_path = os.path.join(base_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved at {scaler_path}")

    # Save the transformed dataset to a CSV file
    scaled_dataset_path = os.path.join(os.getcwd(), 'scaled_data.csv')
    dataset_scaled_df = pd.DataFrame(dataset)
    dataset_scaled_df.to_csv(scaled_dataset_path, index=False)
    print(f"Scaled dataset saved at {scaled_dataset_path}")
    return dataset, scaler


# # load_data("./data.csv")
# dataset, scaler = load_data("./data.csv")
# # Print scaler factors outside the function
# print("Scaler Factors outside function:", scaler.scale_)


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


class Data_cstr(data.Dataset):
    def __init__(self, dataset):
        self.dataset_tensor = torch.from_numpy(dataset)
        self.X = self.dataset_tensor[:, :1]
        self.Y = self.dataset_tensor[:, 1:]
        self.train_set, self.val_set, self.test_set = self.split_data(0.2)  # initial val_ratio -> 0.2
        
        tau = V / Q
        self.A = torch.tensor([[0]
                                ])  #changed
        self.B = torch.tensor([[ 0, 0, 0, -tau, +1]
                                ])  #changed
        self.b = torch.tensor([Cao])    #changed


        self.constrained_indexes = list(set([index for index in torch.nonzero(self.B)[:, -1].tolist()]))
        self.unconstrained_indexes = [item for item in range(self.B.shape[1]) if item not in self.constrained_indexes]

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


# def get_violation(args, data, X, pred):
    
#     violation = torch.mm(data['A'], X.T) + torch.mm(data['B'], pred.T) - data['b'].repeat(1, X.T.shape[1])
    
#     return violation
tau = V / Q

def get_violation(args, data, X, pred):
    """
    Nonlinear steady-state CSTR constraint violation (diagnostic only)

    Parameters
    ----------
    X    : torch.Tensor, shape [N, 1]
           Scaled temperature input
    pred : torch.Tensor, shape [N, 3]
           Scaled model outputs [Ca, Cb, Cc]

    Returns
    -------
    violation : torch.Tensor, shape [N, 1]
                Nonlinear physics residual
    """

    scaler = data['scaler']

    # -------------------------------
    # Unscale input
    # -------------------------------
    T = X[:, 0:1] * scaler.scale_[0]   # [K]

    # -------------------------------
    # Unscale outputs
    # -------------------------------
    Ca = pred[:, 0:1] * scaler.scale_[1]
    Cb = pred[:, 1:2] * scaler.scale_[2]
    Cc = pred[:, 2:3] * scaler.scale_[3]

    # -------------------------------
    # Reaction kinetics
    # -------------------------------
    kf = Afo * torch.exp(-Eaf / (R * T))
    kr = Aro * torch.exp(-Ear / (R * T))

    g = kf * Ca * (Cb ** 2)
    f = kr * (Cao - Ca + Cbo - Cb + Cco)

    # -------------------------------
    # Steady-state mass balance residual
    # (same physics as data generation)
    # -------------------------------
    violation = (Ca - Cao) - f*tau + g*tau  
    

    return violation





    
