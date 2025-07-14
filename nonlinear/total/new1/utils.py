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
    
    A_list, B_list, b_list = get_scaledABb_list(dataset.A_list, dataset.B_list,
                                       dataset.b_list, scaler)
    
    
        # --- cast each fixed‐layer tensor to the same dtype as A,B,b above ---
    if args.dtype == 32:
        A_list = [A_i.float() for A_i in A_list]
        B_list = [B_i.float() for B_i in B_list]
        b_list = [b_i.float() for b_i in b_list]
    else:
        A_list = [A_i.double() for A_i in A_list]
        B_list = [B_i.double() for B_i in B_list]
        b_list = [b_i.double() for b_i in b_list]


    data_dict = {'train_loader': train_loader, 'val_loader': val_loader, 'test_loader': test_loader,
                 'dataset': dataset, 'A_list': A_list, 'B_list': B_list, 'b_list': b_list,
             # still keep stacked versions for violation metrics
             'A': torch.cat(A_list), 'B': torch.cat(B_list),
             'b': torch.cat(b_list).unsqueeze(1)
                }
    return data_dict


def LoadModel(args, data):
    if args.model == 'NN':
        model = NN(args.input_dim, args.hidden_dim, args.hidden_num, args.z0_dim)
    elif args.model == 'KKThPINN':
        model = NNOPT(args.input_dim, args.hidden_dim, args.hidden_num, args.z0_dim,
                      data['A_list'], data['B_list'], data['b_list'])
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
    scaler.scale_[0] = max(scaler.scale_[0], 800)  # Assuming temperature is the first feature
    
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



class Data_cstr(data.Dataset):
    def __init__(self, dataset):
        self.dataset_tensor = torch.from_numpy(dataset)
        self.X = self.dataset_tensor[:, :1]
        self.Y = self.dataset_tensor[:, 1:]
        self.train_set, self.val_set, self.test_set = self.split_data(0.2)  # initial val_ratio -> 0.2
        
        
        # self.A = torch.tensor([[-19.91427610060009]
        #                         ])  #changed
        # self.B = torch.tensor([[ -12059.334807989864, -12058.334807988307, 0]
        #                         ])  #changed
        # self.b = torch.tensor([-23450.209793805465])    #changed
        
        
        # 3 lines
        self.A_list = [
            torch.tensor([[- 0.00301071551554214]]),torch.tensor([[- 0.0287912896000818]]) ,torch.tensor([[- 0.0712637450806569]]), torch.tensor([[- 25.5032870206085]])
            # … add more rows if want more lines
        ]
        self.B_list = [
            torch.tensor([[ -1.02825709223637, - 0.0282570922363733, 0]]),
            torch.tensor([[ -1.54923960097239, - 0.549239600972388, 0]]),
            torch.tensor([[ -10.2140189737442, - 9.21401897374424, 0]]),
            torch.tensor([[-15979.9623522557, - 15978.9623522524, 0]]) 
            
        ]
        self.b_list = [
            torch.tensor([-1.93327845562254]),torch.tensor([-11.1707510081181]),torch.tensor([-39.8296448877009]), torch.tensor([-30645.4682099649])
        ]
        
        # first half linearization
        # self.A_list = [
        #     torch.tensor([[-0.02879478968746971]])
        #     # … add more rows if want more lines
        # ]
        # self.B_list = [
        #     torch.tensor([[ -1.549379806828716, -0.5493798071408083, 0]])
            
        # ]
        # self.b_list = [
        #     torch.tensor([-11.172181579708148])
        # ]
        
        # second half linearization
        # self.A_list = [
        #     torch.tensor([[-19.91427610060009]])
        #     # … add more rows if want more lines
        # ]
        # self.B_list = [
            
        #     torch.tensor([[-12059.334807989864, -12058.334807988307, 0]]) 
            
        # ]
        # self.b_list = [
        #     torch.tensor([-23450.209793805465])
        # ]
        
        # keep convenience matrices for PINN / ALM
        self.A = torch.cat(self.A_list, dim=0)   # (L × x_dim)
        self.B = torch.cat(self.B_list, dim=0)   # (L × z_dim)
        self.b = torch.cat(self.b_list, dim=0)   # (L)

        

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


def get_violation(args, data, X, pred):
    if isinstance(data['A_list'], list):
        A = torch.cat(data['A_list']);  B = torch.cat(data['B_list'])
        b = torch.cat(data['b_list']).unsqueeze(1)
    else:
        A, B, b = data['A'], data['B'], data['b']
    violation = A @ X.T + B @ pred.T - b.repeat(1, X.shape[0])
    # violation = torch.mm(data['A'], X.T) + torch.mm(data['B'], pred.T) - data['b'].repeat(1, X.T.shape[1])
    return violation
    
