import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.utils import shuffle 
from sklearn.preprocessing import MaxAbsScaler
import torch.utils.data as data
import numpy as np
import pandas as pd
import os
import pickle
import argparse
import csv
import time


device = "cpu"


class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, output_dim,  activation="softplus",):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(hidden_num - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        
        if activation == "relu":
            self.activation = nn.ReLU()

        elif activation == "softplus":
            self.activation = nn.Softplus()

        elif activation == "tanh":
            self.activation = nn.Tanh()

        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

        else:
            raise ValueError(
                f"Unsupported activation: {activation}"
            )

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)
    
    
def LoadData(args):

    dataset_arr, scaler = load_data(args.dataset_path)

    dataset = Data_cstr(dataset_arr)

    train_loader = data.DataLoader(
        dataset.train_set,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = data.DataLoader(
        dataset.val_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    test_loader = data.DataLoader(
        dataset.test_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "dataset": dataset,
        "scaler": scaler,
    }


def LoadModel(args, data):
    if args.model == "NN":
        model = NN(args.input_dim, args.hidden_dim, args.hidden_num, args.z0_dim, activation=args.activation)
    else:
        raise ValueError("Model not supported!")
    return model.to(device)



def get_optimizer(args, model):
    if args.optimizer == "adam":
        return optim.Adam(model.parameters(), lr=args.lr)
    if args.optimizer == "SGD":
        return optim.SGD(model.parameters(), lr=args.lr)
    raise ValueError("Invalid optimizer")


def get_loss_func(args, data):
    if args.loss_type == "MSE":
        return nn.MSELoss()


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

class Data_cstr(data.Dataset):
    def __init__(self, dataset):
        self.dataset_tensor = torch.from_numpy(dataset).float()
        self.X = self.dataset_tensor[:, :2]
        self.Y = self.dataset_tensor[:, 2:]
        self.train_set, self.val_set, self.test_set = self.split_data(0.2)

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


def run_training(args, data):
    model = LoadModel(args, data)
    optimizer = get_optimizer(args, model)
    loss_func = get_loss_func(args, data)

    # min_loss = np.inf
    train_losses, val_losses = [], []

    for epoch in range(args.epochs):
        train_loss = 0.0

        for X, Y in data["train_loader"]:
            X, Y = X.to(device), Y.to(device)
            mse = optimizer_step(model, optimizer, loss_func, X, Y, args, data)
            train_loss += mse

        train_loss /= len(data["train_loader"])

        val_loss = test(model, data, args)
        train_losses.append(train_loss)
       
        val_losses.append(val_loss)

        # if np.mean(val_loss) < min_loss:
        #     min_loss = np.mean(val_loss)
        #     checkpoint(model, val_loss, args, epoch)

        if (epoch + 1) % 50 == 0:
            print(f"epoch: {epoch+1:05d}",
                  f"loss_train: {train_loss:.5f}",
                  f"loss_val: {val_loss:.5f}")


    # Save the final-epoch model for consistent cross-method comparison.
    checkpoint(model,val_losses[-1],args,args.epochs - 1)
    
    save_history(args, train_losses, val_losses)
    return model


def optimizer_step(model, optimizer, loss_func, X, Y, args, data, lambda_k=None, mu_k=None):

    if isinstance(loss_func, nn.MSELoss):
        model.train(); optimizer.zero_grad()
        pred = model(X)
        loss = loss_func(pred, Y)
        loss.backward(); optimizer.step()
        return loss.item()


def test(model, data, args):
    loss_func = get_loss_func(args, data)
    model.eval()
    test_loss = 0.0


    with torch.no_grad():
        for X, Y in data["val_loader"]:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
        
            test_loss += loss_func(pred, Y).item()
           

    return test_loss / len(data["val_loader"])


def checkpoint(model, val_loss, args, epoch):
    # ckpt = {"state_dict": model.state_dict()}
    ckpt = {
        "state_dict": model.state_dict(),

        # model architecture
        "input_dim": args.input_dim,
        "hidden_dim": args.hidden_dim,
        "hidden_num": args.hidden_num,
        "output_dim": args.z0_dim,

        # activation
        "activation": args.activation,
    }
    out_dir = f"./models/{args.dataset_type}/{args.model}/{args.val_ratio}"
    os.makedirs(out_dir, exist_ok=True)
    # path = f"{out_dir}/{args.model_id}_{args.val_ratio}_{args.run}.pth"
    path = (
    f"{out_dir}/"
    f"{args.activation}_{args.model_id}_{args.val_ratio}_{args.run}.pth")
    torch.save(ckpt, path)


def evaluate_model(data, args):
    loss_func = nn.MSELoss()
    scores = {}
    try:
        model = LoadModel(args, data)
        load_weights(model, args.model_id, args)
        model.to(device).eval()

        rmse_total = 0.0
        prediction_time = 0.0

        with torch.no_grad():
            for X, Y in data["test_loader"]:
                X, Y = X.to(device), Y.to(device)
                
                # Pure prediction = NN forward
                prediction_start = time.perf_counter()
        
                pred = model(X)
                
                prediction_time += time.perf_counter() - prediction_start
                
             
                rmse_total += loss_func(pred, Y).item()
           
        rmse_total = np.sqrt(rmse_total / len(data["test_loader"]))
        

        scores.update({
            "rmse_total": float(rmse_total),
            "prediction_time_sec": float(prediction_time),
        })

        create_report(scores, args)
        return scores

    except FileNotFoundError:
        nan_scores = {"rmse_total": float("nan"), "prediction_time_sec": float("nan")}
        create_report(nan_scores, args)
        return nan_scores


def args_to_dict(args):
    return vars(args)


def save_history(args, train_losses, val_losses):
    out_dir = f"./data/learning_curves/{args.dataset_type}/{args.model}/{args.val_ratio}"
    os.makedirs(out_dir, exist_ok=True)
    # np.save(f"{out_dir}/{args.model_id}_train_losses_run{args.run}.npy", train_losses)
    # np.save(f"{out_dir}/{args.model_id}_val_losses_run{args.run}.npy", val_losses)
    np.save(
        f"{out_dir}/{args.activation}_{args.model_id}_train_losses_run{args.run}.npy",
        train_losses
    )

    np.save(
        f"{out_dir}/{args.activation}_{args.model_id}_val_losses_run{args.run}.npy",
        val_losses
    )

def load_weights(model, model_id, args):
    path = (
        f"./models/{args.dataset_type}/{args.model}/{args.val_ratio}/"
        f"{args.activation}_{model_id}_{args.val_ratio}_{args.run}.pth"
    )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model


def create_report(scores, args):
    d = args_to_dict(args) | scores
    out_dir = f"./data/tables/{args.dataset_type}/{args.model}/{args.val_ratio}"
    os.makedirs(out_dir, exist_ok=True)
    with open(
        f"{out_dir}/{args.activation}_{args.model_id}_{args.val_ratio}_{args.run}.csv",
        "w",
        newline=""
    ) as f:
        writer = csv.writer(f)
        for k, v in d.items():
            writer.writerow([k, v])


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="NN")
    parser.add_argument("--model_id", type=str, default="MODELID")
    parser.add_argument("--input_dim", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--hidden_num", type=int, default=2)
    parser.add_argument("--z0_dim", type=int, default=3)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dtype", default=32, type=int)
    parser.add_argument("--dataset_type", type=str, default="cstr")
    parser.add_argument("--dataset_path", type=str, default="data.csv")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--job", type=str, required=True, choices=["train", "experiment"])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--activation",type=str,default="softplus",choices=["relu","softplus","tanh","sigmoid",],
)
    return parser.parse_args()


def main(args):
    args.loss_type = "MSE"
    data = LoadData(args)

    if args.job == "train":
        run_training(args, data)
        return

    if args.job == "experiment":
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        eval_start = time.perf_counter()
        scores = evaluate_model(data, args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"Evaluation time: {time.perf_counter() - eval_start:.4f} s")
        print(scores)
        return

    raise ValueError("projection_check is handled by diagnostics.projection_check")


if __name__ == "__main__":
    main(add_arguments())


