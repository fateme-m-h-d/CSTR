import csv
import os
import numpy as np
import torch
import torch.nn as nn
from utils import LoadModel, get_optimizer, get_loss_func, get_violation, PINNLoss, ALMLoss

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
torch.set_default_dtype(torch.float64)


def run_training(args, data):
    model = LoadModel(args, data)
    optimizer = get_optimizer(args, model)
    loss_func = get_loss_func(args, data)

    print("Start Training…")
    min_loss = np.inf
    train_losses, val_losses = [], []
    train_violations, val_violations = [], []

    for epoch in range(args.epochs):
        train_loss = 0.0
        train_violation = 0.0

        for X, Y in data["train_loader"]:
            X, Y = X.to(device), Y.to(device)
            mse = optimizer_step(model, optimizer, loss_func, X, Y, args, data)
            pred_diff = conservation_step(model, X, data, args)
            train_loss += mse
            train_violation += torch.abs(pred_diff).mean()

        train_loss /= len(data["train_loader"])
        train_violation /= len(data["train_loader"])

        val_loss, val_violation = test(model, data, args)
        train_losses.append(train_loss)
        train_violations.append(train_violation.detach().item())
        val_losses.append(val_loss)
        val_violations.append(val_violation.detach().item())

        if np.mean(val_loss) < min_loss:
            min_loss = np.mean(val_loss)
            checkpoint(model, val_loss, args, epoch)

        if (epoch + 1) % 50 == 0:
            print(f"epoch: {epoch+1:05d}",
                  f"loss_train: {train_loss:.5f}",
                  f"loss_val: {val_loss:.5f}",
                  f"viol_train: {train_violation:.5e}",
                  f"viol_val: {val_violation:.5e}")

    print("Training Finished!")
    save_history(args, train_losses, val_losses, train_violations, val_violations)


def optimizer_step(model, optimizer, loss_func, X, Y, args, data, lambda_k=None, mu_k=None):
    if isinstance(loss_func, PINNLoss):
        model.train(); optimizer.zero_grad()
        pred = model(X)
        mse_loss, pinn_loss = loss_func(X, pred, Y)
        
        loss = mse_loss + pinn_loss
        loss.backward()
        optimizer.step()

        return loss.item()

    if isinstance(loss_func, nn.MSELoss):
        model.train(); optimizer.zero_grad()
        pred = model(X)
        loss = loss_func(pred, Y)
        loss.backward(); optimizer.step()
        return loss.item()

    if isinstance(loss_func, ALMLoss):
        mse_loss = None
        for _ in range(args.max_subiter + 1):
            model.train(); optimizer.zero_grad()
            pred = model(X)
            mse_loss, penalty = loss_func(X, pred, Y, lambda_k, mu_k)
            (mse_loss + penalty).backward(); optimizer.step()
        return mse_loss.item()


def conservation_step(model, X, data, args):
    model.eval()
    with torch.no_grad():
        pred = model(X)
        return get_violation(args, data, X, pred)


def test(model, data, args):
    loss_func = get_loss_func(args, data)
    model.eval()
    test_loss = 0.0
    test_violation = 0.0

    with torch.no_grad():
        for batch_idx, (X, Y) in enumerate(data['val_loader']):
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            pred_diff = get_violation(args, data, X, pred)
            if args.loss_type == 'PINN':
                mse_loss, pinn_loss = loss_func(X, pred, Y)
                loss = mse_loss + pinn_loss
                test_loss += loss.item()
            elif args.loss_type == 'MSE':
                loss = loss_func(pred, Y)
                test_loss += loss.item()
            test_violation += torch.abs(pred_diff.view(-1)).mean()
    test_loss /= len(data['val_loader'])  # Test set Average loss
    test_violation /= len(data['val_loader'])  # Test set Average violation
    return test_loss, test_violation



def checkpoint(model, val_loss, args, epoch):
    ckpt = {"state_dict": model.state_dict()}
    out_dir = f"./models/{args.dataset_type}/{args.model}/{args.val_ratio}"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/{args.model_id}_{args.val_ratio}_{args.run}.pth"
    torch.save(ckpt, path)


def evaluate_model(data, args):
    loss_func = nn.MSELoss()
    scores = {}
    try:
        model = LoadModel(args, data)
        print(f"Loading: {args.model_id}_{args.val_ratio}_{args.run}.pth")
        load_weights(model, args.model_id, args)
        model.to(device).eval()

        rmse_total = 0.0
        violation = 0.0
        v_nl_batches = []

        with torch.no_grad():
            for X, Y in data["test_loader"]:
                X, Y = X.to(device), Y.to(device)
                pred = model(X)
                pred_diff = get_violation(args, data, X, pred)
                rmse_total += loss_func(pred, Y).item()
                violation += torch.abs(pred_diff).mean()
                

        rmse_total = np.sqrt(rmse_total / len(data["test_loader"]))
        violation = (violation / len(data["test_loader"])).item()
        

        scores.update({
            "rmse_total": float(rmse_total),
            "violation": float(violation)
        })

        print(scores)
        create_report(scores, args)
        return scores

    except FileNotFoundError:
        print(f"Model not found: {args.model_id}_{args.val_ratio}_{args.run}.pth")
        nan_scores = {"rmse_total": float("nan"), "violation": float("nan")}
        create_report(nan_scores, args)
        return nan_scores


def args_to_dict(args):
    return vars(args)


def save_history(args, train_losses, val_losses, train_violations, val_violations):
    out_dir = f"./data/learning_curves/{args.dataset_type}/{args.model}/{args.val_ratio}"
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/{args.model_id}_train_losses_run{args.run}.npy", train_losses)
    np.save(f"{out_dir}/{args.model_id}_val_losses_run{args.run}.npy", val_losses)
    np.save(f"{out_dir}/{args.model_id}_train_violations_run{args.run}.npy", train_violations)
    np.save(f"{out_dir}/{args.model_id}_val_violations_run{args.run}.npy", val_violations)


def load_weights(model, model_id, args):
    path = f"./models/{args.dataset_type}/{args.model}/{args.val_ratio}/{model_id}_{args.val_ratio}_{args.run}.pth"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model


def create_report(scores, args):
    d = args_to_dict(args) | scores
    out_dir = f"./data/tables/{args.dataset_type}/{args.model}/{args.val_ratio}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/{args.model_id}_{args.val_ratio}_{args.run}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for k, v in d.items():
            writer.writerow([k, v])
