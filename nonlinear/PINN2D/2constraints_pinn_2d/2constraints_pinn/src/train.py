import csv
import os

import numpy as np
import torch
import torch.nn as nn

import time

from utils import (
    ALMLoss,
    PINNLoss,
    LoadModel,
    compute_violation_original_nonlinear,
    get_loss_func,
    get_optimizer,
    get_violation,
    nonlinear_residuals,
)

device = "cpu"
torch.set_default_dtype(torch.float64) #could change to torch.float32 for float32


def run_training(args, data):
    model = LoadModel(args, data)
    optimizer = get_optimizer(args, model)
    loss_func = get_loss_func(args, data)

    min_loss = np.inf
    train_losses, val_losses = [], []
    train_violations, val_violations = [], []

    for epoch in range(args.epochs):
        train_loss = 0.0
        train_violation = 0.0

        for X, Y in data["train_loader"]:
            X, Y = X.to(device), Y.to(device)
            train_loss += optimizer_step(
                model, optimizer, loss_func, X, Y, args, data
            )
            pred_diff = conservation_step(model, X, data, args)
            train_violation += torch.abs(pred_diff.reshape(-1)).nanmean()

        train_loss /= len(data["train_loader"])
        train_violation /= len(data["train_loader"])

        val_loss, val_violation = test(model, data, args)
        train_losses.append(train_loss)
        train_violations.append(train_violation.detach().item())
        val_losses.append(val_loss)
        val_violations.append(val_violation.detach().item())

        if val_loss < min_loss:
            min_loss = val_loss
            checkpoint(model, args)

        if (epoch + 1) % 50 == 0:
            print(
                f"epoch: {epoch + 1:05d}",
                f"loss_train: {train_loss:.5f}",
                f"loss_val: {val_loss:.5f}",
                f"viol_train: {train_violation:.5f}",
                f"viol_val: {val_violation:.5f}",
            )

    save_history(
        args,
        train_losses,
        val_losses,
        train_violations,
        val_violations,
    )
    return model


def optimizer_step(
    model,
    optimizer,
    loss_func,
    X,
    Y,
    args,
    data,
    lambda_k=None,
    mu_k=None,
):
    if isinstance(loss_func, PINNLoss):
        model.train()
        optimizer.zero_grad()
        pred = model(X)
        mse_loss, pinn_loss = loss_func(X, pred, Y)
        total_loss = mse_loss + pinn_loss
        total_loss.backward()
        optimizer.step()
        return total_loss.item()

    if isinstance(loss_func, nn.MSELoss):
        model.train()
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_func(pred, Y)
        loss.backward()
        optimizer.step()
        return loss.item()

    if isinstance(loss_func, ALMLoss):
        mse_loss = None
        for _ in range(args.max_subiter + 1):
            model.train()
            optimizer.zero_grad()
            pred = model(X)
            mse_loss, penalty = loss_func(X, pred, Y, lambda_k, mu_k)
            (mse_loss + penalty).backward()
            optimizer.step()
        return mse_loss.item()


def conservation_step(model, X, data, args):
    model.eval()
    with torch.no_grad():
        pred = model(X)
        return nonlinear_residuals(X, pred, data["scaler"])


def test(model, data, args):
    loss_func = get_loss_func(args, data)
    model.eval()
    val_loss_sum = 0.0
    val_sample_count = 0
    val_violation_sum = 0.0
    val_residual_count = 0

    with torch.no_grad():
        for X, Y in data["val_loader"]:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            pred_diff = nonlinear_residuals(X, pred, data["scaler"])

            if args.loss_type == "PINN":
                mse, physics = loss_func(X, pred, Y)
                batch_loss = mse + physics
            else:
                batch_loss = loss_func(pred, Y)

            val_loss_sum += batch_loss.item() * X.shape[0]
            val_sample_count += X.shape[0]
            val_violation_sum += torch.abs(pred_diff).sum().item()
            val_residual_count += pred_diff.numel()

    val_loss = val_loss_sum / val_sample_count
    val_violation = torch.tensor(
        val_violation_sum / val_residual_count,
        dtype=torch.get_default_dtype(),
        device=device,
    )
    return val_loss, val_violation


def checkpoint(model, args):
    output_dir = f"./models/{args.dataset_type}/{args.model}/{args.val_ratio}"
    os.makedirs(output_dir, exist_ok=True)
    path = f"{output_dir}/{args.model_id}_{args.val_ratio}_{args.run}.pth"
    torch.save({"state_dict": model.state_dict()}, path)


def evaluate_model(data, args, split="test"):
    try:
        model = LoadModel(args, data)
        load_weights(model, args.model_id, args)
        model.to(device).eval()

        loader = data[f"{split}_loader"]
        squared_error_sum = 0.0
        output_count = 0
        pl_violation_sum = 0.0
        pl_residual_count = 0
        nonlinear_sum = torch.zeros(2, dtype=torch.float64)
        sample_count = 0
        prediction_time = 0.0

        with torch.no_grad():
            for X, Y in loader:
                X, Y = X.to(device), Y.to(device)
                
                
                # Pure prediction timing:
                # NN forward + explicit PL-KKT projection only.
                prediction_start = time.perf_counter()
                pred = model(X)
                
                prediction_time += time.perf_counter() - prediction_start
                
                pred_diff = get_violation(args, data, X, pred)
                squared_error_sum += torch.sum((pred - Y) ** 2).item()
                output_count += Y.numel()
                pl_violation_sum += torch.abs(pred_diff).sum().item()
                pl_residual_count += pred_diff.numel()

                nonlinear_violation = compute_violation_original_nonlinear(
                    X, pred, data["scaler"], device=device
                )
                nonlinear_sum += nonlinear_violation.sum(dim=0).cpu().double()
                sample_count += X.shape[0]

        scores = {
            "eval_split": split,
            "rmse_total": float(np.sqrt(squared_error_sum / output_count)),
            "violation": float(pl_violation_sum / pl_residual_count),
            "violation_original_nonlinear": float(
                nonlinear_sum.sum().item() / (2 * sample_count)
            ),
            "violation_rxn": float(nonlinear_sum[0].item() / sample_count),
            "violation_mb": float(nonlinear_sum[1].item() / sample_count),
            "prediction_time_sec": float(prediction_time),
        }
        print(scores)
        create_report(scores, args)
        return scores

    except FileNotFoundError:
        scores = {
            "eval_split": split,
            "rmse_total": float("nan"),
            "violation": float("nan"),
            "violation_original_nonlinear": float("nan"),
            "violation_rxn": float("nan"),
            "violation_mb": float("nan"),
        }
        create_report(scores, args)
        return scores


def save_history(
    args,
    train_losses,
    val_losses,
    train_violations,
    val_violations,
):
    output_dir = (
        f"./data/learning_curves/{args.dataset_type}/{args.model}/"
        f"{args.val_ratio}"
    )
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{output_dir}/{args.model_id}"
    np.save(f"{prefix}_train_losses_run{args.run}.npy", train_losses)
    np.save(f"{prefix}_val_losses_run{args.run}.npy", val_losses)
    np.save(
        f"{prefix}_train_violations_run{args.run}.npy", train_violations
    )
    np.save(f"{prefix}_val_violations_run{args.run}.npy", val_violations)


def load_weights(model, model_id, args):
    path = (
        f"./models/{args.dataset_type}/{args.model}/{args.val_ratio}/"
        f"{model_id}_{args.val_ratio}_{args.run}.pth"
    )
    checkpoint_data = torch.load(
        path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint_data["state_dict"])
    model.to(device).eval()
    return model


def create_report(scores, args):
    report = vars(args) | scores
    output_dir = f"./data/tables/{args.dataset_type}/{args.model}/{args.val_ratio}"
    os.makedirs(output_dir, exist_ok=True)
    path = f"{output_dir}/{args.model_id}_{args.val_ratio}_{args.run}.csv"
    with open(path, "w", newline="") as report_file:
        writer = csv.writer(report_file)
        writer.writerows(report.items())
