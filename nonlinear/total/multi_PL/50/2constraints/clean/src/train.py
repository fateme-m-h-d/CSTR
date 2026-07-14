import csv
import os

import numpy as np
import torch
import torch.nn as nn

from .utils import (
    ALMLoss,
    PINNLoss,
    LoadModel,
    compute_violation_original_nonlinear,
    get_loss_func,
    get_optimizer,
    get_violation,
)

device = "cpu"
torch.set_default_dtype(torch.float32)


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

        checkpoint(model, val_loss, min_loss, args, epoch)

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
        (mse_loss + pinn_loss).backward()
        optimizer.step()
        return mse_loss.item()

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
        return get_violation(args, data, X, pred)


def test(model, data, args):
    loss_func = get_loss_func(args, data)
    model.eval()
    val_loss = 0.0
    val_violation = 0.0

    with torch.no_grad():
        for X, Y in data["val_loader"]:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            pred_diff = get_violation(args, data, X, pred)

            if args.loss_type == "PINN":
                mse, _ = loss_func(X, pred, Y)
                val_loss += mse.item()
            else:
                val_loss += loss_func(pred, Y).item()

            val_violation += torch.abs(pred_diff.reshape(-1)).nanmean()

    val_loss /= len(data["val_loader"])
    val_violation /= len(data["val_loader"])
    return val_loss, val_violation


def checkpoint(model, val_loss, min_loss, args, epoch):
    if np.mean(val_loss) < min_loss:
        output_dir = f"./models/{args.dataset_type}/{args.model}/{args.val_ratio}"
        os.makedirs(output_dir, exist_ok=True)
        path = f"{output_dir}/{args.model_id}_{args.val_ratio}_{args.run}.pth"
        torch.save({"state_dict": model.state_dict()}, path)


def evaluate_model(data, args):
    loss_func = nn.MSELoss()
    try:
        model = LoadModel(args, data)
        load_weights(model, args.model_id, args)
        model.to(device).eval()

        rmse_total = 0.0
        violation = 0.0
        nonlinear_violation_batches = []

        with torch.no_grad():
            for X, Y in data["test_loader"]:
                X, Y = X.to(device), Y.to(device)
                pred = model(X)
                pred_diff = get_violation(args, data, X, pred)
                rmse_total += loss_func(pred, Y).item()
                violation += torch.abs(pred_diff.reshape(-1)).nanmean()

                nonlinear_violation = compute_violation_original_nonlinear(
                    X, pred, data["scaler"], device=device
                )
                nonlinear_violation_batches.append(
                    nonlinear_violation.mean().item()
                )

        scores = {
            "rmse_total": float(
                np.sqrt(rmse_total / len(data["test_loader"]))
            ),
            "violation": float(
                (violation / len(data["test_loader"])).item()
            ),
            "violation_original_nonlinear": float(
                np.mean(nonlinear_violation_batches)
            ),
        }
        print(scores)
        create_report(scores, args)
        return scores

    except FileNotFoundError:
        scores = {
            "rmse_total": float("nan"),
            "violation": float("nan"),
            "violation_original_nonlinear": float("nan"),
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
