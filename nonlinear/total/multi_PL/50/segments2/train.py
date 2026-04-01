import csv
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from utils import (
    LoadModel, get_optimizer, get_loss_func,
    get_violation, PINNLoss, ALMLoss
)
import copy
import os
import pickle
import pandas as pd

from utils import compute_violation_original_nonlinear

# run‐time device choice
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
torch.set_default_dtype(torch.float32)

# Load the scaler used for normalization
scaler_path = "./scaler.pkl"  # adjust if needed
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
    print("Scaler loaded successfully.")
    print("Scaler Type:", type(scaler))
    if hasattr(scaler, 'scale_'):
        print("Scaler Factors (scale_):", scaler.scale_)


def run_training(args, data):
    model      = LoadModel(args, data)
    optimizer  = get_optimizer(args, model)
    loss_func  = get_loss_func(args, data)

    print('Start Training…')
    min_loss = np.inf
    train_losses, val_losses = [], []
    train_violations, val_violations = [], []

    for epoch in range(args.epochs):
        train_loss = 0.0
        train_violation = 0.0

        for batch_idx, (X, Y) in enumerate(data['train_loader']):
            X, Y = X.to(device), Y.to(device)
            # standard optimizer + PINN/ALM step
            mse = optimizer_step(model, optimizer, loss_func, X, Y, args, data)
            pred_diff = conservation_step(model, X, data, args)
            train_loss      += mse
            train_violation += torch.abs(pred_diff.reshape(-1)).nanmean()   # ← CHANGED

        train_loss      /= len(data['train_loader'])
        train_violation /= len(data['train_loader'])

        val_loss, val_violation = test(model, data, args)
        train_losses.append(train_loss)
        train_violations.append(train_violation.detach().item())
        val_losses.append(val_loss)
        val_violations.append(val_violation.detach().item())

        checkpoint(model, val_loss, min_loss, args, epoch)

        if (epoch + 1) % 50 == 0:
            print(f"epoch: {epoch+1:05d}",
                  f"loss_train: {train_loss:.5f}",
                  f"loss_val: {val_loss:.5f}",
                  f"viol_train: {train_violation:.5f}",
                  f"viol_val: {val_violation:.5f}")

    print("Training Finished!")
    save_history(args, train_losses, val_losses,
                 train_violations, val_violations)
    
    # after training loop finishes
    # with torch.no_grad():
    #     # Raw temps you want to inspect
    #     T_raw = np.linspace(280, 600, 500).reshape(-1,1)

    #     # SCALE THEM with your existing scaler (use the exact object & method you trained with)
    #     T_scaled = torch.from_numpy(T_raw / 800).float().to(device)

    #     # transition_points & steepness: use the same ones you passed to forward()
    #     M = model.get_masks(T_scaled, transition_points=[0.375, 0.425, 0.45, 0.5, 0.575, 0.625, 0.6625, 0.6875, 0.70625, 0.7225, 0.7375], steepness=800000)   # (N, n_regions)
    #     # for t, row in zip(T_raw.ravel(), M.cpu().numpy()):
    #     #     print(f"T={t:.1f} -> " + ", ".join([f"{m:.4f}" for m in row]))
    #     df = pd.DataFrame(
    #         M.cpu().numpy(),
    #         columns=[f"mask_{i}" for i in range(M.shape[1])]
    #     )
    #     df.insert(0, "T", T_raw.ravel())
    #     df.to_csv("mask_usage.csv", index=False)

# Deleted (new) this one as it was redundant with the same function in main.py

    # if args.job == 'experiment':
    #     scores = evaluate_model(data, args)
    #     return scores


def optimizer_step(model, optimizer, loss_func, X, Y, args, data,
                   lambda_k=None, mu_k=None):
    """One gradient step (MSE, PINN, or ALM)."""
    if isinstance(loss_func, PINNLoss):
        model.train()
        optimizer.zero_grad()
        pred = model(X)
        mse_loss, pinn_loss = loss_func(X, pred, Y)
        (mse_loss + pinn_loss).backward()
        optimizer.step()
        return mse_loss.item()

    elif isinstance(loss_func, nn.MSELoss):
        model.train()
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_func(pred, Y)
        loss.backward()
        optimizer.step()
        return loss.item()

    elif isinstance(loss_func, ALMLoss):
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
    """Compute instantaneous constraint‐violation for PINN/ALM."""
    model.eval()
    with torch.no_grad():
        pred = model(X)
        pred_diff = get_violation(args, data, X, pred, steepT=8e5, steepC=8e5)
    return pred_diff


def test(model, data, args):
    """Compute val‐loss and val‐violation."""
    loss_func = get_loss_func(args, data)
    model.eval()

    test_loss = 0.0
    test_violation = 0.0

    with torch.no_grad():
        for X, Y in data['val_loader']:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            pred_diff = get_violation(args, data, X, pred, steepT=8e5, steepC=8e5)

            if args.loss_type == 'PINN':
                mse, pinn = loss_func(X, pred, Y)
                test_loss += mse.item()
            else:  # MSE
                test_loss += loss_func(pred, Y).item()

            test_violation += torch.abs(pred_diff.reshape(-1)).nanmean()

    test_loss      /= len(data['val_loader'])
    test_violation /= len(data['val_loader'])
    return test_loss, test_violation


def checkpoint(model, val_loss, min_loss, args, epoch):
    """Save best model by validation loss."""
    if np.mean(val_loss) < min_loss:
        ckpt = {'state_dict': model.state_dict()}
        out_dir = f'./models/{args.dataset_type}/{args.model}/{args.val_ratio}'
        os.makedirs(out_dir, exist_ok=True)
        path = f'{out_dir}/{args.model_id}_{args.val_ratio}_{args.run}.pth'
        torch.save(ckpt, path)


def evaluate_model(data, args):
    """Load a trained model and compute test RMSE, violation, and post‐NN RMSE."""
    loss_func = nn.MSELoss()
    scores = {}
    try:
        model = LoadModel(args, data)
        print(f"Loading: {args.model_id}_{args.val_ratio}_{args.run}.pth")
        load_weights(model, args.model_id, args)
        model.to(device).eval()

        # 1) baseline NN RMSE + violation
        rmse_total = 0.0
        violation  = 0.0
        v_nl_batches = []  # NEW accumulator for original nonlinear violation
        with torch.no_grad():
            for X, Y in data['test_loader']:
                X, Y = X.to(device), Y.to(device)
                pred = model(X)
                pred_diff = get_violation(args, data, X, pred, steepT=8e5, steepC=8e5)
                rmse_total += loss_func(pred, Y).item()
                violation  += torch.abs(pred_diff.reshape(-1)).nanmean()
                
                # --- NEW: compute per-sample |eq1| in original space, then batch-mean
                v_nl = compute_violation_original_nonlinear(X, pred, scaler, device=device)  # shape (batch,)
                v_nl_batches.append(v_nl.mean().item())

        rmse_total = np.sqrt(rmse_total / len(data['test_loader']))
        violation  = (violation / len(data['test_loader'])).item()
        
        # --- aggregate to a single float like your other metrics ---
        violation_original_nonlinear = float(np.mean(v_nl_batches))
        
        scores.update({'rmse_total': float(rmse_total),
                       'violation': float(violation),
                       'violation_original_nonlinear': violation_original_nonlinear})

        # 2) piecewise‐linear post‐processing for pure‐NN case
        # if args.model == 'NN':
        #     post_rmse = 0.0

        #     # grab your lists from the data dict:
        #     A_list = data['A_list']
        #     B_list = data['B_list']
        #     b_list = data['b_list']
        #     z0 = args.z0_dim

        #     with torch.no_grad():
        #         for X, Y in data['test_loader']:
        #             X, Y = X.to(device), Y.to(device)
        #             z_nn = model(X)  # network latent output

        #             region_outs = []
        #             for Ai, Bi, bi in zip(A_list, B_list, b_list):
        #                 # projector chunk = Bᵀ (B Bᵀ)⁻¹
        #                 chunk = Bi.t() @ torch.inverse(Bi @ Bi.t())
        #                 Astar = -chunk @ Ai
        #                 Bstar = torch.eye(z0, device=device) - chunk @ Bi
        #                 bstar = (chunk @ bi.view(-1,1)).squeeze(-1)

        #                 # local linear prediction
        #                 lin = X @ Astar.T + z_nn @ Bstar.T \
        #                       + torch.ones((X.size(0),1), device=device) \
        #                         @ bstar.unsqueeze(0)

        #                 # apply same sigmoid masks:
        #                 def σ(x, t, s=800000):
        #                     w = (x - t) / (100/s)
        #                     return torch.sigmoid(w)

        #                 tp = [0.375, 0.425, 0.45, 0.5, 0.575, 0.625, 0.6625, 0.6875, 0.70625, 0.7225, 0.7375]
        #                 if Ai is A_list[0]:
        #                     mask = 1.0 - σ(X, tp[0])
        #                 elif Ai is A_list[-1]:
        #                     mask = σ(X, tp[-1])
        #                 else:
        #                     i = A_list.index(Ai)
        #                     mask = σ(X, tp[i-1]) * (1.0 - σ(X, tp[i]))

        #                 region_outs.append(lin * mask)

        #             post_pred = torch.stack(region_outs, dim=0).sum(dim=0)
        #             post_rmse += loss_func(post_pred, Y).item()

            # post_rmse = np.sqrt(post_rmse / len(data['test_loader']))
            # scores['post_rmse_total'] = float(post_rmse)

        print(scores)
        create_report(scores, args)
        return scores

    except FileNotFoundError:
        print(f"Model not found: {args.model_id}_{args.val_ratio}_{args.run}.pth")
        nan_scores = {'rmse_total': float('nan'),
                      'violation': float('nan')}
        if args.model == 'NN':
            nan_scores['post_rmse_total'] = float('nan')
        create_report(nan_scores, args)
        return nan_scores


def args_to_dict(args):
    return vars(args)


def save_history(args, train_losses, val_losses, train_violations, val_violations):
    out_dir = f'./data/learning_curves/{args.dataset_type}/{args.model}/{args.val_ratio}'
    os.makedirs(out_dir, exist_ok=True)
    np.save(f'{out_dir}/{args.model_id}_train_losses_run{args.run}.npy', train_losses)
    np.save(f'{out_dir}/{args.model_id}_val_losses_run{args.run}.npy', val_losses)
    np.save(f'{out_dir}/{args.model_id}_train_violations_run{args.run}.npy', train_violations)
    np.save(f'{out_dir}/{args.model_id}_val_violations_run{args.run}.npy', val_violations)


def load_weights(model, model_id, args):
    PATH = f'./models/{args.dataset_type}/{args.model}/{args.val_ratio}/{model_id}_{args.val_ratio}_{args.run}.pth'
    ckpt = torch.load(PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()
    return model


def create_report(scores, args):
    d = args_to_dict(args) | scores
    out_dir = f'./data/tables/{args.dataset_type}/{args.model}/{args.val_ratio}'
    os.makedirs(out_dir, exist_ok=True)
    with open(f'{out_dir}/{args.model_id}_{args.val_ratio}_{args.run}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for k,v in d.items():
            writer.writerow([k, v])
            




