import csv
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from utils import LoadModel, get_optimizer, get_loss_func, get_violation, PINNLoss, ALMLoss
import copy
import os
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
from models import Cbo, Cco, tau, kf_const, kr_const
import math

import os, csv
import numpy as np
import torch


def update_region_stats(X, sample_violation, data, region_sum, region_count):
    """
    Accumulate mean violation per Cao region.

    X is scaled.
    sample_violation is one violation value per sample, in physical residual units.
    """

    scale_Cao = float(data["scaler"].scale_[0])

    Cao_phys = X[:, 0].detach().cpu().numpy() * scale_Cao
    sample_violation_np = sample_violation.detach().cpu().numpy()

    n_regions = len(region_sum)

    Cao_all_phys = (
        data["dataset"].dataset_tensor[:, 0].detach().cpu().numpy()
        * scale_Cao
    )

    cao_min = float(Cao_all_phys.min())
    cao_max = float(Cao_all_phys.max())

    edges = np.linspace(cao_min, cao_max, n_regions + 1)

    region_ids = np.searchsorted(edges[1:-1], Cao_phys, side="right")

    for rid, v in zip(region_ids, sample_violation_np):
        region_sum[rid] += float(v)
        region_count[rid] += 1

def recovered_kkt_loss(pred, Y, args):
    """
    pred for KKThPINN:
        [Ca, Cb1, Cb2, Cc, h]

    Y from data:
        [Ca_true, Cb_true, Cc_true, h_true]
    """

    Ca_pred  = pred[:, 0:1]
    Cb1_pred = pred[:, 1:2]
    Cb2_pred = pred[:, 2:3]
    Cc_pred  = pred[:, 3:4]

    Ca_true = Y[:, 0:1]
    Cb_true = Y[:, 1:2]
    Cc_true = Y[:, 2:3]

    w_ca = args.ca_weight
    w_cb1 = args.cb1_weight
    w_cb2 = args.cb2_weight
    w_cc = args.cc_weight

    w_sum = w_ca + w_cb1 + w_cb2 + w_cc

    loss = (
        w_ca * (Ca_pred - Ca_true).pow(2)
        + w_cb1 * (Cb1_pred - Cb_true).pow(2)
        + w_cb2 * (Cb2_pred - Cb_true).pow(2)
        + w_cc * (Cc_pred - Cc_true).pow(2)
    ).mean() / w_sum

    return loss


def cb_consistency_loss(pred, data, args):
    """
    Penalize inconsistency between Cb1 and Cb2.

    pred for KKThPINN:
        [Ca, Cb1, Cb2, Cc, h]

    This version uses scaled space, because the data loss is also in scaled space.
    Minimizing scaled Cb1-Cb2 also minimizes physical Cb1-Cb2 because both use
    the same Cb scale factor.
    """

    if args.model != "KKThPINN":
        return torch.zeros((), device=pred.device, dtype=pred.dtype)

    Cb1_pred = pred[:, 1:2]
    Cb2_pred = pred[:, 2:3]

    return (Cb1_pred - Cb2_pred).pow(2).mean()


def recovered_kkt_eval_mse(pred, Y):
    """
    Fixed evaluation metric for all weight-tuning cases.

    pred for KKThPINN:
        [Ca, Cb1, Cb2, Cc, h]

    Y from data:
        [Ca_true, Cb_true, Cc_true, h_true]

    This uses fixed 0.5 / 0.5 Cb weights no matter what
    training weights were used.
    """

    Ca_pred  = pred[:, 0:1]
    Cb1_pred = pred[:, 1:2]
    Cb2_pred = pred[:, 2:3]
    Cc_pred  = pred[:, 3:4]

    Ca_true = Y[:, 0:1]
    Cb_true = Y[:, 1:2]
    Cc_true = Y[:, 2:3]

    mse = (
        (Ca_pred - Ca_true).pow(2)
        + 0.5 * (Cb1_pred - Cb_true).pow(2)
        + 0.5 * (Cb2_pred - Cb_true).pow(2)
        + (Cc_pred - Cc_true).pow(2)
    ).mean() / 3.0

    return mse

def append_pred_row_csv(big_path: str, phase: str, epoch: int, batch_idx: int,
                        y_true_t: torch.Tensor, y_pred_t: torch.Tensor):
    """
    Appends to a single CSV file:
      columns = phase, epoch, batch_idx, row, row_MSE, y_true_0.., y_pred_0..
    - y_true_t, y_pred_t: [N, D] tensors in *scaled* space (exactly what loss uses)
    """
    os.makedirs(os.path.dirname(big_path), exist_ok=True)
    y_true = y_true_t.detach().cpu().numpy()        # [N, D]
    y_pred = y_pred_t.detach().cpu().numpy()        # [N, D]
    N, D = y_true.shape
    row_mse = ((y_pred - y_true)**2).mean(axis=1)   # per-sample MSE across D

    write_header = not os.path.exists(big_path)
    with open(big_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            header = (["phase","epoch","batch_idx","row","row_MSE"]
                      + [f"y_true_{j}" for j in range(D)]
                      + [f"y_pred_{j}" for j in range(D)])
            w.writerow(header)
        for r in range(N):
            w.writerow([phase, epoch, batch_idx, r, float(row_mse[r]),
                        *y_true[r].tolist(), *y_pred[r].tolist()])



def _augment_with_lifted_from_nn(X, pred2, data, args):
    """
    Plain NN predicts only:
        [Ca, Cb]

    This function builds:
        [Ca, Cb, kfCb2, krCa2]
    """

    sc = data['scaler']

    T = X[:, 0:1] * torch.as_tensor(
        sc.scale_[0], device=pred2.device, dtype=pred2.dtype
    )

    Ca_scaled = pred2[:, 0:1]
    Cb_scaled = pred2[:, 1:2]

    Ca = Ca_scaled * torch.as_tensor(
        sc.scale_[1], device=pred2.device, dtype=pred2.dtype
    )

    Cb = Cb_scaled * torch.as_tensor(
        sc.scale_[2], device=pred2.device, dtype=pred2.dtype
    )

    kf = torch.as_tensor(Afo, device=pred2.device, dtype=pred2.dtype) * torch.exp(
        -torch.as_tensor(Eaf, device=pred2.device, dtype=pred2.dtype)
        / (torch.as_tensor(R, device=pred2.device, dtype=pred2.dtype) * T)
    )

    kr = torch.as_tensor(Aro, device=pred2.device, dtype=pred2.dtype) * torch.exp(
        -torch.as_tensor(Ear, device=pred2.device, dtype=pred2.dtype)
        / (torch.as_tensor(R, device=pred2.device, dtype=pred2.dtype) * T)
    )

    kfCa = (kf * tau + 1) * Ca
    krCb2 = kr * (Cb ** 2)

    kfCa_scaled = kfCa / torch.as_tensor(
        sc.scale_[3], device=pred2.device, dtype=pred2.dtype
    )

    krCb2_scaled = krCb2 / torch.as_tensor(
        sc.scale_[4], device=pred2.device, dtype=pred2.dtype
    )

    return torch.cat(
        [Ca_scaled, Cb_scaled, kfCa_scaled, krCb2_scaled],
        dim=1
    )



# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
# torch.set_default_dtype(torch.float64)


# Load the scaler used for normalization
# scaler_path = "./scaler.pkl"  # Adjust this path if needed
# with open(scaler_path, 'rb') as f:
#     scaler = pickle.load(f)
#     print("Scaler loaded successfully.")
#     print("Scaler Type:", type(scaler))
#     if hasattr(scaler, 'scale_'):
#         print("Scaler Factors (scale_):", scaler.scale_)
        

def run_training(args, data):
    
    # at start of run_training (once)
    train_csv = f'./training_batch_epoch_rmse.csv'
    val_csv = f'./validation_batch_epoch_rmse.csv'
    if not os.path.exists(train_csv):
        with open(train_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['epoch','batch_idx','phase','batch_size','batch_MSE','batch_RMSE','epoch_MSE','epoch_RMSE'])
    if not os.path.exists(val_csv):
        with open(val_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['epoch','batch_idx','phase','batch_size','batch_MSE','batch_RMSE','epoch_MSE','epoch_RMSE'])


    model = LoadModel(args, data)
    optimizer = get_optimizer(args, model)
    loss_func = get_loss_func(args, data)
    
    # Call save function here after loading the data
    # save_training_data(data['train_loader'], "training_data.csv")
    
    print('Start Training...')
    min_loss = np.inf
    train_losses = []
    val_losses = []
    train_violations = []
    val_violations = []

    for epoch in range(args.epochs):
        
        if args.model == "KKThPINN" and hasattr(model, "use_kkt_projection"):
            model.use_kkt_projection = (epoch + 1) > args.kkt_warmup_epochs

            if epoch + 1 == 1:
                    print(f"KKT projection warmup: OFF for first {args.kkt_warmup_epochs} epochs")

            if epoch + 1 == args.kkt_warmup_epochs + 1:
                    print(f"KKT projection turned ON at epoch {epoch + 1}")
        #print('-------- Epoch ' + str(epoch + 1) + ' --------')
        train_loss = 0
        train_violation = 0
        train_batch_mse = []
        train_batch_sizes = []

        for batch_idx, (X, Y) in enumerate(data['train_loader']):
            X, Y = X.to(device), Y.to(device)
            
            # === APPEND TRAIN PREDICTIONS TO ONE CSV ===
            # with torch.no_grad():                    # no gradients
            #     pred_dbg = model(X)
            pred = model(X)
            Y_eff = Y[:, :args.z0_inner_dim]
            # append_pred_row_csv("./debug/preds_train_all.csv",
            #                         "train", epoch + 1, batch_idx,
            #                         Y_eff, pred)
            # === END APPEND ===

            
            mse_loss = optimizer_step(model, optimizer, loss_func, X, Y, args, data)
            pred_diff = conservation_step(model, X, data, args)
            train_loss += mse_loss
            train_violation += torch.abs(pred_diff.view(-1)).mean()
            train_batch_mse.append(float(mse_loss))
            train_batch_sizes.append(int(X.shape[0]))

        train_loss /= len(data['train_loader'])
        epoch_train_rmse = float(np.sqrt(train_loss))
        train_violation /= len(data['train_loader'])

        val_loss, val_violation, val_batch_mse, val_batch_sizes  = test(model, data, args, current_epoch=epoch+1)
        epoch_val_rmse = float(np.sqrt(val_loss))
        train_losses.append(train_loss), train_violations.append(train_violation.detach().item())
        val_losses.append(val_loss), val_violations.append(val_violation.detach().item())

        checkpoint(model, val_loss, min_loss, args, epoch)
        # best = np.minimum(min_loss, np.mean(val_loss))
        
        #####
        # write TRAIN rows
        with open(train_csv, 'a', newline='') as f:
            w = csv.writer(f)
            # per-batch
            for bidx, (bmse, bsz) in enumerate(zip(train_batch_mse, train_batch_sizes)):
                w.writerow([epoch+1, bidx, 'train', bsz, bmse, math.sqrt(bmse), train_loss, epoch_train_rmse])
            # epoch summary row (batch_idx = -1)
            w.writerow([epoch+1, -1, 'train', sum(train_batch_sizes), train_loss, epoch_train_rmse, train_loss, epoch_train_rmse])

        # write VAL rows
        with open(val_csv, 'a', newline='') as f:
            w = csv.writer(f)
            for bidx, bmse in enumerate(val_batch_mse):
                # batch size is from the loader; if needed, re-run a small loop to capture sizes
                w.writerow([epoch+1, bidx, 'val', val_batch_sizes[bidx], bmse, math.sqrt(bmse), val_loss, epoch_val_rmse])
            w.writerow([epoch+1, -1, 'val', sum(val_batch_sizes), val_loss, epoch_val_rmse, val_loss, epoch_val_rmse])
        #####


        if (epoch + 1) % 50 == 0:
            print('epoch: {:05d}'.format(epoch + 1),
                  'loss_train: {:.5f}'.format(train_loss),
                  'loss_val: {:.5f}'.format(val_loss),
                  'violation_train: {:.5f}'.format(train_violation),
                  'violation_val: {:.5f}'.format(val_violation))
    print("Training Finished!")
    save_history(args, train_losses, val_losses, train_violations, val_violations)
    if args.job == 'experiment':
        scores = evaluate_model(data, args)
        


def ca_cb_weighted_loss(pred_eff, Y_eff, data, args):
    """
    Loss = mean[ 1*(Ca - Ca_true)^2 + (2*Cb_mid)*(Cb - Cb_true)^2 ]
    """

    ca_err = pred_eff[:, 0:1] - Y_eff[:, 0:1]
    cb_err = pred_eff[:, 1:2] - Y_eff[:, 1:2]

    cb_scale = torch.as_tensor(
        data['scaler'].scale_[2],
        device=Y_eff.device,
        dtype=Y_eff.dtype
    )

    # true Cb in physical space, only for selecting the segment midpoint
    Cb_true_phys = Y_eff[:, 1:2] * cb_scale

    nseg = args.cb_loss_segments
    cb_min = args.cb_loss_min
    cb_max = args.cb_loss_max

    edges = torch.linspace(
        cb_min,
        cb_max,
        nseg + 1,
        device=Y_eff.device,
        dtype=Y_eff.dtype
    )

    idx = torch.bucketize(Cb_true_phys.squeeze(-1), edges[1:-1])

    Cb_mid = 0.5 * (edges[idx] + edges[idx + 1])
    Cb_mid = Cb_mid.unsqueeze(1)

    w_cb = 2.0 * Cb_mid

    loss = (ca_err.pow(2) + w_cb * cb_err.pow(2)).mean()

    return loss

def z4_margin_penalty(pred, data, args):
    """
    Penalize projected physical z4 if it is below a small positive margin.

    z4 = kr*Cb^2 should be nonnegative.
    We use a margin so the model stays away from the zero boundary.
    """

    if args.model != "KKThPINN":
        return torch.zeros((), device=pred.device, dtype=pred.dtype)

    if pred.shape[1] < 4:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)

    z4_scale = torch.as_tensor(
        data["scaler"].scale_[4],
        device=pred.device,
        dtype=pred.dtype
    )

    z4_phys = pred[:, 3:4] * z4_scale

    delta_z4 = torch.as_tensor(
        args.z4_margin,
        device=pred.device,
        dtype=pred.dtype
    )

    penalty = torch.relu(delta_z4 - z4_phys).pow(2).mean()

    return penalty

def optimizer_step(model, optimizer, loss_func, X, Y, args, data, lambda_k=None, mu_k=None):
    if isinstance(loss_func, PINNLoss):
        model.train()
        optimizer.zero_grad()
        pred = model(X)
        mse_loss, pinn_loss = loss_func(X, pred, Y)
        loss = mse_loss + pinn_loss
        loss.backward()
        optimizer.step()
        return mse_loss.item()
    elif isinstance(loss_func, nn.MSELoss):
        model.train()
        optimizer.zero_grad()
        pred = model(X)
        
        if args.model == "KKThPINN":
            data_loss = recovered_kkt_loss(pred, Y, args)

            cb_penalty = cb_consistency_loss(pred, data, args)

            loss = data_loss + args.lambda_cb_consistency * cb_penalty

        else:
            pred_eff = pred[:, :args.z0_inner_dim]
            Y_eff = Y[:, :args.z0_inner_dim]
            data_loss = loss_func(pred_eff, Y_eff)

            loss = data_loss
        
        
        loss.backward()
        optimizer.step()

        return data_loss.item()

        
    elif isinstance(loss_func, ALMLoss):
        for sub_iteration in range(args.max_subiter + 1):
            # if sub_iteration % 100 == 0:
            #     print(f'sub_iteration: {sub_iteration}')
            model.train()
            optimizer.zero_grad()
            pred = model(X)
            mse_loss, penalty_loss = loss_func(X, pred, Y, lambda_k, mu_k)
            loss = mse_loss + penalty_loss
            loss.backward()
            optimizer.step()
        return mse_loss.item()
    

def conservation_step(model, X, data, args):
    model.eval()
    pred = model(X)
    # if args.model == 'NN':
    #     pred = _augment_with_lifted_from_nn(X, pred, data, args)
    pred_diff = get_violation(args, data, X, pred)
    return pred_diff


def test(model, data, args, current_epoch=None):
    loss_func = get_loss_func(args, data)
    model.eval()
    test_loss = 0
    test_violation = 0
    
    batch_mse_list = []
    batch_size_list = []   # NEW

    with torch.no_grad():
        for batch_idx, (X, Y) in enumerate(data['val_loader']):
            X, Y = X.to(device), Y.to(device)
            # with torch.no_grad():
            #     pred_dbg = model(X)
            #     Y_eff_dbg = Y[:, :args.z0_inner_dim] if args.model == "NN" else Y
            #     append_pred_row_csv("./debug/preds_val_all.csv",
            #                         "val", (current_epoch or 0), batch_idx,
            #                         Y_eff_dbg, pred_dbg)
            
            pred = model(X)
            
            bsz = int(X.shape[0])           # NEW
            batch_size_list.append(bsz)     # NEW
            
    #         with torch.no_grad():
    # # Recompute f,g from the *labels* Ca,Cb,Cc using your existing function
    #             pred5_from_labels = _augment_with_fg_from_nn(X, Y[:, :3], data, args)  # [N,5]
    #             fg_calc  = pred5_from_labels[:, 3:5]        # recomputed f,g (scaled space)
    #             fg_label = Y[:, 3:5]                        # f,g from the dataset (scaled space)

    #             diff_mean = torch.abs(fg_calc - fg_label).mean().item()
    #             diff_max  = torch.abs(fg_calc - fg_label).max().item()
    #             print("DEBUG dataset f,g mismatch (should be ~0) mean:", diff_mean, "max:", diff_max)
            # if args.model == 'NN':
            #     pred_for_violation = _augment_with_lifted_from_nn(X, pred, data, args)  # [N,4]
            # else:
            #     pred_for_violation = pred
            
            pred_diff = get_violation(args, data, X, pred)
            if args.loss_type == 'PINN':
                mse_loss, pinn_loss = loss_func(X, pred, Y)
                loss = mse_loss + pinn_loss
                test_loss += mse_loss.item()
            elif args.loss_type == 'MSE':
                if args.model == "KKThPINN":
                    data_loss = recovered_kkt_loss(pred, Y, args)

                    cb_penalty = cb_consistency_loss(pred, data, args)

                    total_loss = data_loss + args.lambda_cb_consistency * cb_penalty

                else:
                    pred_eff = pred[:, :args.z0_inner_dim]
                    Y_eff = Y[:, :args.z0_inner_dim]
                    data_loss = loss_func(pred_eff, Y_eff)

                    total_loss = data_loss

                test_loss += data_loss.item()
                
            batch_mse_list.append(data_loss.item()) 
            test_violation += torch.abs(pred_diff.view(-1)).mean()
    test_loss /= len(data['val_loader'])  # Test set Average loss
    test_violation /= len(data['val_loader'])  # Test set Average violation
    return test_loss, test_violation, batch_mse_list, batch_size_list


def checkpoint(model, val_loss, min_loss, args, epoch):
    if np.mean(val_loss) < min_loss:
        checkpoint = {'model': model, 'state_dict': model.state_dict()}
        newpath = f'./models/{args.dataset_type}/{args.model}/{args.val_ratio}'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        torch.save(checkpoint, f'./models/{args.dataset_type}/{args.model}/{args.val_ratio}/{args.model_id}_{args.val_ratio}_{args.run}.pth')
        
        
def create_report(scores, args):
    args_dict = args_to_dict(args)
    # combine scores and args dict
    args_scores_dict = args_dict | scores
    # save dict
    save_dict(args_scores_dict, args)
    

def evaluate_model(data, args):
    rmse_total = 0
    violation = 0
    rmse_inner = 0.0
    # viol_max = 0.0
    cb_diff_sse = 0.0
    cb_diff_n = 0
    
    violation_g1 = 0.0
    violation_g2 = 0.0
    
    # worst-point metrics
    violation_max_any_constraint = 0.0
    violation_max_sample_mean = 0.0
    violation_g1_max = 0.0
    violation_g2_max = 0.0

    # worst-region metrics
    n_regions = getattr(args, "cao_regions_for_worst", 90)
    region_sum = np.zeros(n_regions)
    region_count = np.zeros(n_regions)
    loss_func = nn.MSELoss()
    
    try:

        model = LoadModel(args, data)
        print(f"Trying to load model: {args.model_id}_{args.val_ratio}_{args.run}.pth")
        load_weights(model, args.model_id, args)
        model.eval()
        
        
        exp_rows = []  # to capture per-batch entries
        with torch.no_grad():
            for batch_idx, (X, Y) in enumerate(data['test_loader']):
                X, Y = X.to(device), Y.to(device)
                pred = model(X)

                if args.model == "KKThPINN":
                    # Evaluation uses the same weights as training
                    objective_loss = recovered_kkt_loss(pred, Y, args)
                    eval_loss = objective_loss
                else:
                    objective_loss = loss_func(
                        pred[:, :args.z0_inner_dim],
                        Y[:, :args.z0_inner_dim]
                    )
                    eval_loss = objective_loss
                if args.model == "KKThPINN":
                    cb_scale = torch.as_tensor(
                        data["scaler"].scale_[2],
                        device=pred.device,
                        dtype=pred.dtype
                    )

                    Cb1_phys = pred[:, 1:2] * cb_scale
                    Cb2_phys = pred[:, 2:3] * cb_scale

                    cb_diff_sse += torch.sum((Cb1_phys - Cb2_phys).pow(2)).item()
                    cb_diff_n += Cb1_phys.numel()    

                pred_diff = get_violation(args, data, X, pred)
                
                abs_diff = torch.abs(pred_diff)

                # worst single constraint value over all test samples
                violation_max_any_constraint = max(
                    violation_max_any_constraint,
                    float(abs_diff.max().item())
                )

                # worst g1 and g2 separately
                violation_g1_max = max(
                    violation_g1_max,
                    float(abs_diff[:, 0].max().item())
                )

                violation_g2_max = max(
                    violation_g2_max,
                    float(abs_diff[:, 1].max().item())
                )

                # one violation value per sample = average of |g1| and |g2|
                sample_violation = abs_diff.mean(dim=1)

                violation_max_sample_mean = max(
                    violation_max_sample_mean,
                    float(sample_violation.max().item())
                )

                # region-wise accumulation
                update_region_stats(
                    X=X,
                    sample_violation=sample_violation,
                    data=data,
                    region_sum=region_sum,
                    region_count=region_count,
                )

                rmse_total += eval_loss.item()
                rmse_inner += objective_loss.item()
                violation += torch.abs(pred_diff.view(-1)).mean()
                violation_g1 += torch.abs(pred_diff[:, 0]).mean()
                violation_g2 += torch.abs(pred_diff[:, 1]).mean()

                batch_size = int(X.shape[0])
                batch_mse_total = float(eval_loss.item())
                batch_mse_inner = float(objective_loss.item())
                
        rmse_total /= len(data['test_loader'])
        rmse_total = np.sqrt(rmse_total)
            
        rmse_inner /= len(data['test_loader'])
        rmse_inner = np.sqrt(rmse_inner)
            
        violation /= len(data['test_loader'])
        violation = violation.item()
        
        violation_g1 /= len(data['test_loader'])
        violation_g1 = violation_g1.item()

        violation_g2 /= len(data['test_loader'])
        violation_g2 = violation_g2.item()
        
        valid = region_count > 0

        region_mean = np.full(n_regions, np.nan)
        region_mean[valid] = region_sum[valid] / region_count[valid]

        worst_region_id = int(np.nanargmax(region_mean))
        worst_region_violation = float(np.nanmax(region_mean))

        rmse_total = float(rmse_total)      # if rmse_total is np.float64
        violation = float(violation)        # if violation is tensor or np.float64
        violation_g1 = float(violation_g1)
        violation_g2 = float(violation_g2)
        rmse_inner = float(rmse_inner)
        
        if args.model == "KKThPINN" and cb_diff_n > 0:
            cb1_cb2_l2 = float(np.sqrt(cb_diff_sse / cb_diff_n))
        else:
            cb1_cb2_l2 = float("nan")
        
        exp_rows.append({
            'batch_idx': batch_idx,
            'batch_size': batch_size,
            'batch_MSE_total': batch_mse_total,
            'batch_RMSE_total': float(np.sqrt(batch_mse_total)),
            'batch_MSE_inner': batch_mse_inner,
            'batch_RMSE_inner': float(np.sqrt(batch_mse_inner)),
        })                
        
        # after rmse_total/rmse_inner/violation computed and cast to float
        summary = {
            'batch_idx': -1,
            'batch_size': sum(r['batch_size'] for r in exp_rows) if exp_rows else 0,
            'batch_MSE_total': float(np.mean([r['batch_MSE_total'] for r in exp_rows])) if exp_rows else float('nan'),
            'batch_RMSE_total': rmse_total,
            'batch_MSE_inner': float(np.mean([r['batch_MSE_inner'] for r in exp_rows])) if exp_rows else float('nan'),
            'batch_RMSE_inner': rmse_inner,
        }
        exp_rows.append(summary)

        csv_path = f'./experiment_batch_rmse_{args.model}_{args.val_ratio}_{args.run}.csv'
        import csv
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(exp_rows[0].keys()))
            if write_header: w.writeheader()
            w.writerows(exp_rows)

        
        scores = {
            'rmse_total': rmse_total,
            'rmse_inner': rmse_inner,
            'violation': violation,
            'cb1_cb2_l2': cb1_cb2_l2,
            'violation_g1': violation_g1,
            'violation_g2': violation_g2,

            # new worst-point metrics
            'violation_max_any_constraint': violation_max_any_constraint,
            'violation_max_sample_mean': violation_max_sample_mean,
            'violation_g1_max': violation_g1_max,
            'violation_g2_max': violation_g2_max,

            # new worst-region metric
            'worst_region_violation': worst_region_violation,
            'worst_region_id': worst_region_id,
        }
        
        # print(f"DEBUG |Ax+Bz-b| mean: {violation:.3e}  max: {viol_max:.3e}") 

        # if args.model == 'NN':
        #     post_rmse_total = 0
        #     post_rmse_unconstrained = 0
        #     post_rmse_constrained = 0

        #     chunk = torch.mm(data['B'].t(),
        #                     torch.inverse(
        #                         torch.mm(data['B'], data['B'].t())
        #                     )
        #                     )
        #     Astar = - torch.mm(chunk, data['A'])
        #     Bstar = torch.eye(args.z0_dim).to(device) - torch.mm(chunk, data['B'])
        #     bstar = torch.matmul(chunk, data['b']).squeeze(-1)

        #     with torch.no_grad():
        #         for batch_idx, (X, Y) in enumerate(data['test_loader']):
        #             X, Y = X.to(device), Y.to(device)
        #             pred = model(X)
        #             e = torch.ones((X.shape[0], 1)).to(device)
        #             pred = torch.mm(X, Astar.T) + torch.mm(pred, Bstar.T) + torch.mm(e, bstar.unsqueeze(1).T)
        #             Y_eff = Y[:, :args.z0_inner_dim] if args.model == 'NN' else Y
        #             loss = loss_func(pred, Y_eff)
                    
        #             post_rmse_total += loss.item()

        #         post_rmse_total /= len(data['test_loader'])
        #         post_rmse_total = np.sqrt(post_rmse_total)
        #         post_rmse_total = float(post_rmse_total)  # again if it’s np.float64
        #         scores.update({'post_rmse_total': post_rmse_total})

        print(scores)
        create_report(scores, args)
        return scores
    
    except FileNotFoundError as e:
        print(f"Error: Model file not found for {args.model} run {args.run}.")
        print(f"Make sure you've trained this model first: {args.model_id}_{args.val_ratio}_{args.run}.pth")
        scores = {'rmse_total': float('nan'),
                  'violation': float('nan'),
                  'cb1_cb2_l2': float('nan'),
                  'violation_g1': float('nan'),
                  'violation_g2': float('nan')}
        if args.model == 'NN':
            scores.update({'post_rmse_total': float('nan')})
        create_report(scores, args)
        return scores
    
    
def args_to_dict(args):
    return vars(args)


def save_dict(dictionary, args):
    newpath = f'./data/tables/{args.dataset_type}/{args.model}/{args.val_ratio}'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    w = csv.writer(open(f'./data/tables/{args.dataset_type}/{args.model}/{args.val_ratio}/{args.model_id}_{args.val_ratio}_{args.run}.csv', 'w'))
    # loop over dictionary keys and values
    for key, val in dictionary.items():
        # write every key and value to file
        w.writerow([key, val])



def load_weights(model, model_id, args):
    PATH = f'./models/{args.dataset_type}/{args.model}/{args.val_ratio}/{model_id}_{args.val_ratio}_{args.run}.pth'
    checkpoint = torch.load(PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(device)
    return model

def save_history(args, train_losses, val_losses, train_violations, val_violations):
    newpath1 = f'./data/learning_curves/{args.dataset_type}/{args.model}/{args.val_ratio}'
    print(f"Attempting to create directory at: {os.path.abspath(newpath1)}")
    if not os.path.exists(newpath1):
        os.makedirs(newpath1)
    np.save(f'./data/learning_curves/{args.dataset_type}/{args.model}/{args.val_ratio}/{args.model_id}_train_losses_run{args.run}.npy', train_losses)
    np.save(f'./data/learning_curves/{args.dataset_type}/{args.model}/{args.val_ratio}/{args.model_id}_val_losses_run{args.run}.npy', val_losses)
    np.save(f'./data/learning_curves/{args.dataset_type}/{args.model}/{args.val_ratio}/{args.model_id}_train_violations_run{args.run}.npy', train_violations)
    np.save(f'./data/learning_curves/{args.dataset_type}/{args.model}/{args.val_ratio}/{args.model_id}_val_violations_run{args.run}.npy', val_violations)
        

