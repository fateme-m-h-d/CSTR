import csv
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from utils import LoadModel, get_optimizer, get_loss_func, get_violation
import copy
import os
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
from models2 import Afo, Eaf, Aro, Ear, R, Cao, Cbo, Cco, V, Q
import math

import os, csv
import numpy as np
import torch


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



def _augment_with_fg_from_nn(X, pred3, data, args):
    """
    X:  [N, input_dim]   (T is the first feature, scaled)
    pred3: [N, 3]        (Ca, Cb, Cc) from plain NN
    returns: [N, 5]      (Ca, Cb, Cc, f, g)
    """
    # Prefer data['scaler']; fall back to the global 'scaler' loaded in train.py
    sc = data.get('scaler', globals().get('scaler'))

    # Unscale temperature (MaxAbsScaler)
    T = X[:, 0:1] * torch.as_tensor(sc.scale_[0], device=pred3.device, dtype=pred3.dtype)

    # Split Ca, Cb, Cc
    Ca = pred3[:, 0:1]
    Cb = pred3[:, 1:2]
    Cc = pred3[:, 2:3]
    
    Ca_unscaled = Ca * torch.as_tensor(sc.scale_[1], device=pred3.device, dtype=pred3.dtype)  # unscale Ca
    Cb_unscaled = Cb * torch.as_tensor(sc.scale_[2], device=pred3.device, dtype=pred3.dtype)  # unscale Cb

    # Cast constants to the right device/dtype
    Afo_t = torch.as_tensor(Afo, device=pred3.device, dtype=pred3.dtype)
    Eaf_t = torch.as_tensor(Eaf, device=pred3.device, dtype=pred3.dtype)
    Aro_t = torch.as_tensor(Aro, device=pred3.device, dtype=pred3.dtype)
    Ear_t = torch.as_tensor(Ear, device=pred3.device, dtype=pred3.dtype)
    R_t   = torch.as_tensor(R,   device=pred3.device, dtype=pred3.dtype)
    Cao_t = torch.as_tensor(Cao, device=pred3.device, dtype=pred3.dtype)
    Cbo_t = torch.as_tensor(Cbo, device=pred3.device, dtype=pred3.dtype)
    Cco_t = torch.as_tensor(Cco, device=pred3.device, dtype=pred3.dtype)

    # Kinetics
    kf = Afo_t * torch.exp(-Eaf_t / (R_t * T))
    kr = Aro_t * torch.exp(-Ear_t / (R_t * T))

    # Use the same scaling indices you use elsewhere (4->f, 5->g)
    scale_f = torch.as_tensor(sc.scale_[4], device=pred3.device, dtype=pred3.dtype)
    scale_g = torch.as_tensor(sc.scale_[5], device=pred3.device, dtype=pred3.dtype)

    # Build g and f
    g = (kf / scale_g) * Ca_unscaled * (Cb_unscaled ** 2)
    f = (kr / scale_f) * (Cao_t - Ca_unscaled + Cbo_t - Cb_unscaled + Cco_t)

    return torch.cat([Ca, Cb, Cc, f, g], dim=1)


def forward_kkt_style(model, X, args, data):
    """
    Returns (pred3, pred5) for both models.

    - NN:    pred3 = model(X), pred5 = augment_with_fg_from_nn(...)
    - KKT:   pred3, pred5 = model(X)
    """
    if args.model == 'NN':
        pred3 = model(X)  # [N, 3]
        pred5 = _augment_with_fg_from_nn(X, pred3, data, args)  # [N, 5]
    else:  # 'KKThPINN'
        pred3, pred5 = model(X)  # already [N,3], [N,5]
    return pred3, pred5


# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
torch.set_default_dtype(torch.float64)

# CSTR residence time (V/Q)
tau = V / Q


def compute_three_violations(args, data, X, pred3, pred5):
    """Return three diagnostic violations as scalars (mean absolute values).

    1) v_lin: linear lifted-space residual using projected 5D output:
       r_lin = A x + B z5 - b
    2) v_nl:  nonlinear physics residual computed ONLY from pred3 (Ca,Cb,Cc)
       via utils.get_violation
    3) v_fg:  lifting consistency gap between projected (f,g) and recomputed (f*,g*)

    All are computed as mean(abs(...)).
    """
    # (1) linear lifted residual (scaled space)
    r_lin = torch.mm(data['A'], X.T) + torch.mm(data['B'], pred5.T) - data['b'].repeat(1, X.T.shape[1])
    v_lin = torch.abs(r_lin).mean()

    # (2) nonlinear residual from pred3 only (unscaled physics residual)
    r_nl = get_violation(args, data, X, pred3)  # [N,1]
    v_nl = torch.abs(r_nl).mean()

    # (3) f,g consistency gap in *unscaled* units
    sc = data['scaler']
    # scale factors on correct device/dtype
    sT  = torch.as_tensor(sc.scale_[0], device=X.device, dtype=X.dtype)
    sCa = torch.as_tensor(sc.scale_[1], device=X.device, dtype=X.dtype)
    sCb = torch.as_tensor(sc.scale_[2], device=X.device, dtype=X.dtype)
    sf  = torch.as_tensor(sc.scale_[4], device=X.device, dtype=X.dtype)
    sg  = torch.as_tensor(sc.scale_[5], device=X.device, dtype=X.dtype)

    T  = X[:, 0:1] * sT
    Ca = pred3[:, 0:1] * sCa
    Cb = pred3[:, 1:2] * sCb

    # constants to tensor (device/dtype safe)
    Afo_t = torch.as_tensor(Afo, device=X.device, dtype=X.dtype)
    Eaf_t = torch.as_tensor(Eaf, device=X.device, dtype=X.dtype)
    Aro_t = torch.as_tensor(Aro, device=X.device, dtype=X.dtype)
    Ear_t = torch.as_tensor(Ear, device=X.device, dtype=X.dtype)
    R_t   = torch.as_tensor(R,   device=X.device, dtype=X.dtype)
    Cao_t = torch.as_tensor(Cao, device=X.device, dtype=X.dtype)
    Cbo_t = torch.as_tensor(Cbo, device=X.device, dtype=X.dtype)
    Cco_t = torch.as_tensor(Cco, device=X.device, dtype=X.dtype)

    kf = Afo_t * torch.exp(-Eaf_t / (R_t * T))
    kr = Aro_t * torch.exp(-Ear_t / (R_t * T))
    f_star = kr * (Cao_t - Ca + Cbo_t - Cb + Cco_t)        # unscaled
    g_star = kf * Ca * (Cb ** 2)                           # unscaled

    f_proj = pred5[:, 3:4] * sf                            # unscaled
    g_proj = pred5[:, 4:5] * sg                            # unscaled

    r_fg = torch.cat([f_proj - f_star, g_proj - g_star], dim=1)  # [N,2]
    v_fg = torch.abs(r_fg).mean()

    return v_lin, v_nl, v_fg


# NOTE: do NOT load/fit any separate scaler here.
# Use ONLY data['scaler'] everywhere to avoid mismatches.
        

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
    
    
    
    with torch.no_grad():
        X, Y = next(iter(data['train_loader']))  # or test_loader
        X, Y = X.to(device), Y.to(device)

        # Y contains [Ca,Cb,Cc,f,g] in SCALED space
        r_true_nl = get_violation(args, data, X, Y[:, :3])  # uses only Ca,Cb,Cc scaled
        print("nonlinear violation on TRUE labels (mean abs):", r_true_nl.abs().mean().item())
        print("nonlinear violation on TRUE labels (max abs):",  r_true_nl.abs().max().item())

    # Call save function here after loading the data
    # save_training_data(data['train_loader'], "training_data.csv")
    
    print('Start Training...')
    min_loss = np.inf
    train_losses = []
    val_losses = []
    # store 3 diagnostic violations
    train_violations_lin, train_violations_nl, train_violations_fg = [], [], []
    val_violations_lin, val_violations_nl, val_violations_fg = [], [], []

    for epoch in range(args.epochs):
        #print('-------- Epoch ' + str(epoch + 1) + ' --------')
        train_loss = 0
        train_violation_lin = 0
        train_violation_nl = 0
        train_violation_fg = 0
        train_batch_mse = []
        train_batch_sizes = []

        for batch_idx, (X, Y) in enumerate(data['train_loader']):
            X, Y = X.to(device), Y.to(device)
            
            # # === APPEND TRAIN PREDICTIONS TO ONE CSV ===
            # with torch.no_grad():                    # no gradients
            #     pred_dbg = model(X)
            # pred3, pred5 = model(X)
            # Y_eff = Y[:, :args.z0_inner_dim] if args.model == "NN" else Y
            # # Y_eff = Y[:, :args.z0_inner_dim]
            # append_pred_row_csv("./debug/preds_train_all.csv",
            #                         "train", epoch + 1, batch_idx,
            #                         Y_eff, pred5)
            # # === END APPEND ===
            
            mse_loss = optimizer_step(model, optimizer, loss_func, X, Y, args, data)
            v_lin, v_nl, v_fg = conservation_step(model, X, data, args)
            train_loss += mse_loss
            train_violation_lin += v_lin
            train_violation_nl += v_nl
            train_violation_fg += v_fg
            train_batch_mse.append(float(mse_loss))
            train_batch_sizes.append(int(X.shape[0]))

        train_loss /= len(data['train_loader'])
        epoch_train_rmse = float(np.sqrt(train_loss))
        train_violation_lin /= len(data['train_loader'])
        train_violation_nl /= len(data['train_loader'])
        train_violation_fg /= len(data['train_loader'])

        val_loss, val_violation_lin, val_violation_nl, val_violation_fg, val_batch_mse, val_batch_sizes  = test(model, data, args, current_epoch=epoch+1)
        epoch_val_rmse = float(np.sqrt(val_loss))
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_violations_lin.append(train_violation_lin.detach().item())
        train_violations_nl.append(train_violation_nl.detach().item())
        train_violations_fg.append(train_violation_fg.detach().item())
        val_violations_lin.append(val_violation_lin.detach().item())
        val_violations_nl.append(val_violation_nl.detach().item())
        val_violations_fg.append(val_violation_fg.detach().item())

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
                  'v_lin(train/val): {:.3e}/{:.3e}'.format(train_violation_lin, val_violation_lin),
                  'v_nl(train/val): {:.3e}/{:.3e}'.format(train_violation_nl, val_violation_nl),
                  'v_fg(train/val): {:.3e}/{:.3e}'.format(train_violation_fg, val_violation_fg))
    print("Training Finished!")
    save_history(args, train_losses, val_losses,
                 train_violations_lin, train_violations_nl, train_violations_fg,
                 val_violations_lin, val_violations_nl, val_violations_fg)
    if args.job == 'experiment':
        scores = evaluate_model(data, args)
        

def optimizer_step(model, optimizer, loss_func, X, Y, args, data, lambda_k=None, mu_k=None):
    if isinstance(loss_func, nn.MSELoss):
        model.train()
        optimizer.zero_grad()
        # pred = model(X)
        
        pred3, pred5 = forward_kkt_style(model, X, args, data)
        
        # NEW: train NN on only the first z0_inner_dim targets (e.g., Ca,Cb,Cc)
        Y_eff = Y[:, :args.z0_inner_dim]
    
        loss = loss_func(pred3, Y_eff)
        loss.backward()
        optimizer.step()
        return loss.item()
    

def conservation_step(model, X, data, args):
    model.eval()
    pred3, pred5 = forward_kkt_style(model, X, args, data)
    return compute_three_violations(args, data, X, pred3, pred5)


def test(model, data, args, current_epoch=None):
    loss_func = get_loss_func(args, data)
    model.eval()
    test_loss = 0
    test_violation_lin = 0
    test_violation_nl = 0
    test_violation_fg = 0
    
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
            
            # pred = model(X)
            pred3, pred5 = forward_kkt_style(model, X, args, data)
            
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
            #     pred_for_violation = _augment_with_fg_from_nn(X, pred, data, args)  # [N,5]
            # else:
            #     pred_for_violation = pred
            # three diagnostic violations
            v_lin, v_nl, v_fg = compute_three_violations(args, data, X, pred3, pred5)
            # if args.loss_type == 'PINN':
            #     mse_loss, pinn_loss = loss_func(X, pred, Y)
            #     loss = mse_loss + pinn_loss
            #     test_loss += mse_loss.item()
            if args.loss_type == 'MSE':
                # Y_eff = Y[:, :args.z0_inner_dim] if args.model == 'NN' else Y
                # loss = loss_func(pred, Y_eff)
                Y_eff = Y[:, :args.z0_inner_dim]
                loss = loss_func(pred3, Y_eff)
                test_loss += loss.item()
            batch_mse_list.append(loss.item()) 
            test_violation_lin += v_lin
            test_violation_nl += v_nl
            test_violation_fg += v_fg
    test_loss /= len(data['val_loader'])  # Test set Average loss
    test_violation_lin /= len(data['val_loader'])
    test_violation_nl /= len(data['val_loader'])
    test_violation_fg /= len(data['val_loader'])
    return test_loss, test_violation_lin, test_violation_nl, test_violation_fg, batch_mse_list, batch_size_list


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
    violation_lin = 0
    violation_nl = 0
    violation_fg = 0
    rmse_inner = 0.0
    # viol_max = 0.0    
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
                # pred = model(X)           # NN: [N,3], KKT: [N,5]
                pred3, pred5 = forward_kkt_style(model, X, args, data)
                
                
    # Recompute f,g from the *labels* Ca,Cb,Cc using your existing function
                # if args.model == 'KKThPINN':
                    
                #     pred5_from_labels = _augment_with_fg_from_nn(X, Y[:, :3], data, args)  # [N,5]
                #     fg_calc  = pred5_from_labels[:, 3:5]        # recomputed f,g (scaled space)
                #     fg_label = Y[:, 3:5]                        # f,g from the dataset (scaled space)

                #     diff_mean = torch.abs(fg_calc - fg_label).mean().item()
                #     diff_max  = torch.abs(fg_calc - fg_label).max().item()
                #     print("DEBUG dataset f,g mismatch (should be ~0) mean:", diff_mean, "max:", diff_max)
                #     mse_per_dim = torch.zeros(Y.shape[1], dtype=torch.float64)
                #     mse_per_dim += ((pred - Y)**2).mean(0).cpu()
                #     rmse_per_dim = (mse_per_dim / len(data['test_loader'])).sqrt()
                #     print("DEBUG rmse dims:", rmse_per_dim.tolist())
                    # DEBUG: Check how close NN outputs are to the KKT projection    
                
                    # replicate the basis that flows into fc_fixed1:
                    # (this mirrors models.NNOPT.forward)
                    # T = X[:, :1] * torch.as_tensor(data['scaler'].scale_[0], device=X.device, dtype=X.dtype)
                    # z0 = pred[:, :3]  # KKThPINN's inner NN output is embedded inside pred; if not exposed, recompute by passing X through the inner stack or log in forward once (print-only)
                    # Ca, Cb, Cc = z0[:,0:1], z0[:,1:2], z0[:,2:3]
                    # Ca_u = Ca * data['scaler'].scale_[1]; Cb_u = Cb * data['scaler'].scale_[2]
                    # kf = torch.as_tensor(Afo) * torch.exp(-torch.as_tensor(Eaf)/(torch.as_tensor(R)*T))
                    # kr = torch.as_tensor(Aro) * torch.exp(-torch.as_tensor(Ear)/(torch.as_tensor(R)*T))
                    # g = (kf / data['scaler'].scale_[5]) * Ca_u * (Cb_u**2)
                    # f = (kr / data['scaler'].scale_[4]) * (torch.as_tensor(Cao) - Ca_u + torch.as_tensor(Cbo) - Cb_u + torch.as_tensor(Cco))
                    # pre = torch.cat([Ca, Cb, Cc, f, g], 1)       # pre-projection
                    # delta = (pred - pre).abs().max().item()
                    # print("DEBUG KKT max|projection delta|:", delta)
                    
                    

                # For violation: augment NN to 5D, KKT already 5D
                # if args.model == 'NN':
                #     pred_for_violation = _augment_with_fg_from_nn(X, pred, data, args)  # [N,5]
                # else:
                #     pred_for_violation = pred
            
                # pred_diff = get_violation(args, data, X, pred_for_violation)
                # Y_eff = Y[:, :args.z0_inner_dim] if args.model == 'NN' else Y
                # loss = loss_func(pred, Y_eff)
                
                # three diagnostic violations
                v_lin, v_nl, v_fg = compute_three_violations(args, data, X, pred3, pred5)

                Y_eff = Y[:, :args.z0_inner_dim]
                loss = loss_func(pred3, Y_eff)
                

                rmse_total += loss.item()
                violation_lin += v_lin
                violation_nl += v_nl
                violation_fg += v_fg
                # viol_max = max(viol_max, float(torch.abs(pred_diff.view(-1)).max())) 
                
                batch_size = int(X.shape[0])
                batch_mse_total = float(loss.item())
                
                # ---- FAIR RMSE on first 3 outputs (same style as above) ----
                # loss_inner = loss_func(pred[:, :args.z0_inner_dim], Y[:, :args.z0_inner_dim])
                # “inner” (3-dim) RMSE – same as above here
                loss_inner = loss_func(pred3, Y_eff)
                
                batch_mse_inner = float(loss_inner.item())
                
                rmse_inner += loss_inner.item()

        rmse_total /= len(data['test_loader'])
        rmse_total = np.sqrt(rmse_total)
            
        rmse_inner /= len(data['test_loader'])
        rmse_inner = np.sqrt(rmse_inner)

        violation_lin /= len(data['test_loader'])
        violation_nl  /= len(data['test_loader'])
        violation_fg  /= len(data['test_loader'])

        rmse_total = float(rmse_total)
        rmse_inner = float(rmse_inner)
        violation_lin = float(violation_lin.item())
        violation_nl  = float(violation_nl.item())
        violation_fg  = float(violation_fg.item())
        
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
            # keep old key for backward compatibility (use nonlinear as the main "physics" violation)
            'violation': violation_nl,
            'violation_lin': violation_lin,
            'violation_nl': violation_nl,
            'violation_fg': violation_fg,
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
        scores = {
            'rmse_total': float('nan'),
            'rmse_inner': float('nan'),
            'violation': float('nan'),
            'violation_lin': float('nan'),
            'violation_nl': float('nan'),
            'violation_fg': float('nan'),
        }
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

def save_history(args, train_losses, val_losses,
                 train_violations_lin, train_violations_nl, train_violations_fg,
                 val_violations_lin, val_violations_nl, val_violations_fg):
    newpath1 = f'./data/learning_curves/{args.dataset_type}/{args.model}/{args.val_ratio}'
    print(f"Attempting to create directory at: {os.path.abspath(newpath1)}")
    if not os.path.exists(newpath1):
        os.makedirs(newpath1)
    np.save(f'./data/learning_curves/{args.dataset_type}/{args.model}/{args.val_ratio}/{args.model_id}_train_losses_run{args.run}.npy', train_losses)
    np.save(f'./data/learning_curves/{args.dataset_type}/{args.model}/{args.val_ratio}/{args.model_id}_val_losses_run{args.run}.npy', val_losses)
    # Save all three diagnostic violations
    np.save(f'./data/learning_curves/{args.dataset_type}/{args.model}/{args.val_ratio}/{args.model_id}_train_violation_lin_run{args.run}.npy', train_violations_lin)
    np.save(f'./data/learning_curves/{args.dataset_type}/{args.model}/{args.val_ratio}/{args.model_id}_train_violation_nl_run{args.run}.npy', train_violations_nl)
    np.save(f'./data/learning_curves/{args.dataset_type}/{args.model}/{args.val_ratio}/{args.model_id}_train_violation_fg_run{args.run}.npy', train_violations_fg)
    np.save(f'./data/learning_curves/{args.dataset_type}/{args.model}/{args.val_ratio}/{args.model_id}_val_violation_lin_run{args.run}.npy', val_violations_lin)
    np.save(f'./data/learning_curves/{args.dataset_type}/{args.model}/{args.val_ratio}/{args.model_id}_val_violation_nl_run{args.run}.npy', val_violations_nl)
    np.save(f'./data/learning_curves/{args.dataset_type}/{args.model}/{args.val_ratio}/{args.model_id}_val_violation_fg_run{args.run}.npy', val_violations_fg)
