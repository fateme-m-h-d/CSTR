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
from models import Afo, Eaf, Aro, Ear, R, Cao, Cbo, Cco

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
    g = (kf / scale_g) * Ca * (Cb ** 2)
    f = (kr / scale_f) * (Cao_t - Ca + Cbo_t - Cb + Cco_t)

    return torch.cat([Ca, Cb, Cc, f, g], dim=1)



# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
torch.set_default_dtype(torch.float64)


# Load the scaler used for normalization
scaler_path = "./scaler.pkl"  # Adjust this path if needed
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
    print("Scaler loaded successfully.")
    print("Scaler Type:", type(scaler))
    if hasattr(scaler, 'scale_'):
        print("Scaler Factors (scale_):", scaler.scale_)
        

def run_training(args, data):
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
        #print('-------- Epoch ' + str(epoch + 1) + ' --------')
        train_loss = 0
        train_violation = 0

        for batch_idx, (X, Y) in enumerate(data['train_loader']):
            X, Y = X.to(device), Y.to(device)
            mse_loss = optimizer_step(model, optimizer, loss_func, X, Y, args, data)
            pred_diff = conservation_step(model, X, data, args)
            train_loss += mse_loss
            train_violation += torch.abs(pred_diff.view(-1)).mean()

        train_loss /= len(data['train_loader'])
        train_violation /= len(data['train_loader'])

        val_loss, val_violation = test(model, data, args)
        train_losses.append(train_loss), train_violations.append(train_violation.detach().item())
        val_losses.append(val_loss), val_violations.append(val_violation.detach().item())

        checkpoint(model, val_loss, min_loss, args, epoch)
        # best = np.minimum(min_loss, np.mean(val_loss))

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
        
        # NEW: train NN on only the first z0_inner_dim targets (e.g., Ca,Cb,Cc)
        Y_eff = Y[:, :args.z0_inner_dim] if args.model == 'NN' else Y
    
        loss = loss_func(pred, Y_eff)
        loss.backward()
        optimizer.step()
        return loss.item()
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
    if args.model == 'NN':
        pred = _augment_with_fg_from_nn(X, pred, data, args)
    pred_diff = get_violation(args, data, X, pred)
    return pred_diff


def test(model, data, args):
    loss_func = get_loss_func(args, data)
    model.eval()
    test_loss = 0
    test_violation = 0

    with torch.no_grad():
        for batch_idx, (X, Y) in enumerate(data['val_loader']):
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            if args.model == 'NN':
                pred_for_violation = _augment_with_fg_from_nn(X, pred, data, args)  # [N,5]
            else:
                pred_for_violation = pred
            pred_diff = get_violation(args, data, X, pred_for_violation)
            if args.loss_type == 'PINN':
                mse_loss, pinn_loss = loss_func(X, pred, Y)
                loss = mse_loss + pinn_loss
                test_loss += mse_loss.item()
            elif args.loss_type == 'MSE':
                Y_eff = Y[:, :args.z0_inner_dim] if args.model == 'NN' else Y
                loss = loss_func(pred, Y_eff)
                test_loss += loss.item()
            test_violation += torch.abs(pred_diff.view(-1)).mean()
    test_loss /= len(data['val_loader'])  # Test set Average loss
    test_violation /= len(data['val_loader'])  # Test set Average violation
    return test_loss, test_violation


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
    loss_func = nn.MSELoss()
    
    try:

        model = LoadModel(args, data)
        print(f"Trying to load model: {args.model_id}_{args.val_ratio}_{args.run}.pth")
        load_weights(model, args.model_id, args)
        model.eval()

        with torch.no_grad():
            for batch_idx, (X, Y) in enumerate(data['test_loader']):
                X, Y = X.to(device), Y.to(device)
                pred = model(X)
                if args.model == 'NN':
                    pred_for_violation = _augment_with_fg_from_nn(X, pred, data, args)  # [N,5]
                else:
                    pred_for_violation = pred
                pred_diff = get_violation(args, data, X, pred_for_violation)
                Y_eff = Y[:, :args.z0_inner_dim] if args.model == 'NN' else Y
                loss = loss_func(pred, Y_eff)

                rmse_total += loss.item()
                violation += torch.abs(pred_diff.view(-1)).mean()

            rmse_total /= len(data['test_loader'])
            rmse_total = np.sqrt(rmse_total)

            violation /= len(data['test_loader'])
            violation = violation.item()

        rmse_total = float(rmse_total)      # if rmse_total is np.float64
        violation = float(violation)        # if violation is tensor or np.float64
        
        scores = {'rmse_total': rmse_total,
                    'violation': violation}

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
                  'violation': float('nan')}
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
        

