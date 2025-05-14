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

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)

# Load the scaler used for normalization
scaler_path = "./scaler.pkl"  # Adjust this path if needed
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
    print("Scaler loaded successfully.")
    print("Scaler Type:", type(scaler))
    if hasattr(scaler, 'scale_'):
        print("Scaler Factors (scale_):", scaler.scale_)
    
# def save_training_data(train_loader, filename="training_data.csv"):
#     # Assuming the train_loader provides batches of (X, Y) tuples
#     data_list = []
    
#     for batch_idx, (X, Y) in enumerate(train_loader):
#         # Convert the tensors to numpy arrays
#         X_numpy = X.numpy()
#         Y_numpy = Y.numpy()
        
#         # Append the data to a list
#         for x, y in zip(X_numpy, Y_numpy):
#             data_list.append(list(x) + list(y))  # Combine X and Y values into one row
    
#     # Convert the list to a DataFrame
#     df = pd.DataFrame(data_list, columns=[f"Feature_{i}" for i in range(X.shape[1])] + [f"Target_{i}" for i in range(Y.shape[1])])
    
#     # Save to a CSV file
#     df.to_csv(filename, index=False)
#     print(f"Training data saved to {filename}")

# # Load the data from the CSV file
# df = pd.read_csv("training_data.csv")

# # Separate features and targets
# feature_columns = [col for col in df.columns if col.startswith("Feature_")]
# target_columns = [col for col in df.columns if col.startswith("Target_")]

# # Combine features and targets into a single array for inverse transformation
# scaled_data = pd.concat([df[feature_columns], df[target_columns]], axis=1)

# # Apply the scaler's inverse transformation
# original_data = scaler.inverse_transform(scaled_data)

# # Split the data back into features and targets
# df[feature_columns] = original_data[:, :len(feature_columns)]
# df[target_columns] = original_data[:, len(feature_columns):]

# # Save the scaled data back to a CSV file
# df.to_csv("original_training_data.csv", index=False)

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
    
    all_diffs_NN = []
    all_diffs_KKThPINN = []

    lambda_k = torch.zeros(data['A'].shape[0], requires_grad=False).to(device)
    c_best = torch.tensor(np.inf)

    for epoch in range(args.epochs):
        #print('-------- Epoch ' + str(epoch + 1) + ' --------')
        train_loss = 0
        train_violation = 0
        epoch_diffs_NN = []
        epoch_diffs_KKThPINN = []

        if args.model == 'AugLagNN':
            loss_func = ALMLoss(data['A'], data['B'], data['b'])
            for batch_idx, (X, Y) in enumerate(data['train_loader']):
                #print(f"batch {batch_idx+1} - Input X: {X}, Target Y: {Y}")
                mu_k = (batch_idx + 1) * args.mu
                X, Y = X.to(device), Y.to(device)
                mse_loss = optimizer_step(model, optimizer, loss_func, X, Y, args, data, lambda_k, mu_k)
                pred_diff = conservation_step(model, X, data, args)
                # update
                with torch.no_grad():
                    if torch.norm(pred_diff) <= args.eta * torch.norm(c_best):
                        lambda_k = (lambda_k + mu_k * pred_diff.mean(dim=-1))
                        c_best = pred_diff
                        mu_k = mu_k
                    else:
                        lambda_k = lambda_k
                        mu_k = min(args.sigma * mu_k, args.mu_safe)
                train_loss += mse_loss
                train_violation += torch.abs(pred_diff.view(-1)).mean()

        else:
            for batch_idx, (X, Y) in enumerate(data['train_loader']):
                X, Y = X.to(device), Y.to(device)
                mse_loss = optimizer_step(model, optimizer, loss_func, X, Y, args, data)

                # Forward pass using the model selected in args
                pred = model(X).cpu().detach().numpy()

                # Retrieve clean Y values (not noisy)
                clean_Y = clean_data.loc[data['train_indexes'], ['Ca', 'Cb', 'Cc']].values  # Select corresponding rows

                # Compute error per batch
                diff = pred - clean_Y
                epoch_diffs_NN.append(diff)
            
                
                pred_diff = conservation_step(model, X, data, args)
                train_loss += mse_loss
                train_violation += torch.abs(pred_diff.view(-1)).mean()
            
            # Store per-epoch differences
            all_diffs_NN.append(np.concatenate(epoch_diffs_NN, axis=0))

        train_loss /= len(data['train_loader'])
        train_violation /= len(data['train_loader'])

        val_loss, val_violation = test(model, data, args)
        train_losses.append(train_loss), train_violations.append(train_violation.detach().item())
        val_losses.append(val_loss), val_violations.append(val_violation.detach().item())

        checkpoint(model, val_loss, min_loss, args, epoch)
        best = np.minimum(min_loss, np.mean(val_loss))

        if (epoch + 1) % 50 == 0:
            print('epoch: {:05d}'.format(epoch + 1),
                  'loss_train: {:.5f}'.format(train_loss),
                  'loss_val: {:.5f}'.format(val_loss),
                  'violation_train: {:.5f}'.format(train_violation),
                  'violation_val: {:.5f}'.format(val_violation))
    print("Finished!")
    save_history(args, train_losses, val_losses, train_violations, val_violations)
    # Save per-epoch differences
    np.save('./data/learning_curves/cstr/NN/0.2/MODELID_train_diff.npy', np.array(all_diffs_NN))
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
        loss = loss_func(pred, Y)
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
            pred_diff = get_violation(args, data, X, pred)
            if args.loss_type == 'PINN':
                mse_loss, pinn_loss = loss_func(X, pred, Y)
                loss = mse_loss + pinn_loss
                test_loss += mse_loss.item()
            elif args.loss_type == 'MSE':
                loss = loss_func(pred, Y)
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
    rmse_unconstrained = 0
    rmse_constrained = 0
    violation = 0
    loss_func = nn.MSELoss()

    model = LoadModel(args, data)
    load_weights(model, args.model_id, args)
    model.eval()

    with torch.no_grad():
        for batch_idx, (X, Y) in enumerate(data['test_loader']):
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            pred_diff = get_violation(args, data, X, pred)
            loss = loss_func(pred, Y)

            for constrained_index in data['constrained_indexes']:
                rmse_constrained = loss_func(pred[:, constrained_index], Y[:, constrained_index]).item()
            for unconstrained_index in data['unconstrained_indexes']:
                rmse_unconstrained = loss_func(pred[:, unconstrained_index], Y[:, unconstrained_index]).item()
            rmse_total += loss.item()
            rmse_constrained /= len(data['constrained_indexes'])
            rmse_unconstrained /= len(data['unconstrained_indexes'])
            violation += torch.abs(pred_diff.view(-1)).mean()

        rmse_total /= len(data['test_loader'])
        rmse_total = np.sqrt(rmse_total)

        rmse_unconstrained /= len(data['test_loader'])
        rmse_unconstrained = np.sqrt(rmse_unconstrained)

        rmse_constrained /= len(data['test_loader'])
        rmse_constrained = np.sqrt(rmse_constrained)

        violation /= len(data['test_loader'])
        violation = violation.item()

    scores = {'rmse_total': rmse_total, 'rmse_unconstrained': rmse_unconstrained, 'rmse_constrained': rmse_constrained,
                'violation': violation}

    if args.model == 'NN':
        post_rmse_total = 0
        post_rmse_unconstrained = 0
        post_rmse_constrained = 0

        chunk = torch.mm(data['B'].t(),
                         torch.inverse(
                             torch.mm(data['B'], data['B'].t())
                         )
                         )
        Astar = - torch.mm(chunk, data['A'])
        Bstar = torch.eye(args.z0_dim).to(device) - torch.mm(chunk, data['B'])
        bstar = torch.matmul(chunk, data['b']).squeeze(-1)

        with torch.no_grad():
            for batch_idx, (X, Y) in enumerate(data['test_loader']):
                X, Y = X.to(device), Y.to(device)
                pred = model(X)
                e = torch.ones((X.shape[0], 1)).to(device)
                pred = torch.mm(X, Astar.T) + torch.mm(pred, Bstar.T) + torch.mm(e, bstar.unsqueeze(1).T)
                loss = loss_func(pred, Y)
                for constrained_index in data['constrained_indexes']:
                    post_rmse_constrained = loss_func(pred[:, constrained_index], Y[:, constrained_index]).item()
                for unconstrained_index in data['unconstrained_indexes']:
                    post_rmse_unconstrained = loss_func(pred[:, unconstrained_index], Y[:, unconstrained_index]).item()
                post_rmse_constrained /= len(data['constrained_indexes'])
                post_rmse_unconstrained /= len(data['unconstrained_indexes'])
                post_rmse_total += loss.item()

            post_rmse_total /= len(data['test_loader'])
            post_rmse_total = np.sqrt(post_rmse_total)
            scores.update({'post_rmse_total': post_rmse_total})

            post_rmse_unconstrained /= len(data['test_loader'])
            post_rmse_unconstrained = np.sqrt(post_rmse_unconstrained)
            scores.update({'post_rmse_unconstrained': post_rmse_unconstrained})

            post_rmse_constrained /= len(data['test_loader'])
            post_rmse_constrained = np.sqrt(post_rmse_constrained)
            scores.update({'post_rmse_constrained': post_rmse_constrained})

    print(scores)
    create_report(scores, args)


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
    checkpoint = torch.load(PATH)
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

loss_NN = np.load('./data/learning_curves/cstr/NN/0.2/MODELID_train_losses_run0.npy')
loss_KKThPINN = np.load('./data/learning_curves/cstr/KKThPINN/0.2/MODELID_train_losses_run0.npy') 

clean_data = pd.read_csv("./data.csv")
noisy_data = pd.read_csv("./large_noisy_data.csv")

diff_NN = (noisy_data[['Ca', 'Cb', 'Cc']].values - clean_data[['Ca', 'Cb', 'Cc']].values)
corrected_loss_NN = loss_NN + np.mean(np.abs(diff_NN), axis=1)
corrected_loss_KKThPINN=loss_KKThPINN + np.mean(np.abs(diff_NN), axis=1)

np.save('./data/learning_curves/cstr/NN/0.2/MODELID_train_losses_run0.npy', corrected_loss_NN)
np.save('./data/learning_curves/cstr/KKThPINN/0.2/MODELID_train_losses_run0.npy', corrected_loss_KKThPINN)
loss_KKThPINN = np.load('./data/learning_curves/cstr/KKThPINN/0.2/MODELID_train_losses_run0.npy') 
val_loss_KKThPINN = np.load('./data/learning_curves/cstr/KKThPINN/0.2/MODELID_val_losses_run0.npy')  
violation_KKThPINN = np.load('./data/learning_curves/cstr/KKThPINN/0.2/MODELID_train_violations_run0.npy')  
val_violation_KKThPINN = np.load('./data/learning_curves/cstr/KKThPINN/0.2/MODELID_val_violations_run0.npy') 

loss_NN = np.load('./data/learning_curves/cstr/NN/0.2/MODELID_train_losses_run0.npy')  
val_loss_NN = np.load('./data/learning_curves/cstr/NN/0.2/MODELID_val_losses_run0.npy')  
violation_NN = np.load('./data/learning_curves/cstr/NN/0.2/MODELID_train_violations_run0.npy') 
val_violation_NN = np.load('./data/learning_curves/cstr/NN/0.2/MODELID_val_violations_run0.npy') 



# Print the last epoch losses for KKThPINN
print(f"KKThPINN Training Loss (Epoch 1000): {loss_KKThPINN[-1]}")
print(f"KKThPINN Validation Loss (Epoch 1000): {val_loss_KKThPINN[-1]}")

# Print the last epoch losses for NN
print(f"NN Training Loss (Epoch 1000): {loss_NN[-1]}")
print(f"NN Validation Loss (Epoch 1000): {val_loss_NN[-1]}")

# loss_ECNN = np.load('./data/learning_curves/cstr/ECNN/0.2/MODELID_train_losses_run0.npy')  
# val_loss_ECNN = np.load('./data/learning_curves/cstr/ECNN/0.2/MODELID_val_losses_run0.npy')  
# violation_ECNN = np.load('./data/learning_curves/cstr/ECNN/0.2/MODELID_train_violations_run0.npy')  
# val_violation_ECNN = np.load('./data/learning_curves/cstr/ECNN/0.2/MODELID_val_violations_run0.npy')


# Plotting the loss curve
plt.figure(figsize=(10, 5))
plt.grid()
plt.plot(loss_KKThPINN, 'g--', label='KKThPINN Training Loss')
plt.plot(val_loss_KKThPINN, 'g', label='KKThPINN Validation Loss')

# plt.plot(loss_ECNN, 'r--', label='ECNN Training Loss')
# plt.plot(val_loss_ECNN, 'r', label='ECNN Validation Loss')

plt.plot(loss_NN, 'b--', label=' NN Training Loss')
plt.plot(val_loss_NN, 'b', label='NN Validation Loss')

plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.ylim((0, 1))
plt.legend()
plt.show()


# Plotting the accuracy curve
plt.figure(figsize=(10, 5))
plt.plot(violation_KKThPINN, 'g--', label='KKThPINN Training violation')
plt.plot(val_violation_KKThPINN, 'g', label='KKThPINN Validation violation')

# plt.plot(violation_ECNN, 'r--', label='ECNN Training violation')
# plt.plot(val_violation_ECNN, 'r', label='ECNN Validation violation')

plt.plot(violation_NN, 'b--', label='NN Training violation')
plt.plot(val_violation_NN, 'b', label='NN Validation violation')

plt.title('violation over Epochs')
plt.xlabel('Epochs')
plt.ylabel('violation')
plt.yscale('log')
plt.legend()
plt.show()

clean_data = pd.read_csv("./data.csv")
noisy_data = pd.read_csv("./large_noisy_data.csv")

diff_NN = (noisy_data[['Ca', 'Cb', 'Cc']].values - clean_data[['Ca', 'Cb', 'Cc']].values)
corrected_loss_NN = loss_NN + np.mean(np.abs(diff_NN), axis=1)

np.save('./data/learning_curves/cstr/NN/0.2/MODELID_train_losses_run0.npy', corrected_loss_NN)


# Load per-epoch difference
diff_NN = np.load('./data/learning_curves/cstr/NN/0.2/MODELID_train_diff.npy')

# Compute RMSE per epoch
error_NN = np.sqrt(np.mean(diff_NN**2, axis=(1, 2)))  # Compute per-epoch RMSE

# Normalize errors for visualization
error_scaling_factor_NN = np.max(error_NN)
normalized_error_NN = error_NN / error_scaling_factor_NN

# Plot error vs. training loss
plt.figure(figsize=(10, 5))
plt.plot(loss_NN, label='NN Training Loss', linestyle='--', color='blue')
plt.plot(normalized_error_NN, label='NN Error w.r.t Clean Data', linestyle='-', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss / Normalized Error')
plt.title('Training Loss vs. Error Relative to Clean Data')
plt.legend()
plt.grid()
plt.show()



