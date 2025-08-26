# plot_with_model.py

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')      # comment out if you want an interactive window
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection

# import your loader & predictor from load_data.py
from load_data import load_saved_model, make_prediction, get_scaledABb_list

# === 1) Plot original data ===
df = pd.read_csv('data.csv')
T_data  = df['Temperature (T)'].values
Ca_data = df['Ca'].values
Cb_data = df['Cb'].values

fig = plt.figure(figsize=(8,6))
ax  = fig.add_subplot(111, projection='3d')
ax.scatter(
    T_data, Ca_data, Cb_data,
    c='teal', s=15, alpha=0.05, label='Original Data'
)

# === 2) Load scaler & model ===
# adjust path if needed
scaler_path = 'scaler.pkl'
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# make sure the first (temperature) scale is large enough
scaler.scale_[0] = max(scaler.scale_[0], 800)

# Define your A_list/B_list/b_list exactly as in training,
# then scale them via get_scaledABb_list:
import torch
# raw lists (copy these from your training script)
A_list = [
    torch.tensor([[-0.00301071551554214]]),
    torch.tensor([[-0.0287912896000818]]),
    torch.tensor([[-0.0589977043951539]]),
    torch.tensor([[-0.175928980736293]]),
    torch.tensor([[-5.68379236916079]]),
    torch.tensor([[-198.802430563989]])
]
B_list = [
    torch.tensor([[-1.02825709223637, -0.0282570922363733, 0]]),
    torch.tensor([[-1.54923960097239, -0.549239600972388,    0]]),
    torch.tensor([[-6.41562952963354, -5.41562952963354,     0]]),
    torch.tensor([[-47.4935472673711, -46.4935472673712,     0]]),
    torch.tensor([[-2909.08659408943, -2908.08659408949,   0]]),
    torch.tensor([[-168482.732203137, -168481.732203863,   0]])
]
b_list = [
    torch.tensor([-1.93327845562254]),
    torch.tensor([-11.1707510081181]),
    torch.tensor([-29.674961918774]),
    torch.tensor([-131.782655072763]),
    torch.tensor([-6065.64229189764]),
    torch.tensor([-286884.958823058])
]

# scale A/B/b for the model
A_list, B_list, b_list = get_scaledABb_list(A_list, B_list, b_list, scaler)
A_list = [A_i.float() for A_i in A_list]
B_list = [B_i.float() for B_i in B_list]
b_list = [b_i.float() for b_i in b_list]

# model hyperparameters (as used in training)
input_dim, hidden_dim, hidden_num, z0_dim = 1, 32, 2, 3

# choose your checkpoint
model_type        = 'KKT'    # or 'NN'
checkpoint_folder = './models/cstr/KKThPINN/0.2'
model_path        = os.path.join(checkpoint_folder,
                                 'MODELID_0.2_0.pth')

model = load_saved_model(
    model_path, model_type,
    input_dim, hidden_dim, hidden_num, z0_dim,
    A_list=A_list, B_list=B_list, b_list=b_list
)

# === 3) Predict along a fine T grid ===
T_pred = np.linspace(280, 600, 500)
predictions = make_prediction(model, scaler, T_pred)
Ca_pred = predictions[:,0]
Cb_pred = predictions[:,1]
# Cc_pred = predictions[:,2]  # if you also want to plot Cc

# overlay the predicted curve
ax.plot(
    T_pred, Ca_pred, Cb_pred,
    color='black', linewidth=2.5, label='Predicted'
)

# === 4) Finalize plot ===
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Ca (mol/L)')
ax.set_zlabel('Cb (mol/L)')
ax.set_title('Original Data vs. Predicted: $(T, C_A, C_B)$')
ax.legend()
plt.tight_layout()
plt.savefig('Original Data_vs_predicted.png', dpi=150)
