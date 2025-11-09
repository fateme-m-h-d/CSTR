# visualize_violation_nonlinear.py
import os
import numpy as np
import torch

# --- backend must be set BEFORE importing pyplot ---
import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from argparse import Namespace
from utils import LoadData, LoadModel, get_violation
from train import load_weights

# ==== USER CONFIG ====
args = Namespace(
    model="KKThPINN",
    model_id="MODELID",          # <-- replace with your saved id
    dataset_type="cstr",
    dataset_path="./data.csv",
    val_ratio=0.2,
    run=0,
    input_dim=1,
    hidden_dim=32,
    hidden_num=2,
    z0_dim=5,
    z0_inner_dim=3,
    dtype=64,
    batch_size=64                 # <-- REQUIRED by LoadData
)

# ==== LOAD DATA & MODEL ====
data = LoadData(args)
model = LoadModel(args, data)
load_weights(model, args.model_id, args)
model.eval()

# ==== EVALUATE ====
T_all, violation_all = [], []
with torch.no_grad():
    for X, Y in data['test_loader']:
        pred = model(X)  # KKThPINN returns 5-dim (scaled)
        v = get_violation(args, data, X, pred)            # shape [1, batch]
        v = torch.abs(v).squeeze(0).cpu().numpy()         # per-sample |Ax+Bz-b|
        T_unscaled = (X[:, 0] * data['scaler'].scale_[0]).cpu().numpy()
        T_all.append(T_unscaled)
        violation_all.append(v)

T_all = np.concatenate(T_all)
violation_all = np.concatenate(violation_all)

# ==== PLOT ====
os.makedirs("./plots", exist_ok=True)
plt.figure(figsize=(8,5))
plt.scatter(T_all, violation_all, s=20, alpha=0.75)
plt.xlabel("Temperature (K)")
plt.ylabel("|A·x + B·z − b| (violation)")
plt.yscale("log")
plt.title("Nonlinear KKThPINN – Constraint Violation vs Temperature")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./plots/violation_vs_temperature_nonlinear.png", dpi=300)
print("Saved: ./plots/violation_vs_temperature_nonlinear.png")
