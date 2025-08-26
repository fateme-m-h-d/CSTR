import os
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from load_data import load_saved_model, make_prediction, Cao, Cbo, Cco, equations
from utils import get_scaledABb, MaxAbsScaler, load_data

# ───────────────────────── CONFIG ─────────────────────────
MODEL_PATH   = "./models/cstr/KKThPINN/0.2/MODELID_0.2_0.pth"
MODEL_TYPE   = "KKT"     # or "NN"
INPUT_DIM    = 1
HIDDEN_DIM   = 32
HIDDEN_NUM   = 2
Z0_DIM       = 3
DATA_PATH    = "./data.csv"
TEMPS        = np.linspace(280, 600, 200)  # grid K
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CSV_SCALED    = "predictions_scaled.csv"
CSV_UNSCALED  = "predictions_unscaled.csv"
PLOT_PATH     = "violations_vs_T.png"
ATOL_CHECK    = 1e-5

# Unscaled coefficients (single constraint)
A_raw = torch.tensor([[-198.802430563989]], dtype=torch.float64)
B_raw = torch.tensor([[-168482.732203137, -168481.732203863, 0]], dtype=torch.float64)
b_raw = torch.tensor([-286884.958823058], dtype=torch.float64)

# ───────────── LOAD SCALER & SCALE COEFFICIENTS ────────────
_, scaler = load_data(DATA_PATH)  # returns (scaled_dataset, scaler)
# scaler.scale_[0] = max(scaler.scale_[0], 800)
print(f"Scaler: {scaler}")
A_scaled, B_scaled, b_scaled = get_scaledABb(A_raw, B_raw, b_raw, scaler)
A_scaled = A_scaled.to(DEVICE).double()
B_scaled = B_scaled.to(DEVICE).double()
b_scaled = b_scaled.to(DEVICE).double()

# Also put raw coeffs on device
A_raw = A_raw.to(DEVICE).double()
B_raw = B_raw.to(DEVICE).double()
b_raw = b_raw.to(DEVICE).double()

# ─────────────── LOAD MODEL ────────────────────────────────
model = load_saved_model(MODEL_PATH, MODEL_TYPE, INPUT_DIM, HIDDEN_DIM, HIDDEN_NUM, Z0_DIM,
                         A=A_scaled, B=B_scaled, b=b_scaled)
# model = model.double().to(DEVICE)
# model.eval()
import inspect, models
print("models.py file =", models.__file__)
print("forward defined at", inspect.getsourcefile(type(model).forward))

# ───────────── PREPARE INPUTS (scaled & unscaled) ──────────
T_unscaled = TEMPS.reshape(-1, 1)   # (N,1) raw K
# Build dummy rows to transform T consistently with training scaler
dummy = np.zeros((len(T_unscaled), 4), dtype=np.float64)
dummy[:, 0] = T_unscaled[:, 0]
X_scaled_np = scaler.transform(dummy)[:, :1]           # scaled T only
X_scaled    = torch.tensor(X_scaled_np, dtype=torch.float64, device=DEVICE)  # (N,1)

# ───────────── PREDICT (scaled outputs) ────────────────────
with torch.no_grad():
    Z_scaled = model(X_scaled)  # (N,3), still in scaled space
    chunk = B_scaled.t() @ torch.inverse(B_scaled @ B_scaled.t())
    Astar_now = - chunk @ A_scaled
    Bstar_now = torch.eye(3, dtype=B_scaled.dtype, device=B_scaled.device) - chunk @ B_scaled
    bstar_now = (chunk @ b_scaled.view(-1,1)).view(-1)

print("ΔBstar =", (model.fc_fixed1.weight - Bstar_now).abs().max().item())
print("ΔAstar =", (model.fc_fixed2.weight - Astar_now).abs().max().item())
print("Δbstar =", (model.fc_fixed2.bias   - bstar_now).abs().max().item())   

        

Z_scaled_np = Z_scaled.detach().cpu().numpy()

# Inverse-scale outputs to raw (T, Ca, Cb, Cc)
def inverse_outputs(T_raw, Z_scaled_np):
    N = len(T_raw)
    pack = np.zeros((N, 4), dtype=np.float64)
    # scaled T (first column)
    pack[:, 0] = scaler.transform(np.column_stack([T_raw, np.zeros((N, 3))]))[:, 0]
    # scaled preds (next three columns)
    pack[:, 1:4] = Z_scaled_np
    inv = scaler.inverse_transform(pack)
    return inv[:, 0], inv[:, 1], inv[:, 2], inv[:, 3]  # T, Ca, Cb, Cc (raw)

T_raw_check, Ca_raw, Cb_raw, Cc_raw = inverse_outputs(T_unscaled.flatten(), Z_scaled_np)
Z_raw_vals = np.column_stack([Ca_raw, Cb_raw, Cc_raw])  # (N,3)

# ───────────── VIOLATIONS (single constraint) ─────────────
# scaled-space violation: A_scaled * X_scaled + B_scaled * Z_scaled - b_scaled
with torch.no_grad():
    v_scaled_t = (A_scaled @ X_scaled.T) + (B_scaled @ Z_scaled.T) - b_scaled.view(-1, 1)  # (1,N)
v_scaled = v_scaled_t.squeeze(0).abs().detach().cpu().numpy()  # (N,)

# raw-space violation: A_raw * T + B_raw * Z_raw - b_raw
X_raw_t = torch.tensor(T_unscaled, dtype=torch.float64, device=DEVICE)          # (N,1)
Z_raw_t = torch.tensor(Z_raw_vals, dtype=torch.float64, device=DEVICE)          # (N,3)
with torch.no_grad():
    v_raw_t = (A_raw @ X_raw_t.T) + (B_raw @ Z_raw_t.T) - b_raw.view(-1, 1)     # (1,N)
v_raw = v_raw_t.squeeze(0).abs().detach().cpu().numpy()                          # (N,)

# ───────────── SANITY CHECK (scaled vs raw) ────────────────
# Because we scaled A and B as A*xscale and B*zscale and left b unchanged,
# these two violations should match numerically (up to tiny FP drift).
if not np.allclose(v_scaled, v_raw, atol=ATOL_CHECK, rtol=0):
    mx = np.max(np.abs(v_scaled - v_raw))
    print(f"[WARN] |v_scaled - v_raw| max diff = {mx:.3e} > {ATOL_CHECK}")
else:
    print("Sanity check passed: scaled vs raw violations match within tolerance.")

# ───────────── SAVE CSVS ───────────────────────────────────
# Scaled CSV (inputs/outputs are scaled; v_scaled is in scaled space)
df_scaled = pd.DataFrame({
    "T_scaled": X_scaled_np.flatten(),
    "Ca_scaled": Z_scaled_np[:, 0],
    "Cb_scaled": Z_scaled_np[:, 1],
    "Cc_scaled": Z_scaled_np[:, 2],
    "violation_scaled": v_scaled
})
df_scaled.to_csv(CSV_SCALED, index=False)
print(f"Saved scaled predictions → {CSV_SCALED}")

# Unscaled CSV (raw variables and raw violation)
df_unscaled = pd.DataFrame({
    "T":  T_unscaled.flatten(),
    "Ca": Ca_raw,
    "Cb": Cb_raw,
    "Cc": Cc_raw,
    "violation": v_raw
})
df_unscaled.to_csv(CSV_UNSCALED, index=False)
print(f"Saved unscaled predictions → {CSV_UNSCALED}")

# ───────────── PLOT ────────────────────────────────────────
plt.figure(figsize=(9, 5))
plt.scatter(T_unscaled.flatten(), v_scaled, s=12, label="|A·X + B·Z − b|")
plt.axhline(0, linestyle='--', linewidth=0.8)
plt.xlabel("Temperature (K)")
plt.ylabel("Violation (absolute)")
plt.title("Constraint violation vs Temperature")
plt.yscale('log')
plt.ylim(1e-12, 1e4)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=300)
plt.show()
print(f"Plot saved → {PLOT_PATH}")

# ───────────── AVERAGE VIOLATION PER TEMPERATURE ──────────
overall_avg_raw = np.mean(np.abs(v_raw))
overall_avg_scl = np.mean(np.abs(v_scaled))
print(f"Average absolute violation (raw):   {overall_avg_raw:.6f}")
print(f"Average absolute violation (scaled):{overall_avg_scl:.6f}")
