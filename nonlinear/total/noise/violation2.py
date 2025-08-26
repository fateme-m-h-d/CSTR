import os
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from load_data import load_saved_model, make_prediction, Cao, Cbo, Cco, equations
from utils import get_scaledABb_list, MaxAbsScaler, load_data

# ───────────────────────── CONFIG ─────────────────────────
MODEL_PATH   = "./models/cstr/KKThPINN/0.2/MODELID_0.2_0.pth"
MODEL_TYPE   = "KKT"     # or "NN"
INPUT_DIM    = 1
HIDDEN_DIM   = 32
HIDDEN_NUM   = 2
Z0_DIM       = 3
DATA_PATH    = "./moderate_noisy_data.csv"
TEMPS        = np.linspace(280, 600, 500)  # grid K
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CSV_SCALED    = "predictions_scaled.csv"
CSV_UNSCALED  = "predictions_unscaled.csv"
PLOT_PATH     = "violations_vs_T.png"
ATOL_CHECK    = 1e-5

# T ranges (inclusive start, exclusive end except last)
RANGES = [(280,300),(300,340),(340,360),(360,400),(400,500),(500,600)]

# Unscaled coefficients you provided (same order as RANGES)
A_raw = [
    torch.tensor([[-0.00301071551554214]], dtype=torch.float64),
    torch.tensor([[-0.0287912896000818]], dtype=torch.float64),
    torch.tensor([[-0.0589977043951539]], dtype=torch.float64),
    torch.tensor([[-0.175928980736293]], dtype=torch.float64),
    torch.tensor([[-5.68379236916079]], dtype=torch.float64),
    torch.tensor([[-198.802430563989]], dtype=torch.float64)
]
B_raw = [
    torch.tensor([[-1.02825709223637, -0.0282570922363733, 0]], dtype=torch.float64),
    torch.tensor([[-1.54923960097239, -0.549239600972388,   0]], dtype=torch.float64),
    torch.tensor([[-6.41562952963354, -5.41562952963354,     0]], dtype=torch.float64),
    torch.tensor([[-47.4935472673711, -46.4935472673712,     0]], dtype=torch.float64),
    torch.tensor([[-2909.08659408943, -2908.08659408949,   0]], dtype=torch.float64),
    torch.tensor([[-168482.732203137, -168481.732203863,   0]], dtype=torch.float64)
]
b_raw = [
    torch.tensor([-1.93327845562254], dtype=torch.float64),
    torch.tensor([-11.1707510081181], dtype=torch.float64),
    torch.tensor([-29.674961918774], dtype=torch.float64),
    torch.tensor([-131.782655072763], dtype=torch.float64),
    torch.tensor([-6065.64229189764], dtype=torch.float64),
    torch.tensor([-286884.958823058], dtype=torch.float64)
]

# A_raw = [
#             torch.tensor([[- 198.802430563989]]),torch.tensor([[- 198.802430563989]]) ,torch.tensor([[- 198.802430563989]]), torch.tensor([[- 198.802430563989]]), torch.tensor([[- 198.802430563989]]), torch.tensor([[- 198.802430563989]])
#             # … add more rows if want more lines
#         ]
# B_raw = [
#             torch.tensor([[ -168482.732203137, - 168481.732203863, 0]]),
#             torch.tensor([[ -168482.732203137, - 168481.732203863, 0]]),
#             torch.tensor([[-168482.732203137, - 168481.732203863, 0]]),
#             torch.tensor([[-168482.732203137, - 168481.732203863, 0]]),
#             torch.tensor([[-168482.732203137, - 168481.732203863, 0]]),
#             torch.tensor([[-168482.732203137, - 168481.732203863, 0]]) 
            
#         ]
# b_raw = [
#             torch.tensor([-286884.958823058]),torch.tensor([-286884.958823058]),torch.tensor([-286884.958823058]), torch.tensor([-286884.958823058]), torch.tensor([-286884.958823058]), torch.tensor([-286884.958823058])
# ]
# ───────────── LOAD SCALER & SCALE COEFFICIENTS ────────────
_, scaler = load_data(DATA_PATH)  # assuming returns (data, scaler)

A_scaled, B_scaled, b_scaled = get_scaledABb_list(A_raw, B_raw, b_raw, scaler)
A_scaled = [A.double().to(DEVICE) for A in A_scaled]
B_scaled = [B.double().to(DEVICE) for B in B_scaled]
b_scaled = [b.double().to(DEVICE) for b in b_scaled]

# Put raw coeffs on device too (float32)
A_raw   = [A.double().to(DEVICE) for A in A_raw]
B_raw   = [B.double().to(DEVICE) for B in B_raw]
b_raw   = [b.double().to(DEVICE) for b in b_raw]

# ─────────────── LOAD MODEL ────────────────────────────────
model = load_saved_model(MODEL_PATH, MODEL_TYPE, INPUT_DIM, HIDDEN_DIM, HIDDEN_NUM, Z0_DIM,
                         A_list=A_scaled, B_list=B_scaled, b_list=b_scaled)
model = model.double().to(DEVICE)
model.eval()

# ─── load & scale your original dataset ────────────────────────────────
df = pd.read_csv(DATA_PATH)                        # shape (N,4)
raw_vals   = df.values                            # (N,4)
scaled_all = scaler.transform(raw_vals)           # (N,4) – same scaler you fit above
# split into X_scaled and Y_scaled
X_scaled = scaled_all[:, :1]                      # (N,1), to rebuild X below if you want
Y_scaled = scaled_all[:, 1:]                      # (N,3)

# make a torch tensor for the true outputs Y
Y = torch.tensor(Y_scaled, dtype=torch.float64, device=DEVICE)  # (N,3)

# ───────────── POINT PREDICTIONS (SCALED & UNSCALED) ─────────────
T_query = np.array([300, 340, 360, 400, 500], dtype=np.float64).reshape(-1, 1)

# 1) Scale T the same way you did for the grid
dummy_q = np.zeros((len(T_query), 4), dtype=np.float64)
dummy_q[:, 0] = T_query[:, 0]
Tq_scaled = scaler.transform(dummy_q)[:, :1]                 # (n,1)
# Tq_scaled = np.array([300/800, 340/800, 360/800, 400/800, 500/800], dtype=float).reshape(-1, 1)
Tq_scaled_t = torch.tensor(Tq_scaled, dtype=torch.float64, device=DEVICE)

with torch.no_grad():
    Zq_scaled_t = model(Tq_scaled_t)                         # (n,3)
Zq_scaled = Zq_scaled_t.cpu().numpy()

# 2) Inverse-scale to get raw Ca,Cb,Cc
pack = np.zeros((len(T_query), 4), dtype=np.float64)
pack[:, 0] = scaler.transform(np.column_stack([T_query, np.zeros((len(T_query),3))]))[:, 0]
pack[:, 1:4] = Zq_scaled
inv = scaler.inverse_transform(pack)
Tq_raw  = inv[:, 0]
Caq_raw = inv[:, 1]
Cbq_raw = inv[:, 2]
Ccq_raw = inv[:, 3]

# 3) Print nicely
print("\n=== Point Predictions ===")
for i, T0 in enumerate(T_query.flatten()):
    print(f"T = {T0:.1f} K")
    print(f"  Scaled -> T_s={Tq_scaled[i,0]:.6f}, Ca_s={Zq_scaled[i,0]:.6f}, Cb_s={Zq_scaled[i,1]:.6f}, Cc_s={Zq_scaled[i,2]:.6f}")
    print(f"  Raw    -> T  ={Tq_raw[i]:.3f}, Ca ={Caq_raw[i]:.6f}, Cb ={Cbq_raw[i]:.6f}, Cc ={Ccq_raw[i]:.6f}")
    print("-"*55)

# ───────────── PREPARE INPUTS (scaled & unscaled) ──────────
T_unscaled = TEMPS.reshape(-1,1)   # (N,1)

# build dummy to use same scaler.transform logic
dummy = np.zeros((len(T_unscaled), 4), dtype=np.float64)
dummy[:,0] = T_unscaled[:,0]  # T column only, others zeros
X_scaled_np = scaler.transform(dummy)[:, :1]   # take first col
# X_scaled = torch.tensor(X_scaled_np, dtype=torch.float64, device=DEVICE)  # (N,1)
X_scaled = torch.tensor(X_scaled, dtype=torch.float64, device=DEVICE).reshape(-1, 1)  # (N,1)

# ───────────── PREDICT (scaled space) ──────────────────────
with torch.no_grad():
    Z_scaled = model(X_scaled)  # (N,3)

# inverse-scale outputs to get unscaled z (Ca,Cb,Cc)
def inverse_outputs(T_raw, Z_scaled_np):
    """Use scaler.inverse_transform to get unscaled (T, Ca, Cb, Cc)."""
    # pack (T_scaled, preds_scaled) then inverse
    N = len(T_raw)
    pack = np.zeros((N,4), dtype=np.float64)
    pack[:,0] = scaler.transform(np.column_stack([T_raw, np.zeros((N,3))]))[:,0]  # scaled T
    pack[:,1:4] = Z_scaled_np  # scaled preds
    inv = scaler.inverse_transform(pack)
    return inv[:,0], inv[:,1], inv[:,2], inv[:,3]  # T, Ca, Cb, Cc (unscaled)

Z_scaled_np = Z_scaled.cpu().numpy()
T_raw_check, Ca_raw, Cb_raw, Cc_raw = inverse_outputs(T_unscaled.flatten(), Z_scaled_np)

# ───────────── VIOLATIONS (both spaces) ────────────────────
def compute_violation_piecewise(T_vals, X_s, Z_s, Y, A_s_list, B_s_list, b_s_list,
                                T_raw_vals, Z_raw_vals, A_r_list, B_r_list, b_r_list):
    """
    Returns:
        v_scaled_all: list of arrays (len=regions), values only where active else nan
        v_raw_all:    same but unscaled
    """
    N = len(T_vals)
    v_scaled_all = []
    v_raw_all    = []
    for idx_region, ((lo, hi), As, Bs, bs, Ar, Br, br) in enumerate(zip(
            RANGES, A_s_list, B_s_list, b_s_list, A_r_list, B_r_list, b_r_list)):
        
        mask = (T_vals >= lo) & (T_vals < hi if idx_region < len(RANGES)-1 else T_vals <= hi)
        
        norm_factor_scaled = torch.sqrt(torch.sum(As**2) + torch.sum(Bs**2)).cpu().item()
    
        # scaled
        v_s_full = np.abs((As @ X_s.T + Bs @ Z_s.T - bs.view(-1,1))).cpu().numpy().flatten()
        # v_s_normalized = np.abs(v_s_full) / norm_factor_scaled 
        # v_s_full = (As @ X_s.T + Bs @ Z_s.T - bs.view(-1,1)).cpu().numpy().flatten() - (As @ X_s.T + Bs @ Y.T - bs.view(-1,1)).cpu().numpy().flatten()
        v_s = np.full(N, np.nan)
        v_s[mask] = v_s_full[mask]
        
        
        # unscaled
        # build tensors
        X_raw_t = torch.tensor(T_raw_vals.reshape(-1,1), dtype=torch.float64, device=DEVICE)
        Z_raw_t = torch.tensor(Z_raw_vals, dtype=torch.float64, device=DEVICE)
        v_r_full = np.abs((Ar @ X_raw_t.T + Br @ Z_raw_t.T - br.view(-1,1))).cpu().numpy().flatten()
        v_r = np.full(N, np.nan)
        v_r[mask] = v_r_full[mask]
        
        v_scaled_all.append(v_s)
        v_raw_all.append(v_r)
        
        # sanity check over active indices
        diff = np.nanmax(np.abs(v_s[mask] - v_r[mask]))
        if diff > ATOL_CHECK:
            print(f"[WARN] Region {idx_region} max |v_scaled - v_raw| = {diff:.3e} > {ATOL_CHECK}")
    
    return v_scaled_all, v_raw_all

# prepare raw outputs array Z_raw_vals shape (N,3)
Z_raw_vals = np.column_stack([Ca_raw, Cb_raw, Cc_raw])

v_scaled_list, v_raw_list = compute_violation_piecewise(
    T_unscaled.flatten(), X_scaled, Z_scaled, Y,
    A_scaled, B_scaled, b_scaled,
    T_unscaled.flatten(), Z_raw_vals,
    A_raw, B_raw, b_raw
)

# global sanity check ignoring nans
s_all = np.concatenate([np.nan_to_num(v) for v in v_scaled_list])
r_all = np.concatenate([np.nan_to_num(v) for v in v_raw_list])
if not np.allclose(s_all, r_all, atol=ATOL_CHECK, rtol=0):
    print("[WARN] Global sanity check FAILED (values differ beyond tolerance).")
else:
    print("Sanity check passed: scaled vs unscaled violations match within tolerance.")

# ───────────── SAVE CSVS ───────────────────────────────────
# Scaled CSV
df_scaled = pd.DataFrame({
    "T_scaled": X_scaled_np.flatten(),
    "Ca_scaled": Z_scaled_np[:,0],
    "Cb_scaled": Z_scaled_np[:,1],
    "Cc_scaled": Z_scaled_np[:,2]
})

# add region violations (scaled)
for i, v in enumerate(v_scaled_list):
    df_scaled[f"v_scaled_region{i+1}"] = v

df_scaled.to_csv(CSV_SCALED, index=False)
print(f"Saved scaled predictions & violations → {CSV_SCALED}")

# Unscaled CSV
df_unscaled = pd.DataFrame({
    "T": T_unscaled.flatten(),
    "Ca": Ca_raw,
    "Cb": Cb_raw,
    "Cc": Cc_raw
})
for i, v in enumerate(v_raw_list):
    df_unscaled[f"v_region{i+1}"] = v

df_unscaled.to_csv(CSV_UNSCALED, index=False)
print(f"Saved unscaled predictions & violations → {CSV_UNSCALED}")

# ───────────── PLOT ────────────────────────────────────────
plt.figure(figsize=(9,5))
for i, v in enumerate(v_scaled_list):
    plt.scatter(T_unscaled, v, label=f"Region {i+1}: {RANGES[i][0]}–{RANGES[i][1]} K")
plt.axhline(0, linestyle='--', linewidth=0.8)
plt.xlabel("Temperature (K)")
plt.ylabel("Violation  A·X + B·Y − b")
plt.title("Constraint violations vs Temperature (per region)")
plt.legend()
plt.yscale('log')
plt.ylim(1e-18, 2e4)  
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=300)
plt.show()
print(f"Plot saved → {PLOT_PATH}")

# ───────────── AVERAGE VIOLATION PER TEMPERATURE ─────────────
# For unscaled violations
df_unscaled = pd.read_csv(CSV_UNSCALED)
violation_cols = [col for col in df_unscaled.columns if col.startswith("v_region")]
df_unscaled["violation_avg"] = df_unscaled[violation_cols].abs().mean(axis=1)
overall_avg = df_unscaled["violation_avg"].mean()
print(f"Average absolute violation per temperature (unscaled): {overall_avg:.6f}")

# For scaled violations (optional)
df_scaled = pd.read_csv(CSV_SCALED)
violation_cols_scaled = [col for col in df_scaled.columns if col.startswith("v_scaled_region")]
df_scaled["violation_avg"] = df_scaled[violation_cols_scaled].abs().mean(axis=1)
overall_avg_scaled = df_scaled["violation_avg"].mean()
print(f"Average violation per temperature (scaled): {overall_avg_scaled:.6f}")
