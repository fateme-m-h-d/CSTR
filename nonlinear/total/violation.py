from types import SimpleNamespace
import os
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from load_data import load_saved_model, make_prediction, Cao, Cbo, Cco, equations
from utils import get_scaledABb_list, MaxAbsScaler, load_data

# ─── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_PATH   = "./models/cstr/KKThPINN/0.2/MODELID_0.2_0.pth"
MODEL_TYPE   = "KKT"   # "NN" or "KKT"
INPUT_DIM    = 1
HIDDEN_DIM   = 32
HIDDEN_NUM   = 2
Z0_DIM       = 3
DATA_PATH    = "./data.csv"
TEMPS        = np.linspace(280, 600, 500)  # grid of temperatures
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transition points and steepness (in **normalized** X‐units)
TP    = [0.375, 0.425, 0.45, 0.5, 0.625]
STEEP = 800000

# ─── LOAD SCALER + DATA-LIN. COEFFS ──────────────────────────────────────────────

# 1) load & fit scaler on your original dataset (to get scaling factors)
_, scaler = load_data(DATA_PATH)

# 2) define your raw (un‐scaled) linearization coefficients here:

A_list = [
    torch.tensor([[-0.00301071551554214]], dtype=torch.float64),
    torch.tensor([[-0.0287912896000818]], dtype=torch.float64),
    torch.tensor([[-0.0589977043951539]], dtype=torch.float64),
    torch.tensor([[-0.175928980736293]], dtype=torch.float64),
    torch.tensor([[-5.68379236916079]], dtype=torch.float64),
    torch.tensor([[-198.802430563989]], dtype=torch.float64)
]
B_list = [
    torch.tensor([[-1.02825709223637, -0.0282570922363733, 0]], dtype=torch.float64),
    torch.tensor([[-1.54923960097239, -0.549239600972388,   0]], dtype=torch.float64),
    torch.tensor([[-6.41562952963354, -5.41562952963354,     0]], dtype=torch.float64),
    torch.tensor([[-47.4935472673711, -46.4935472673712,     0]], dtype=torch.float64),
    torch.tensor([[-2909.08659408943, -2908.08659408949,   0]], dtype=torch.float64),
    torch.tensor([[-168482.732203137, -168481.732203863,   0]], dtype=torch.float64)
]
b_list = [
    torch.tensor([-1.93327845562254], dtype=torch.float64),
    torch.tensor([-11.1707510081181], dtype=torch.float64),
    torch.tensor([-29.674961918774], dtype=torch.float64),
    torch.tensor([-131.782655072763], dtype=torch.float64),
    torch.tensor([-6065.64229189764], dtype=torch.float64),
    torch.tensor([-286884.958823058], dtype=torch.float64)
]
# A_list = [
#             torch.tensor([[- 198.802430563989]]),torch.tensor([[- 198.802430563989]]) ,torch.tensor([[- 198.802430563989]]), torch.tensor([[- 198.802430563989]]), torch.tensor([[- 198.802430563989]]), torch.tensor([[- 198.802430563989]])
#             # … add more rows if want more lines
#         ]
# B_list = [
#             torch.tensor([[ -168482.732203137, - 168481.732203863, 0]]),
#             torch.tensor([[ -168482.732203137, - 168481.732203863, 0]]),
#             torch.tensor([[-168482.732203137, - 168481.732203863, 0]]),
#             torch.tensor([[-168482.732203137, - 168481.732203863, 0]]),
#             torch.tensor([[-168482.732203137, - 168481.732203863, 0]]),
#             torch.tensor([[-168482.732203137, - 168481.732203863, 0]]) 
            
#         ]
# b_list = [
#             torch.tensor([-286884.958823058]),torch.tensor([-286884.958823058]),torch.tensor([-286884.958823058]), torch.tensor([-286884.958823058]), torch.tensor([-286884.958823058]), torch.tensor([-286884.958823058])
# ]
# 3) scale them exactly as you do in training
A_list, B_list, b_list = get_scaledABb_list(A_list, B_list, b_list, scaler)
A_list = [A.double().to(DEVICE) for A in A_list]
B_list = [B.double().to(DEVICE) for B in B_list]
b_list = [b.double().to(DEVICE) for b in b_list]


# ─── load & scale your original dataset ────────────────────────────────
df = pd.read_csv(DATA_PATH)                        # shape (N,4)
raw_vals   = df.values                            # (N,4)
scaled_all = scaler.transform(raw_vals)           # (N,4) – same scaler you fit above
# split into X_scaled and Y_scaled
X_scaled = scaled_all[:, :1]                      # (N,1), to rebuild X below if you want
Y_scaled = scaled_all[:, 1:]                      # (N,3)

# make a torch tensor for the true outputs Y
Y = torch.tensor(Y_scaled, dtype=torch.float64, device=DEVICE)  # (N,3)

# ─── LOAD MODEL ────────────────────────────────────────────────────────────────

model = load_saved_model(
    MODEL_PATH, MODEL_TYPE, INPUT_DIM, HIDDEN_DIM, HIDDEN_NUM, Z0_DIM,
    A_list=A_list, B_list=B_list, b_list=b_list
)
model.double().to(DEVICE).eval()

# ─── PREPARE INPUT GRID ────────────────────────────────────────────────────────

# Build a dummy array with shape (N,4) so we can reuse your make_prediction scaler logic
temps = TEMPS.reshape(-1,1)
dummy = np.zeros((len(temps),4), dtype=float)
dummy[:,0] = temps[:,0]
normed = scaler.transform(dummy)[:, :1]                  # (N,1)
X = torch.tensor(normed, dtype=torch.float64, device=DEVICE)

# ─── COMPUTE VIOLATIONS ───────────────────────────────────────────────────────

with torch.no_grad():
    Z = model(X)    # (N,3) model outputs on scaled inputs

def sigmoid(x, t, s=STEEP):
    return torch.sigmoid((x - t) / (100.0/s))

# for each constraint i, compute v_i(T) on the scaled grid:
V = []
for i, (Ai, Bi, bi) in enumerate(zip(A_list, B_list, b_list)):
    # Ai @ X.T  → (1×N),  Bi @ Z.T → (1×N),   bi.view(-1,1) repeat → (1×N)
    v_pred = Ai @ X.T + Bi @ Z.T - bi.view(1,-1)
    v_true = Ai @ X.T + Bi @ Y.T - bi.view(1,-1)
    v = v_pred 
    
    if   i == 0:
        mask = 1.0 - sigmoid(X, TP[0])
    elif i == len(A_list)-1:
        mask = sigmoid(X, TP[-1])
    else:
        mask = sigmoid(X, TP[i-1]) * (1.0 - sigmoid(X, TP[i]))
        
    v_masked = (v * mask.T).cpu().numpy().flatten()
    # mask_i_np = mask.cpu().numpy().flatten()
    V.append(v_masked)
# print("violation", V)
V = np.stack(V, axis=0)  # shape (num_regions, N)
# V = np.abs(V)
# V = sum(V)
# --- after building V (a list of 6 arrays of length N) ---

# Convert to an array of shape (6, N) for easy indexing:
V_array = np.stack(V, axis=0)    # shape (6, N)

# Now print each T with its six V’s:
for j, T in enumerate(TEMPS):
    v_vals = V_array[:, j]       # shape (6,)
    # format to something like: T=300.00 → [v0, v1, …, v5]
    print(f"T={T:.2f}: {v_vals.tolist()}")


# ─── AVERAGE VIOLATION PER TEMPERATURE ─────────────────────
# V shape: (num_constraints, N_temperatures)
V_abs = np.abs(V)
violation_avg_per_temp = V_abs.mean(axis=0)  # mean over constraints, for each T
overall_avg_violation = violation_avg_per_temp.mean()  # mean over all T

print(f"Average violation per temperature (across all constraints): {overall_avg_violation:.6f}")
# ...existing code...

# ─── PLOT ─────────────────────────────────────────────────────────────────────

# T_target = 300.0
# idx = np.argmin(np.abs(TEMPS - T_target))  # nearest point to 500 K
# print("T[idx] =", TEMPS[idx])
# for i, v in enumerate(V):
#     print(f"v[{i}] at ~300K =", v[idx])


print("V shape:", V.shape)
print("V max/min:", np.nanmax(V), np.nanmin(V))
print("|V| max:", np.nanmax(np.abs(V)))
for k in range(V.shape[0]):
    print(f"constraint {k}: max |V| =", np.nanmax(np.abs(V[k])))

plt.figure(figsize=(8,5))
# plt.plot(TEMPS, V, label="total violation")
for i, v in enumerate(V_abs):
    plt.scatter(TEMPS, v, label=f"constraint {i}")             # changed from plot to scatter
plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
plt.xlabel("Temperature (K)")
plt.ylabel("Violation $v_i=A_i x + B_i z - b_i$")
plt.yscale('log')                                              # added this line
plt.ylim(1e-25, 2e6)  
plt.title("Constraint violations vs temperature")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# plt.figure(figsize=(8,5))
# for i, v in enumerate(V):
#     v_abs = np.abs(v)
#     v_abs[v_abs == 0] = np.nan         
#     plt.plot(TEMPS, v_abs, label=f"constraint {i}")

# plt.yscale('log')
# plt.xlabel("Temperature (K)")
# plt.ylabel("Violation magnitude |A_i x + B_i z - b_i|")
# plt.title("Constraint violation magnitude vs temperature")
# plt.legend()
# plt.grid(True)
# plt.show()


# build a 1×1 input tensor for T
X_single = torch.tensor([[0.625]], dtype=torch.float64)  
# get the model’s prediction z = model(X_single) → shape (1, z_dim)
z_single = model(X_single)            
# now plug that into the linear piece:
v4 = A_list[4] @ X_single.T + B_list[4] @ z_single.T - b_list[4].view(-1,1)
v5 = A_list[5] @ X_single.T + B_list[5] @ z_single.T - b_list[5].view(-1,1)
print(f"At T=500K, v4={v4.item():.2f}, v5={v5.item():.2f}")




