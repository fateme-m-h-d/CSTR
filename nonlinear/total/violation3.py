import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
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
DATA_PATH    = "./data.csv"
TEMPS        = np.linspace(280, 600, 500)  # grid K
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CSV_SCALED    = "predictions_scaled.csv"
CSV_UNSCALED  = "predictions_unscaled.csv"
PLOT_PATH     = "violations_vs_T.png"

# Unscaled coefficients you provided (same order as RANGES)
A_raw = [
    torch.tensor([[-0.00301071551554214]]),
    torch.tensor([[-0.0287912896000818]]),
    torch.tensor([[-0.0589977043951539]]),
    torch.tensor([[-0.175928980736293]]),
    torch.tensor([[-5.68379236916079]]),
    torch.tensor([[-198.802430563989]])
]
B_raw = [
    torch.tensor([[-1.02825709223637, -0.0282570922363733, 0]]),
    torch.tensor([[-1.54923960097239, -0.549239600972388,   0]]),
    torch.tensor([[-6.41562952963354, -5.41562952963354,     0]]),
    torch.tensor([[-47.4935472673711, -46.4935472673712,     0]]),
    torch.tensor([[-2909.08659408943, -2908.08659408949,     0]]),
    torch.tensor([[-168482.732203137, -168481.732203863,     0]])
]
b_raw = [
    torch.tensor([-1.93327845562254]),
    torch.tensor([-11.1707510081181]),
    torch.tensor([-29.674961918774]),
    torch.tensor([-131.782655072763]),
    torch.tensor([-6065.64229189764]),
    torch.tensor([-286884.958823058])
]

# ───────────── LOAD SCALER & SCALE COEFFICIENTS ────────────
_, scaler = load_data(DATA_PATH)  # assuming returns (data, scaler)

A_scaled, B_scaled, b_scaled = get_scaledABb_list(A_raw, B_raw, b_raw, scaler)
A_list = [A.float().to(DEVICE) for A in A_scaled]
B_list = [B.float().to(DEVICE) for B in B_scaled]
b_list = [b.float().to(DEVICE) for b in b_scaled]

# Put raw coeffs on device too (float32)
A_raw   = [A.float().to(DEVICE) for A in A_raw]
B_raw   = [B.float().to(DEVICE) for B in B_raw]
b_raw   = [b.float().to(DEVICE) for b in b_raw]

# ─────────────── LOAD MODEL ────────────────────────────────
model = load_saved_model(MODEL_PATH, MODEL_TYPE, INPUT_DIM, HIDDEN_DIM, HIDDEN_NUM, Z0_DIM,
                         A_list=A_list, B_list=B_list, b_list=b_list)
model.to(DEVICE).eval()

# ─── 1) Load & scale your data ─────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)                # columns: T, Y1, Y2, Y3, …
raw = df.values                           # shape (N, 1 + output_dim)
scaled = scaler.transform(raw)            # (N, 1+output_dim)
X_scaled = scaled[:, :1]                  # (N,1)
Y_scaled = scaled[:, 1:]                  # (N,output_dim)

# make tensors on the right device
X = torch.tensor(X_scaled, dtype=torch.float32, device=DEVICE)  # (N,1)
Y = torch.tensor(Y_scaled, dtype=torch.float32, device=DEVICE)  # (N,output_dim)

with torch.no_grad():
    Z = model(X)    # (N,3) model outputs on scaled inputs

# pre‑transpose so we can do Ai @ X_T etc.
X_T = X.T    # (1, N)
Z_T = Z.T    # (output_dim, N)
Y_T = Y.T    # (output_dim, N)

# ─── 2) Loop through each region’s A,B,b and plot the difference ───────────
plt.figure(figsize=(8,5))

for i, (Ai, Bi, bi) in enumerate(zip(A_list, B_list, b_list)):
    # model violation
    v_pred = Ai @ X_T + Bi @ Z_T - bi.view(-1,1)
    # data violation
    v_true = Ai @ X_T + Bi @ Y_T - bi.view(-1,1)
    # difference, flattened to (N,)
    v_diff = (v_pred ).abs()
    # v_diff = (v_pred - v_true).abs()
    v_diff = v_diff.cpu().numpy().flatten()

    plt.plot(df["Temperature (T)"], v_diff, label=f"constraint {i}")

plt.xlabel("Temperature")
plt.ylabel("Predicted–True Violation")
plt.title("Piecewise Violation Difference for Each (A,B,b) Set")
plt.legend()
plt.tight_layout()
plt.show()
