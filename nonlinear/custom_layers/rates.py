# compute_rates.py

import torch
import pandas as pd

# ─── 1) Define your constants ───────────────────────────────────────
Cao = 1 #mol/L
Cbo = 2 #mol/L
Cco = 0 #mol/L
V = 10 #L
Q = 1 #L/s
tau = V/Q #s
Afo = 10e12
Eaf = 90000 #J/mol
Aro = 10e10
Ear = 80000 #J/mol
R = 8.314 #J/mol

# ─── 2) Load your raw T values ────────────────────────────────────
# Adjust 'data.csv' and 'T' column name as needed:
df = pd.read_csv('data.csv')
T  = torch.tensor(df['Temperature (T)'].values, dtype=torch.double)  # shape [N]
Ca  = torch.tensor(df['Ca'].values, dtype=torch.double)  # shape [N]
Cb  = torch.tensor(df['Cb'].values, dtype=torch.double)  # shape [N]

T  = torch.tensor(600, dtype=torch.double)  # shape [N]

g1 = torch.tensor(df['g'].values, dtype=torch.double)  # shape [N]
# ─── 3) Compute the Arrhenius rates ───────────────────────────────
g = (Afo * torch.exp(-Eaf / (R * T)))*Ca*(Cb**2)
f = (Aro * torch.exp(-Ear / (R * T)))*(Cao-Ca+Cbo-Cb+Cco)

# ─── 4) Find the max-absolute values ─────────────────────────────
max_kf = torch.max(torch.abs(g))
max_kr = torch.max(torch.abs(f))

# ─── 5) Print results ─────────────────────────────────────────────
print(f"Max |kf| = {max_kf.item():.6e}")
print(f"Max |kr| = {max_kr.item():.6e}")
print(f"Max |g| = {torch.max(torch.abs(g1)).item():.6e}")