import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# --- 1) Read raw data (with header!) and load scaler ---
df = pd.read_csv("data.csv")      # first row is header
raw = df[['Temperature (T)','Ca','Cb','Cc']].values  # shape (N,4)

with open("scaler.pkl","rb") as f:
    scaler = pickle.load(f)

scaled = scaler.transform(raw)    # also scales concentration columns
Tn_all = scaled[:,0]              # normalized temperature
Zn_all = scaled[:,1:4]            # normalized [Ca,Cb,Cc]

# --- 2) Mask to your region (500–550 K) in original units ---
mask = (raw[:,0] >= 500.0) & (raw[:,0] <= 550.0)
Xn = Tn_all[mask].reshape(-1,1)   # shape (M,1)
# pick the species index for region 5:
k = 1  # for example, 0=Ca,1=Cb,2=Cc; set to whatever region 5 actually constrains
zk = Zn_all[mask, k]              # shape (M,)

# --- 3) Regress z_k = m * x_n + c ---
reg = LinearRegression(fit_intercept=True)
reg.fit(Xn, zk)
m = reg.coef_[0]
c = reg.intercept_

# --- 4) Convert back to (A_i, B_i, b_i) form ---
Ai_new = -m
Bi_new = np.zeros(3, dtype=float)
Bi_new[k] = 1.0
bi_new = c

print("Refit region [500–550K]:")
print(f"  A5 = {Ai_new:.6g}")
print(f"  B5 = {Bi_new.tolist()}")
print(f"  b5 = {bi_new:.6g}")
