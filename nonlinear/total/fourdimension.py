# 4D scatter for the user's CSTR data:
# X: Temperature (T)
# Y: Ca
# Z: Cb
# 4th dimension (color & marker size): |A·X + B·Y − b| piecewise, using the 6-region constraints

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Locate the dataset ----

data_path = "./data.csv"

df = pd.read_csv(data_path)
    # If columns are unnamed, assume order T, Ca, Cb, Cc
if df.shape[1] >= 4 and not set(["T","Ca","Cb","Cc"]).issubset(df.columns):
    df.columns = ["T","Ca","Cb","Cc"] + list(df.columns[4:])

# Ensure float64 for numerical stability
for col in ["T", "Ca", "Cb", "Cc"]:
    df[col] = df[col].astype(np.float64)

# ---- 6-piece constraints (unscaled) ----
# Regions: (280–300), (300–340), (340–360), (360–400), (400–500), (500–600)
A_list = [
    np.array([[-0.00301071551554214]], dtype=np.float64),
    np.array([[-0.0287912896000818 ]], dtype=np.float64),
    np.array([[-0.0589977043951539 ]], dtype=np.float64),
    np.array([[-0.175928980736293 ]], dtype=np.float64),
    np.array([[-5.68379236916079]], dtype=np.float64),  # keep as in your 6-seg set
    np.array([[-198.802430563989  ]], dtype=np.float64),
]
B_list = [
    np.array([[-1.02825709223637, -0.0282570922363733, 0.0]], dtype=np.float64),
    np.array([[-1.54923960097239, -0.549239600972388 , 0.0]], dtype=np.float64),
    np.array([[-6.41562952963354, -5.41562952963354  , 0.0]], dtype=np.float64),
    np.array([[-47.4935472673711,-46.4935472673712  , 0.0]], dtype=np.float64),
    np.array([[-2909.08659408943,-2908.08659408949  , 0.0]], dtype=np.float64),
    np.array([[-168482.732203137,-168481.732203863  , 0.0]], dtype=np.float64),
]
b_list = [
    np.array([-1.93327845562254], dtype=np.float64),
    np.array([-11.1707510081181], dtype=np.float64),
    np.array([-29.674961918774 ], dtype=np.float64),
    np.array([-131.782655072763], dtype=np.float64),
    np.array([-6065.64229189764 ], dtype=np.float64),
    np.array([-286884.958823058], dtype=np.float64),
]

ranges = [(280,300),(300,340),(340,360),(360,400),(400,500),(500,600)]

# ---- Compute piecewise violation ----
T = df["T"].to_numpy()
Ca = df["Ca"].to_numpy()
Cb = df["Cb"].to_numpy()
Cc = df["Cc"].to_numpy()
Z  = np.stack([Ca, Cb, Cc], axis=1)  # (N,3)
X  = T.reshape(-1,1)

violation = np.full_like(T, np.nan, dtype=np.float64)
region_id = np.full_like(T, -1, dtype=int)

for idx, ((lo, hi), A, B, b) in enumerate(zip(ranges, A_list, B_list, b_list)):
    mask = (T >= lo) & (T < hi if idx < len(ranges)-1 else T <= hi)
    # Compute A·X + B·Y − b (absolute value)
    v = (A @ X.T + B @ Z.T - b.reshape(-1,1)).reshape(-1)
    v = np.abs(v)
    violation[mask] = v[mask]
    region_id[mask] = idx

# Replace any residual NaNs (outside ranges) with 0 for plotting)
violation = np.nan_to_num(violation, nan=0.0)

# Normalize sizes for plotting
# Add a tiny epsilon to avoid zeros
eps = 1e-16
v_log = np.log10(violation + eps)
v_log -= v_log.min()
if v_log.max() > 0:
    v_log /= v_log.max()
sizes = 10 + 90 * v_log  # marker sizes between 10 and 100

# ---- 3D scatter: T, Ca, Cb; color + size = violation ----
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="3d")
# Do NOT set explicit colors; let matplotlib map 'c' automatically.
sc = ax.scatter(T, Ca, Cb, c=violation, s=sizes)
cb = plt.colorbar(sc, pad=0.1)
cb.set_label("|A·X + B·Y − b|")

ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Ca")
ax.set_zlabel("Cb")
ax.set_title("4D scatter with TRUE 6-region constraints (unscaled)")

out_path = "/mnt/data/4d_scatter_violation.png"
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.show()

# Also show a small preview table
preview = pd.DataFrame({
    "T": T, "Ca": Ca, "Cb": Cb, "Cc": Cc,
    "region": region_id, "violation": violation
}).head(20)

from caas_jupyter_tools import display_dataframe_to_user
display_dataframe_to_user("First 20 rows with 4D fields", preview)

out_path
