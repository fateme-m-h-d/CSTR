# Re-run with graceful fallback to synthetic data if data.csv is missing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, glob

# Try to load real dataset
data_path = "./data.csv"

df = pd.read_csv(data_path)
if df.shape[1] >= 4 and not set(["T","Ca","Cb","Cc"]).issubset(df.columns):
    df.columns = ["T","Ca","Cb","Cc"] + list(df.columns[4:])

for col in ["T","Ca","Cb","Cc"]:
    df[col] = df[col].astype(np.float64)

T = df["T"].to_numpy()
Ca = df["Ca"].to_numpy()
Cb = df["Cb"].to_numpy()
Cc = df["Cc"].to_numpy()
Z  = np.stack([Ca,Cb,Cc], axis=1)
X  = T.reshape(-1,1)

# 12 regions + true 12-piece coefficients
ranges12 = [(280,300),(300,340),(340,360),(360,400),
            (400,460),(460,500),(500,530),(530,550),
            (550,565),(565,578),(578,590),(590,600)]

A_list = [
    np.array([[-0.00301071551554214]]),
    np.array([[-0.0287912896000818 ]]),
    np.array([[-0.0589977043951539 ]]),
    np.array([[-0.175928980736293 ]]),
    np.array([[-2.22034336021516 ]]),
    np.array([[-19.068831993137   ]]),
    np.array([[-67.0529314068883 ]]),
    np.array([[-147.195966751539 ]]),
    np.array([[-242.760891902504 ]]),
    np.array([[-356.302755070124 ]]),
    np.array([[-496.399591699859 ]]),
    np.array([[-650.139480396354 ]]),
]
B_list = [
    np.array([[-1.02825709223637,   -0.0282570922363733, 0.0]]),
    np.array([[-1.54923960097239,   -0.549239600972388,  0.0]]),
    np.array([[-6.41562952963354,   -5.41562952963354,   0.0]]),
    np.array([[-47.4935472673711,   -46.4935472673712,   0.0]]),
    np.array([[-1002.53593524019,   -1001.53593524029,   0.0]]),
    np.array([[-11478.9269831455,   -11477.9269831524,   0.0]]),
    np.array([[-48213.8668993521,   -48212.866899362,    0.0]]),
    np.array([[-119069.962944269,   -119068.962945171,   0.0]]),
    np.array([[-212313.514625447,   -212312.514623723,   0.0]]),
    np.array([[-331422.775549942,   -331421.775550256,   0.0]]),
    np.array([[-487637.901829002,   -487636.901838071,   0.0]]),
    np.array([[-668298.267688647,   -668297.267673516,   0.0]]),
]
b_list = [
    np.array([-1.93327845562254 ]),
    np.array([-11.1707510081181 ]),
    np.array([-29.674961918774  ]),
    np.array([-131.782655072763 ]),
    np.array([-2204.88922143194 ]),
    np.array([-22375.8507664129 ]),
    np.array([-87505.3569136546 ]),
    np.array([-206400.451290504 ]),
    np.array([-357218.284591596 ]),
    np.array([-544832.512463095 ]),
    np.array([-785540.247635705 ]),
    np.array([-1058769.64213322 ]),
]

# Compute piecewise violation
violation = np.zeros_like(T, dtype=np.float64)
region_id = np.full_like(T, -1, dtype=int)

for idx, ((lo,hi), A,B,b) in enumerate(zip(ranges12, A_list, B_list, b_list)):
    mask = (T >= lo) & (T < hi if idx < len(ranges12)-1 else T <= hi)
    v = (A @ X.T + B @ Z.T - b.reshape(-1,1)).reshape(-1)
    violation[mask] = np.abs(v[mask])
    region_id[mask] = idx

# Map sizes via log scale
eps = 1e-16
v_log = np.log10(violation + eps)
v_log -= v_log.min()
if v_log.max() > 0:
    v_log /= v_log.max()
sizes = 10 + 90 * v_log

# Plot
from mpl_toolkits.mplot3d import Axes3D  # noqa
fig = plt.figure(figsize=(8,6))
ax  = fig.add_subplot(111, projection="3d")
sc  = ax.scatter(T, Ca, Cb, c=violation, s=sizes)
cb  = plt.colorbar(sc, pad=0.1)
cb.set_label("|A·X + B·Y − b|")

ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Ca")
ax.set_zlabel("Cb")
ax.set_title("4D scatter with TRUE 12-region constraints (unscaled)")

out_path = "/mnt/data/4d_scatter_violation_12regions_TRUE.png"
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.show()

from caas_jupyter_tools import display_dataframe_to_user
preview = pd.DataFrame({"T":T,"Ca":Ca,"Cb":Cb,"Cc":Cc,"region":region_id,"violation":violation}).head(20)
display_dataframe_to_user("First 20 rows (12 regions, unscaled violation)", preview)

out_path
