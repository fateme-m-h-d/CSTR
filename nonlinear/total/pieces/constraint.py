# plot_4d_predictions.py
# 4D scatter (T, Ca_pred, Cb_pred) with color & size = |A·X + B·Y − b|
# Uses model predictions; toggle 12 vs 6 segments with USE_12_REGIONS.

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.preprocessing import MaxAbsScaler

# -------------------- user settings --------------------
MODEL_PATH = "./models/cstr/KKThPINN/0.2/MODELID_0.2_0.pth"
DATA_CSV   = "data.csv"
USE_12_REGIONS = True         # set False to use the 6-piece set
USE_LOG_COLOR  = True          # Log color helps avoid saturation by a few outliers
ROBUST_PERCENT = None          # e.g., 99 or 99.5 to clip colormap top-tail (optional)
INPUT_DIM, HIDDEN_DIM, HIDDEN_NUM, Z0_DIM = 1, 32, 2, 3
# ------------------------------------------------------

# import your local modules
sys.path.insert(0, ".")
from utils import get_scaledABb_list
from models import NNOPT
import torch

def get_constraints(use_12: bool):
    if use_12:
        ranges = [(280,300),(300,340),(340,360),(360,400),
                  (400,460),(460,500),(500,530),(530,550),
                  (550,565),(565,578),(578,590),(590,600)]
        A = [
            [-0.00301071551554214],
            [-0.0287912896000818 ],
            [-0.0589977043951539 ],
            [-0.175928980736293 ],
            [-2.22034336021516  ],
            [-19.068831993137   ],
            [-67.0529314068883  ],
            [-147.195966751539  ],
            [-242.760891902504  ],
            [-356.302755070124  ],
            [-496.399591699859  ],
            [-650.139480396354  ],
        ]
        B = [
            [-1.02825709223637,   -0.0282570922363733, 0.0],
            [-1.54923960097239,   -0.549239600972388,  0.0],
            [-6.41562952963354,   -5.41562952963354,   0.0],
            [-47.4935472673711,   -46.4935472673712,   0.0],
            [-1002.53593524019,   -1001.53593524029,   0.0],
            [-11478.9269831455,   -11477.9269831524,   0.0],
            [-48213.8668993521,   -48212.866899362,    0.0],
            [-119069.962944269,   -119068.962945171,   0.0],
            [-212313.514625447,   -212312.514623723,   0.0],
            [-331422.775549942,   -331421.775550256,   0.0],
            [-487637.901829002,   -487636.901838071,   0.0],
            [-668298.267688647,   -668297.267673516,   0.0],
        ]
        b = [
            [-1.93327845562254 ],
            [-11.1707510081181 ],
            [-29.674961918774  ],
            [-131.782655072763 ],
            [-2204.88922143194 ],
            [-22375.8507664129 ],
            [-87505.3569136546 ],
            [-206400.451290504 ],
            [-357218.284591596 ],
            [-544832.512463095 ],
            [-785540.247635705 ],
            [-1058769.64213322 ],
        ]
    else:
        ranges = [(280,300),(300,340),(340,360),(360,400),(400,500),(500,600)]
        A = [
            [-0.00301071551554214],
            [-0.0287912896000818 ],
            [-0.0589977043951539 ],
            [-0.175928980736293 ],
            [-5.68379236916079  ],
            [-198.802430563989  ],
        ]
        B = [
            [-1.02825709223637, -0.0282570922363733, 0.0],
            [-1.54923960097239, -0.549239600972388 , 0.0],
            [-6.41562952963354, -5.41562952963354  , 0.0],
            [-47.4935472673711,-46.4935472673712   , 0.0],
            [-2909.08659408943,-2908.08659408949   , 0.0],
            [-168482.732203137,-168481.732203863   , 0.0],
        ]
        b = [
            [-1.93327845562254],
            [-11.1707510081181],
            [-29.674961918774 ],
            [-131.782655072763],
            [-6065.64229189764 ],
            [-286884.958823058],
        ]
    # convert to np arrays with expected shapes
    A_list = [np.array(a, dtype=np.float64).reshape(1,1) for a in A]
    B_list = [np.array(b_, dtype=np.float64).reshape(1,3) for b_ in B]
    b_list = [np.array(bb, dtype=np.float64).reshape(1,)  for bb in b]
    return ranges, A_list, B_list, b_list

def load_data_and_scaler(csv_path: str):
    df = pd.read_csv(csv_path)
    scaler = MaxAbsScaler().fit(df.values)
    # enforce T scale >= 800, matching your training pipeline
    scaler.scale_[0] = max(scaler.scale_[0], 800.0)
    return df, scaler

def build_and_predict(model_path: str, scaler: MaxAbsScaler, T_raw: np.ndarray,
                      A_raw, B_raw, b_raw) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)

    # scale A/B/b for the model space
    A_t = [torch.tensor(a, dtype=torch.float64) for a in A_raw]
    B_t = [torch.tensor(b, dtype=torch.float64) for b in B_raw]
    b_t = [torch.tensor(x, dtype=torch.float64) for x in b_raw]
    A_s, B_s, b_s = get_scaledABb_list(A_t, B_t, b_t, scaler)
    A_s = [A.to(device) for A in A_s]
    B_s = [B.to(device) for B in B_s]
    b_s = [b.to(device) for b in b_s]

    # build/load model
    model = NNOPT(INPUT_DIM, HIDDEN_DIM, HIDDEN_NUM, Z0_DIM, A_s, B_s, b_s).to(device).eval()
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=True)
    else:
        model.load_state_dict(state, strict=True)

    # prepare scaled input (only T is used)
    N = len(T_raw)
    pack = np.zeros((N,4), dtype=np.float64)
    pack[:,0] = T_raw
    X_scaled = scaler.transform(pack)[:, :1]

    with torch.no_grad():
        Xs = torch.tensor(X_scaled, dtype=torch.float64, device=device)
        Z_scaled = model(Xs).cpu().numpy()  # (N,3)

    # inverse-transform predictions back to raw space
    pack[:,0] = scaler.transform(pack)[:,0]  # scaled T
    pack[:,1:4] = Z_scaled                   # scaled outputs
    inv = scaler.inverse_transform(pack)
    Ca_pred, Cb_pred, Cc_pred = inv[:,1], inv[:,2], inv[:,3]
    Z_pred = np.column_stack([Ca_pred, Cb_pred, Cc_pred])
    return Z_pred, Ca_pred, Cb_pred, Cc_pred

def compute_piecewise_violation(T_raw, Z_raw, ranges, A_raw, B_raw, b_raw):
    T = T_raw.astype(np.float64)
    X = T.reshape(-1,1)
    viol = np.zeros_like(T, dtype=np.float64)
    region = np.full_like(T, -1, dtype=int)
    for i, ((lo,hi), A,B,b) in enumerate(zip(ranges, A_raw, B_raw, b_raw)):
        mask = (T >= lo) & (T < hi if i < len(ranges)-1 else T <= hi)
        v = (A @ X.T + B @ Z_raw.T - b.reshape(-1,1)).reshape(-1)
        viol[mask] = np.abs(v[mask])
        region[mask] = i
    return viol, region

def scatter_4d(T, Ca, Cb, val, title, out_path):
    eps = 1e-16
    v = val.copy()
    if ROBUST_PERCENT:
        vmax = np.percentile(v, ROBUST_PERCENT)
        v = np.clip(v, 0, vmax)
    # marker sizes from log magnitude (helps visibility)
    v_pos = np.clip(v, eps, None)
    sizes = 10 + 90 * (np.log10(v_pos) - np.log10(v_pos.min()))
    sizes = np.clip(sizes, 10, 120)

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111, projection="3d")
    if USE_LOG_COLOR:
        sc = ax.scatter(T, Ca, Cb, c=v, s=sizes, norm=LogNorm(vmin=v_pos.min(), vmax=v.max()))
    else:
        sc = ax.scatter(T, Ca, Cb, c=v, s=sizes)
    cb = plt.colorbar(sc, pad=0.1)
    cb.set_label("|A·X + B·Y − b|")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Ca (pred)")
    ax.set_zlabel("Cb (pred)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[saved] {out_path}   max={v.max():.3e}, median={np.median(v):.3e}")

def main():
    # data & scaler
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"DATA_CSV not found: {DATA_CSV}")
    df, scaler = load_data_and_scaler(DATA_CSV)
    T_raw = df["Temperature (T)"].astype(np.float64).values

    # constraints
    ranges, A_raw, B_raw, b_raw = get_constraints(USE_12_REGIONS)

    # predictions
    pred_source = None
    Z_pred = Ca_pred = Cb_pred = Cc_pred = None
    try:
        if os.path.exists(MODEL_PATH):
            Z_pred, Ca_pred, Cb_pred, Cc_pred = build_and_predict(
                MODEL_PATH, scaler, T_raw, A_raw, B_raw, b_raw
            )
            pred_source = f"model:{os.path.basename(MODEL_PATH)}"
        else:
            raise FileNotFoundError("model weights not found")
    except Exception as e:
        print(f"[note] model predictions unavailable ({e}).")
        # try predictions_scaled.csv (scaled outputs)
        csvs = ["predictions_scaled.csv", "predictions.csv"]
        for p in csvs:
            if os.path.exists(p):
                dfp = pd.read_csv(p)
                if {"T_scaled","Ca_scaled","Cb_scaled","Cc_scaled"}.issubset(dfp.columns):
                    N = len(dfp)
                    pack = np.zeros((N,4), dtype=np.float64)
                    pack[:,0] = dfp["T_scaled"].values
                    pack[:,1] = dfp["Ca_scaled"].values
                    pack[:,2] = dfp["Cb_scaled"].values
                    pack[:,3] = dfp["Cc_scaled"].values
                    inv = scaler.inverse_transform(pack)
                    T_raw = inv[:,0]
                    Ca_pred, Cb_pred, Cc_pred = inv[:,1], inv[:,2], inv[:,3]
                    Z_pred = np.column_stack([Ca_pred, Cb_pred, Cc_pred])
                    pred_source = p
                    break
        if pred_source is None:
            print("[fallback] using raw dataset values as predictions.")
            Ca_pred = df["Ca"].values.astype(np.float64)
            Cb_pred = df["Cb"].values.astype(np.float64)
            Cc_pred = df["Cc"].values.astype(np.float64)
            Z_pred = np.column_stack([Ca_pred, Cb_pred, Cc_pred])
            pred_source = "raw data"

    # violations in RAW space (no normalization)
    V_raw, region_id = compute_piecewise_violation(T_raw, Z_pred, ranges, A_raw, B_raw, b_raw)

    tag = "12" if USE_12_REGIONS else "6"
    title = f"4D predictions — {tag} regions — source: {pred_source} (UNSCALED)"
    out   = f"4d_pred_{tag}_unscaled.png"
    scatter_4d(T_raw, Ca_pred, Cb_pred, V_raw, title, out)

    # optional: quick CSV dump for inspection
    out_csv = f"violations_{tag}_unscaled.csv"
    pd.DataFrame({
        "T": T_raw, "Ca_pred": Ca_pred, "Cb_pred": Cb_pred, "Cc_pred": Cc_pred,
        "region": region_id, "violation_unscaled": V_raw
    }).to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")

if __name__ == "__main__":
    main()
