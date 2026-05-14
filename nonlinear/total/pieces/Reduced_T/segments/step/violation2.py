import os
import re
import glob
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from models import NN, NNOPT

# ============================================================
# CONFIG
# ============================================================
SEGMENT_SCENARIOS = [1, 2, 3, 5, 7, 9]

TMIN = 280.0
TMAX = 460.0
N_GRID = 400
T_GRID = np.linspace(TMIN, TMAX, N_GRID)

DEVICE = "cpu"

INPUT_DIM = 1
HIDDEN_DIM = 32
HIDDEN_NUM = 2
Z0_DIM = 3

# Which archived run to use for each scenario/model.
# Use 1 if you want run01. Use None if you want the newest archived run.
RUN_INDEX_TO_USE = 1

# Choose which violation to plot
# "original_nonlinear"   -> |eq1(T, predicted Ca, predicted Cb)|
# "piecewise_linear"     -> |A x + B y - b|
VIOLATION_KIND = "original_nonlinear"

USE_LOG_Y = True

ARCHIVE_ROOT = "models_archive"

OUT_CSV_NN = f"violation_vs_temperature_NN_{VIOLATION_KIND}.csv"
OUT_CSV_KKT = f"violation_vs_temperature_KKThPINN_{VIOLATION_KIND}.csv"

OUT_PLOT_NN = f"violation_vs_temperature_NN_{VIOLATION_KIND}.png"
OUT_PLOT_KKT = f"violation_vs_temperature_KKThPINN_{VIOLATION_KIND}.png"


# ============================================================
# HELPERS
# ============================================================
def custom_sigmoid(x, transition_point, steepness):
    transition_width = 100.0 / steepness
    w = (x - transition_point) / transition_width
    return torch.sigmoid(w)

def hard_window_1d(x, L, U, is_last=False):
    if is_last:
        return ((x >= L) & (x <= U)).to(x.dtype)
    else:
        return ((x >= L) & (x < U)).to(x.dtype)

def get_masks_1d(X, T_edges, steepT=8e5, hard=False):
    nT = len(T_edges) - 1

    if nT == 1:
        return torch.ones((X.shape[0], 1), dtype=X.dtype, device=X.device)

    transition_points = T_edges[1:-1]
    masks = []

    for i in range(nT):
        if hard:
            mask = hard_window_1d(
                X,
                T_edges[i],
                T_edges[i + 1],
                is_last=(i == nT - 1)
            )
        else:
            if i == 0:
                mask = 1.0 - custom_sigmoid(X, transition_points[0], steepT)
            elif i == nT - 1:
                mask = custom_sigmoid(X, transition_points[-1], steepT)
            else:
                mask = (
                    custom_sigmoid(X, transition_points[i - 1], steepT) *
                    (1.0 - custom_sigmoid(X, transition_points[i], steepT))
                )
        masks.append(mask)

    return torch.cat(masks, dim=1)

def parse_run_index(folder_name: str):
    m = re.search(r"_run(\d+)_", folder_name)
    if m:
        return int(m.group(1))
    return None


def find_archives_by_model_and_scenario(model_name: str):
    """
    Return dict:
        scenario_nseg -> archive_dir

    We infer nseg from the number of rows in archived code_snapshot/ABb_matrices.csv.
    """
    pattern = os.path.join(ARCHIVE_ROOT, "*", f"*_{model_name}")
    candidates = [d for d in glob.glob(pattern) if os.path.isdir(d)]

    selected = {}

    for d in candidates:
        ab_path = os.path.join(d, "code_snapshot", "ABb_matrices.csv")
        model_state_path = os.path.join(d, "model_state.pth")
        scaler_path = os.path.join(d, "scaler.pkl")

        if not (os.path.exists(ab_path) and os.path.exists(model_state_path) and os.path.exists(scaler_path)):
            continue

        folder_name = os.path.basename(d)
        run_idx = parse_run_index(folder_name)

        if RUN_INDEX_TO_USE is not None and run_idx != RUN_INDEX_TO_USE:
            continue

        try:
            nseg = len(pd.read_csv(ab_path))
        except Exception:
            continue

        if nseg not in SEGMENT_SCENARIOS:
            continue

        if nseg not in selected:
            selected[nseg] = d
        else:
            # keep the newest archive if there are multiple
            if os.path.getmtime(d) > os.path.getmtime(selected[nseg]):
                selected[nseg] = d

    return selected


def load_scaler(scaler_path):
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


def load_raw_ab_lists(ab_csv_path):
    df = pd.read_csv(ab_csv_path).sort_values("region_id").reset_index(drop=True)

    A_list = []
    B_list = []
    b_list = []

    for _, row in df.iterrows():
        A_list.append(torch.tensor([[row["A_T"]]], dtype=torch.float64, device=DEVICE))
        B_list.append(torch.tensor([[row["B_Ca"], row["B_Cb"], row["B_Cc"]]], dtype=torch.float64, device=DEVICE))
        b_list.append(torch.tensor([row["b"]], dtype=torch.float64, device=DEVICE))

    return A_list, B_list, b_list


def scale_ab_lists(A_list_raw, B_list_raw, b_list_raw, scaler):
    xscale = torch.tensor([scaler.scale_[0]], dtype=torch.float64, device=DEVICE)
    zscale = torch.tensor(scaler.scale_[1:4], dtype=torch.float64, device=DEVICE)

    A_list_scaled = []
    B_list_scaled = []
    b_list_scaled = []

    for A, B, b in zip(A_list_raw, B_list_raw, b_list_raw):
        A_scaled = A * xscale.view(1, -1)
        B_scaled = B * zscale.view(1, -1)
        b_scaled = b.clone()

        A_list_scaled.append(A_scaled)
        B_list_scaled.append(B_scaled)
        b_list_scaled.append(b_scaled)

    return A_list_scaled, B_list_scaled, b_list_scaled


def make_scaled_temperature_input(T_raw, scaler):
    """
    Build scaled X for the 1D input T using the same scaler convention as training.
    """
    dummy = np.zeros((len(T_raw), 4), dtype=np.float64)
    dummy[:, 0] = T_raw
    X_scaled = scaler.transform(dummy)[:, :1]
    return X_scaled


def build_model(model_name, checkpoint_path, scaler, ab_csv_path, nseg):
    A_raw, B_raw, b_raw = load_raw_ab_lists(ab_csv_path)
    A_scaled, B_scaled, b_scaled = scale_ab_lists(A_raw, B_raw, b_raw, scaler)

    T_edges_raw = np.linspace(TMIN, TMAX, nseg + 1, dtype=np.float64)
    T_edges_scaled = T_edges_raw / scaler.scale_[0]

    if model_name == "NN":
        model = NN(INPUT_DIM, HIDDEN_DIM, HIDDEN_NUM, Z0_DIM).double().to(DEVICE)
    elif model_name == "KKThPINN":
        model = NNOPT(
            INPUT_DIM, HIDDEN_DIM, HIDDEN_NUM, Z0_DIM,
            A_scaled, B_scaled, b_scaled, T_edges_scaled
        ).double().to(DEVICE)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return model, A_scaled, B_scaled, b_scaled, T_edges_scaled


# def safe_predict(model, X, steepness=800000):
#     """
#     Safe forward pass that also handles nseg=1 without crashing.
#     """
#     if isinstance(model, NN):
#         with torch.no_grad():
#             return model(X)

#     # NNOPT / KKThPINN
#     with torch.no_grad():
#         x0 = X
#         for layer in model.layers[:-1]:
#             x0 = torch.relu(layer(x0))
#         z0 = model.layers[-1](x0)

#         transition_points = model.T_edges[1:-1]
#         fixed_outputs = []

#         for i, (fc1, fc2) in enumerate(zip(model.fc1_list, model.fc2_list)):
#             z_fixed = fc1(z0) + fc2(X)

#             if len(model.fc1_list) == 1 or transition_points.numel() == 0:
#                 mask = 1.0
#             else:
#                 if i == 0:
#                     mask = 1.0 - model.custom_sigmoid(X, transition_points[0], steepness)
#                 elif i == len(model.fc1_list) - 1:
#                     mask = model.custom_sigmoid(X, transition_points[-1], steepness)
#                 else:
#                     mask = (
#                         model.custom_sigmoid(X, transition_points[i - 1], steepness)
#                         * (1.0 - model.custom_sigmoid(X, transition_points[i], steepness))
#                     )

#             fixed_outputs.append(z_fixed * mask)

#         return sum(fixed_outputs)
def safe_predict(model, X, steepness=800000, hard=False):
    with torch.no_grad():
        if isinstance(model, NN):
            return model(X)
        return model(X, steepness=steepness, hard=hard)

def inverse_predictions(T_raw, pred_scaled_np, scaler):
    """
    pred_scaled_np: shape (N,3) for [Ca, Cb, Cc] in scaled space
    Returns raw T, Ca, Cb, Cc
    """
    N = len(T_raw)
    pack = np.zeros((N, 4), dtype=np.float64)

    X_scaled = make_scaled_temperature_input(T_raw, scaler)
    pack[:, 0] = X_scaled[:, 0]
    pack[:, 1:4] = pred_scaled_np

    inv = scaler.inverse_transform(pack)
    return inv[:, 0], inv[:, 1], inv[:, 2], inv[:, 3]


def compute_original_nonlinear_violation(T_raw, pred_scaled_np, scaler):
    _, Ca_raw, Cb_raw, Cc_raw = inverse_predictions(T_raw, pred_scaled_np, scaler)

    Cao = 1.0
    Cbo = 2.0
    Cco = 0.0
    V = 10.0
    Q = 1.0
    tau = V / Q

    Afo = 1e13
    Eaf = 90000.0
    Aro = 1e11
    Ear = 80000.0
    R = 8.314

    T = T_raw.astype(np.float64)
    Ca = Ca_raw.astype(np.float64)
    Cb = Cb_raw.astype(np.float64)

    kf = Afo * np.exp(-Eaf / (R * T))
    kr = Aro * np.exp(-Ear / (R * T))

    eq1 = (
        (Cao - Ca)
        - kf * Ca * (Cb ** 2) * tau
        + kr * (Cao - Ca + Cbo - Cb + Cco) * tau
    )

    return np.abs(eq1)


def compute_piecewise_linear_violation(
    X_scaled_t, pred_scaled_t, A_list, B_list, b_list, T_edges_scaled,
    hard=False, steepT=8e5
):
    T_edges_scaled = torch.as_tensor(T_edges_scaled, dtype=X_scaled_t.dtype, device=X_scaled_t.device)
    masks = get_masks_1d(X_scaled_t, T_edges_scaled, steepT=steepT, hard=hard)

    violations = []
    for i, (A, B, b) in enumerate(zip(A_list, B_list, b_list)):
        v = (X_scaled_t @ A.T + pred_scaled_t @ B.T - b)      # (N,1)
        v = v * masks[:, i:i+1]
        violations.append(v)

    return torch.abs(torch.cat(violations, dim=1)).sum(dim=1).detach().cpu().numpy()


def compute_violation_curve(model_name, archive_dir):
    checkpoint_path = os.path.join(archive_dir, "model_state.pth")
    scaler_path = os.path.join(archive_dir, "scaler.pkl")
    ab_csv_path = os.path.join(archive_dir, "code_snapshot", "ABb_matrices.csv")

    scaler = load_scaler(scaler_path)
    nseg = len(pd.read_csv(ab_csv_path))

    model, A_scaled, B_scaled, b_scaled, T_edges_scaled = build_model(
        model_name, checkpoint_path, scaler, ab_csv_path, nseg
    )

    X_scaled_np = make_scaled_temperature_input(T_GRID, scaler)
    X_scaled_t = torch.tensor(X_scaled_np, dtype=torch.float64, device=DEVICE)

    # pred_scaled_t = safe_predict(model, X_scaled_t)
    pred_scaled_t = safe_predict(
    model,
    X_scaled_t,
    hard=(model_name == "KKThPINN")
    )
    pred_scaled_np = pred_scaled_t.detach().cpu().numpy()

    if VIOLATION_KIND == "original_nonlinear":
        violation = compute_original_nonlinear_violation(T_GRID, pred_scaled_np, scaler)
    elif VIOLATION_KIND == "piecewise_linear":
        violation = compute_piecewise_linear_violation(
            X_scaled_t, pred_scaled_t, A_scaled, B_scaled, b_scaled, T_edges_scaled, hard=(model_name == "KKThPINN")
        )
    else:
        raise ValueError("VIOLATION_KIND must be 'original_nonlinear' or 'piecewise_linear'")

    return violation, nseg


def plot_for_model(model_name, out_csv, out_plot):
    archive_map = find_archives_by_model_and_scenario(model_name)

    if not archive_map:
        print(f"[WARNING] No archived models found for {model_name}")
        return

    out_df = pd.DataFrame({"Temperature (T)": T_GRID})

    plt.figure(figsize=(9, 5))

    for nseg in SEGMENT_SCENARIOS:
        if nseg not in archive_map:
            print(f"[WARNING] Missing archive for {model_name}, nseg={nseg}")
            continue

        archive_dir = archive_map[nseg]
        violation, _ = compute_violation_curve(model_name, archive_dir)

        out_df[f"viol_nseg_{nseg}"] = violation
        plt.plot(T_GRID, violation, linewidth=2, label=f"{nseg} segments")

        print(
            f"{model_name:10s} | nseg={nseg:2d} | "
            f"mean={np.nanmean(violation):.6e} | max={np.nanmax(violation):.6e}"
        )

    out_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    plt.xlabel("Input temperature (K)")
    if VIOLATION_KIND == "original_nonlinear":
        plt.ylabel("Original nonlinear violation")
        plt.title(f"{model_name}: original nonlinear violation vs temperature")
    else:
        plt.ylabel(r"Piecewise linear violation $|A x + B y - b|$")
        plt.title(f"{model_name}: piecewise linear violation vs temperature")

    if USE_LOG_Y:
        plt.yscale("log")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_plot, dpi=300)
    plt.show()
    print(f"Saved: {out_plot}")


# ============================================================
# MAIN
# ============================================================
def main():
    plot_for_model("NN", OUT_CSV_NN, OUT_PLOT_NN)
    plot_for_model("KKThPINN", OUT_CSV_KKT, OUT_PLOT_KKT)


if __name__ == "__main__":
    main()