import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Settings
# ============================================================

SEGMENT_SCENARIOS = [1, 2, 3, 5, 7, 9, 11, 30, 90]

# If True: NN is shown as one pooled flat baseline across all scenarios.
# If False: NN mean/std are computed separately for each scenario file.
POOL_NN_BASELINE = False

OUT_DIR = "plots_mean_std"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# Helper functions
# ============================================================

def mean_std(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]

    if len(vals) == 0:
        return np.nan, np.nan

    if len(vals) == 1:
        return float(vals[0]), 0.0

    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def safe_read_csv(path):
    if not os.path.exists(path):
        print(f"[WARNING] Missing file: {path}")
        return None
    return pd.read_csv(path)


def get_col(df, col):
    if df is None or col not in df.columns:
        print(f"[WARNING] Missing column: {col}")
        return np.array([np.nan])
    return df[col].dropna().to_numpy(dtype=float)


# ============================================================
# Main experiment plots: mean ± standard deviation
# ============================================================

PLOTS = [
    {
        "nn_col": "NN_Experiment_RMSE",
        "kkt_col": "KKThPINN_Experiment_RMSE",
        "ylabel": "RMSE",
        "save": "rmse_vs_cao_regions_mean_std.png",
        "logy": False,
    },
    {
        "nn_col": "NN_Experiment_VIOL",
        "kkt_col": "KKThPINN_Experiment_VIOL",
        "ylabel": "Piecewise-linear constraint violation",
        "save": "pl_violation_vs_cao_regions_mean_std.png",
        "logy": True,
    },
    {
        "nn_col": "NN_Experiment_VIOL_NL",
        "kkt_col": "KKThPINN_Experiment_VIOL_NL",
        "ylabel": "Original nonlinear violation",
        "save": "nonlinear_violation_vs_cao_regions_mean_std.png",
        "logy": True,
    },
    {
        "nn_col": "NN_Experiment_Time_sec",
        "kkt_col": "KKThPINN_Experiment_Time_sec",
        "ylabel": "Experiment time (sec)",
        "save": "experiment_time_vs_cao_regions_mean_std.png",
        "logy": False,
    },
]

for cfg in PLOTS:
    x_vals = []
    nn_mean = []
    nn_std = []
    kkt_mean = []
    kkt_std = []

    nn_all = []

    for nC in SEGMENT_SCENARIOS:
        path = f"experiment_epoch_errors_nseg_{nC}.csv"
        df = safe_read_csv(path)

        nn_vals = get_col(df, cfg["nn_col"])
        kkt_vals = get_col(df, cfg["kkt_col"])

        nn_all.extend(nn_vals.tolist())

        m_nn, s_nn = mean_std(nn_vals)
        m_kkt, s_kkt = mean_std(kkt_vals)

        x_vals.append(nC)
        nn_mean.append(m_nn)
        nn_std.append(s_nn)
        kkt_mean.append(m_kkt)
        kkt_std.append(s_kkt)

    x_vals = np.array(x_vals, dtype=float)
    nn_mean = np.array(nn_mean, dtype=float)
    nn_std = np.array(nn_std, dtype=float)
    kkt_mean = np.array(kkt_mean, dtype=float)
    kkt_std = np.array(kkt_std, dtype=float)

    # Optional pooled NN baseline
    if POOL_NN_BASELINE:
        pooled_mean, pooled_std = mean_std(nn_all)
        nn_mean_plot = np.full_like(x_vals, pooled_mean, dtype=float)
        nn_std_plot = np.full_like(x_vals, pooled_std, dtype=float)
    else:
        nn_mean_plot = nn_mean
        nn_std_plot = nn_std

    plt.figure(figsize=(9, 6))

    # NN
    plt.plot(
        x_vals,
        nn_mean_plot,
        marker="o",
        linewidth=2.5,
        label="NN"
    )
    plt.fill_between(
        x_vals,
        nn_mean_plot - nn_std_plot,
        nn_mean_plot + nn_std_plot,
        alpha=0.2,
        label="NN ± std"
    )

    # KKT-hPINN
    plt.plot(
        x_vals,
        kkt_mean,
        marker="o",
        linewidth=2.5,
        label="KKT-hPINN"
    )
    plt.fill_between(
        x_vals,
        kkt_mean - kkt_std,
        kkt_mean + kkt_std,
        alpha=0.2,
        label="KKT-hPINN ± std"
    )

    plt.xlabel("Number of Cao regions")
    plt.ylabel(cfg["ylabel"])
    plt.grid(True, alpha=0.3)

    if cfg["logy"]:
        plt.yscale("log")

    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(OUT_DIR, cfg["save"])
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


# ============================================================
# Optional training-time plot, if training CSVs exist
# ============================================================

TRAINING_PLOT = {
    "nn_col": "NN_Training_Time_sec",
    "kkt_col": "KKThPINN_Training_Time_sec",
    "ylabel": "Training time (sec)",
    "save": "training_time_vs_cao_regions_mean_std.png",
    "logy": False,
}

x_vals = []
nn_mean = []
nn_std = []
kkt_mean = []
kkt_std = []

has_training_data = False

for nC in SEGMENT_SCENARIOS:
    path = f"training_epoch_errors_nseg_{nC}.csv"
    df = safe_read_csv(path)

    if df is None:
        continue

    if TRAINING_PLOT["nn_col"] in df.columns and TRAINING_PLOT["kkt_col"] in df.columns:
        has_training_data = True

        nn_vals = get_col(df, TRAINING_PLOT["nn_col"])
        kkt_vals = get_col(df, TRAINING_PLOT["kkt_col"])

        m_nn, s_nn = mean_std(nn_vals)
        m_kkt, s_kkt = mean_std(kkt_vals)

        x_vals.append(nC)
        nn_mean.append(m_nn)
        nn_std.append(s_nn)
        kkt_mean.append(m_kkt)
        kkt_std.append(s_kkt)

if has_training_data:
    x_vals = np.array(x_vals, dtype=float)
    nn_mean = np.array(nn_mean, dtype=float)
    nn_std = np.array(nn_std, dtype=float)
    kkt_mean = np.array(kkt_mean, dtype=float)
    kkt_std = np.array(kkt_std, dtype=float)

    plt.figure(figsize=(9, 6))

    plt.plot(x_vals, nn_mean, marker="o", linewidth=2.5, label="NN")
    plt.fill_between(x_vals, nn_mean - nn_std, nn_mean + nn_std, alpha=0.2, label="NN ± std")

    plt.plot(x_vals, kkt_mean, marker="o", linewidth=2.5, label="KKT-hPINN")
    plt.fill_between(x_vals, kkt_mean - kkt_std, kkt_mean + kkt_std, alpha=0.2, label="KKT-hPINN ± std")

    plt.xlabel("Number of Cao regions")
    plt.ylabel(TRAINING_PLOT["ylabel"])
    plt.grid(True, alpha=0.3)

    if TRAINING_PLOT["logy"]:
        plt.yscale("log")

    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(OUT_DIR, TRAINING_PLOT["save"])
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


# ============================================================
# Deterministic diagnostics:
# linearization accuracy and projection check
# These do not have std unless you repeat them with different data.
# ============================================================

diagnostic_rows = []

for nC in SEGMENT_SCENARIOS:
    row = {"nC_regions": nC}

    lin_path = f"linearization_accuracy_summary_nseg_{nC}.csv"
    proj_path = f"projection_check_nseg_{nC}.csv"

    lin_df = safe_read_csv(lin_path)
    if lin_df is not None:
        if "mean_abs_linearization_error" in lin_df.columns:
            row["mean_abs_linearization_error"] = lin_df["mean_abs_linearization_error"].mean()
        if "rmse_linearization_error" in lin_df.columns:
            row["rmse_linearization_error"] = lin_df["rmse_linearization_error"].mean()
        if "max_abs_linearization_error" in lin_df.columns:
            row["max_abs_linearization_error"] = lin_df["max_abs_linearization_error"].max()

    proj_df = safe_read_csv(proj_path)
    if proj_df is not None:
        for col in [
            "projection_check_output_MAE_scaled",
            "projection_check_PL_violation",
            "projection_check_original_nonlinear_violation",
        ]:
            if col in proj_df.columns:
                row[col] = proj_df[col].iloc[0]

    diagnostic_rows.append(row)

diag_df = pd.DataFrame(diagnostic_rows)
diag_csv = os.path.join(OUT_DIR, "diagnostics_by_cao_regions.csv")
diag_df.to_csv(diag_csv, index=False)
print(f"Saved: {diag_csv}")


DIAGNOSTIC_PLOTS = [
    {
        "col": "mean_abs_linearization_error",
        "ylabel": "Mean absolute linearization error",
        "save": "linearization_error_vs_cao_regions.png",
        "logy": True,
    },
    {
        "col": "rmse_linearization_error",
        "ylabel": "RMSE linearization error",
        "save": "linearization_rmse_vs_cao_regions.png",
        "logy": True,
    },
    {
        "col": "projection_check_output_MAE_scaled",
        "ylabel": "Projection-check output MAE, scaled",
        "save": "projection_check_output_mae_vs_cao_regions.png",
        "logy": True,
    },
    {
        "col": "projection_check_PL_violation",
        "ylabel": "Projection-check PL violation",
        "save": "projection_check_pl_violation_vs_cao_regions.png",
        "logy": True,
    },
    {
        "col": "projection_check_original_nonlinear_violation",
        "ylabel": "Projection-check original nonlinear violation",
        "save": "projection_check_nonlinear_violation_vs_cao_regions.png",
        "logy": True,
    },
]

for cfg in DIAGNOSTIC_PLOTS:
    col = cfg["col"]

    if col not in diag_df.columns:
        print(f"[WARNING] Skipping diagnostic plot. Missing column: {col}")
        continue

    plot_df = diag_df[["nC_regions", col]].dropna()

    if len(plot_df) == 0:
        print(f"[WARNING] Skipping diagnostic plot. No data for: {col}")
        continue

    plt.figure(figsize=(9, 6))
    plt.plot(
        plot_df["nC_regions"].to_numpy(),
        plot_df[col].to_numpy(),
        marker="o",
        linewidth=2.5,
    )

    plt.xlabel("Number of Cao regions")
    plt.ylabel(cfg["ylabel"])
    plt.grid(True, alpha=0.3)

    if cfg["logy"]:
        plt.yscale("log")

    plt.tight_layout()

    save_path = os.path.join(OUT_DIR, cfg["save"])
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


# ============================================================
# Summary table for checking variability
# ============================================================

summary_rows = []

for nC in SEGMENT_SCENARIOS:
    path = f"experiment_epoch_errors_nseg_{nC}.csv"
    df = safe_read_csv(path)

    if df is None:
        continue

    for col in [
        "NN_Experiment_RMSE",
        "KKThPINN_Experiment_RMSE",
        "NN_Experiment_VIOL",
        "KKThPINN_Experiment_VIOL",
        "NN_Experiment_VIOL_NL",
        "KKThPINN_Experiment_VIOL_NL",
        "NN_Experiment_Time_sec",
        "KKThPINN_Experiment_Time_sec",
    ]:
        if col not in df.columns:
            continue

        vals = df[col].dropna().to_numpy(dtype=float)
        m, s = mean_std(vals)

        summary_rows.append({
            "nC_regions": nC,
            "metric": col,
            "mean": m,
            "std": s,
            "min": np.nanmin(vals) if len(vals) else np.nan,
            "max": np.nanmax(vals) if len(vals) else np.nan,
            "n_runs": len(vals),
        })

summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(OUT_DIR, "mean_std_summary_table.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"Saved: {summary_csv}")

print("\nDone. All mean ± std plots were saved in:")
print(OUT_DIR)