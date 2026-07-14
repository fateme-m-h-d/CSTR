import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.stats import sem, t


# ============================================================
# 1D data-efficiency plotting script
#
# Fixed:
#   nC_regions = 30
#
# Vary:
#   n_inner_per_region
#
# Reads:
#   data_ninner_*.csv
#   experiment_epoch_errors_ninner_*.csv
#   training_epoch_errors_ninner_*.csv
#   scenario_diagnostics.csv
#
# Produces plots versus actual number of sample points.
# ============================================================

INNER_SCENARIOS = [0, 1, 2, 5, 10, 15, 20, 25]

BASE_DIR = Path.cwd()
OUT_DIR = BASE_DIR / "plots_data_efficiency_1d"
OUT_DIR.mkdir(exist_ok=True)


# ============================================================
# Helper functions
# ============================================================

def mean_ci_halfwidth(values, confidence=0.95):
    """
    Return mean and 95% confidence interval half-width.
    """
    values = np.asarray(pd.Series(values).dropna().to_numpy(), dtype=float)
    n = len(values)

    if n == 0:
        return np.nan, np.nan

    mean = float(np.mean(values))

    if n < 2:
        return mean, 0.0

    half_width = float(sem(values) * t.ppf((1.0 + confidence) / 2.0, n - 1))
    return mean, half_width


def safe_read_csv(path):
    path = Path(path)

    if not path.exists():
        print(f"Warning: missing {path.name}; skipping.")
        return None

    return pd.read_csv(path)


def get_actual_num_samples(n_inner):
    """
    Read the actual number of sample points from data_ninner_*.csv.
    This is safer than using the theoretical formula.
    """
    data_path = BASE_DIR / f"data_ninner_{n_inner}.csv"
    df = safe_read_csv(data_path)

    if df is None:
        return np.nan

    return int(len(df))


def positive_lower_for_log(y, ci, eps=1e-18):
    """
    For log-scale plots, the lower CI band cannot be <= 0.
    """
    lower = np.asarray(y, dtype=float) - np.asarray(ci, dtype=float)
    return np.maximum(lower, eps)


def save_figure(save_name):
    """
    Save both PNG and PDF versions.
    """
    png_path = OUT_DIR / f"{save_name}.png"
    pdf_path = OUT_DIR / f"{save_name}.pdf"

    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


# ============================================================
# Repeated experiment summary
# ============================================================

def load_experiment_summary():
    """
    Loads experiment_epoch_errors_ninner_*.csv files and computes
    mean ± 95% CI for NN and KKT-hPINN.

    No NN pooling is used here because the training data size changes.
    """

    rows = []

    metrics = [
        {
            "metric_name": "RMSE",
            "nn_col": "NN_Experiment_RMSE",
            "kkt_col": "KKThPINN_Experiment_RMSE",
        },
        {
            "metric_name": "PL_Violation",
            "nn_col": "NN_Experiment_VIOL",
            "kkt_col": "KKThPINN_Experiment_VIOL",
        },
        {
            "metric_name": "Original_Nonlinear_Violation",
            "nn_col": "NN_Experiment_VIOL_NL",
            "kkt_col": "KKThPINN_Experiment_VIOL_NL",
        },
        {
            "metric_name": "Experiment_Time_sec",
            "nn_col": "NN_Experiment_Time_sec",
            "kkt_col": "KKThPINN_Experiment_Time_sec",
        },
    ]

    for n_inner in INNER_SCENARIOS:
        exp_path = BASE_DIR / f"experiment_epoch_errors_ninner_{n_inner}.csv"
        exp_df = safe_read_csv(exp_path)

        if exp_df is None:
            continue

        num_sample_points = get_actual_num_samples(n_inner)

        row = {
            "n_inner_per_region": n_inner,
            "num_sample_points": num_sample_points,
        }

        for metric in metrics:
            metric_name = metric["metric_name"]
            nn_col = metric["nn_col"]
            kkt_col = metric["kkt_col"]

            if nn_col in exp_df.columns:
                nn_vals = exp_df[nn_col].dropna().to_numpy(dtype=float)
            else:
                print(f"Warning: missing column {nn_col} in {exp_path.name}")
                nn_vals = np.array([])

            if kkt_col in exp_df.columns:
                kkt_vals = exp_df[kkt_col].dropna().to_numpy(dtype=float)
            else:
                print(f"Warning: missing column {kkt_col} in {exp_path.name}")
                kkt_vals = np.array([])

            nn_mean, nn_ci = mean_ci_halfwidth(nn_vals)
            kkt_mean, kkt_ci = mean_ci_halfwidth(kkt_vals)

            row[f"NN_{metric_name}_mean"] = nn_mean
            row[f"NN_{metric_name}_ci95"] = nn_ci
            row[f"NN_{metric_name}_n_runs"] = len(nn_vals)

            row[f"KKThPINN_{metric_name}_mean"] = kkt_mean
            row[f"KKThPINN_{metric_name}_ci95"] = kkt_ci
            row[f"KKThPINN_{metric_name}_n_runs"] = len(kkt_vals)

        rows.append(row)

    summary = pd.DataFrame(rows)

    if not summary.empty:
        summary = summary.sort_values("num_sample_points").reset_index(drop=True)

    out_csv = OUT_DIR / "metric_summary_by_samples.csv"
    summary.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    return summary


def load_training_summary():
    """
    Loads training_epoch_errors_ninner_*.csv files and computes
    mean ± 95% CI for training time and final training error.
    """

    rows = []

    metrics = [
        {
            "metric_name": "Training_Time_sec",
            "nn_col": "NN_Training_Time_sec",
            "kkt_col": "KKThPINN_Training_Time_sec",
        },
        {
            "metric_name": "Training_Error",
            "nn_col": "NN_Training_Error",
            "kkt_col": "KKThPINN_Training_Error",
        },
    ]

    for n_inner in INNER_SCENARIOS:
        train_path = BASE_DIR / f"training_epoch_errors_ninner_{n_inner}.csv"
        train_df = safe_read_csv(train_path)

        if train_df is None:
            continue

        num_sample_points = get_actual_num_samples(n_inner)

        row = {
            "n_inner_per_region": n_inner,
            "num_sample_points": num_sample_points,
        }

        for metric in metrics:
            metric_name = metric["metric_name"]
            nn_col = metric["nn_col"]
            kkt_col = metric["kkt_col"]

            if nn_col in train_df.columns:
                nn_vals = train_df[nn_col].dropna().to_numpy(dtype=float)
            else:
                print(f"Warning: missing column {nn_col} in {train_path.name}")
                nn_vals = np.array([])

            if kkt_col in train_df.columns:
                kkt_vals = train_df[kkt_col].dropna().to_numpy(dtype=float)
            else:
                print(f"Warning: missing column {kkt_col} in {train_path.name}")
                kkt_vals = np.array([])

            nn_mean, nn_ci = mean_ci_halfwidth(nn_vals)
            kkt_mean, kkt_ci = mean_ci_halfwidth(kkt_vals)

            row[f"NN_{metric_name}_mean"] = nn_mean
            row[f"NN_{metric_name}_ci95"] = nn_ci
            row[f"NN_{metric_name}_n_runs"] = len(nn_vals)

            row[f"KKThPINN_{metric_name}_mean"] = kkt_mean
            row[f"KKThPINN_{metric_name}_ci95"] = kkt_ci
            row[f"KKThPINN_{metric_name}_n_runs"] = len(kkt_vals)

        rows.append(row)

    summary = pd.DataFrame(rows)

    if not summary.empty:
        summary = summary.sort_values("num_sample_points").reset_index(drop=True)

    out_csv = OUT_DIR / "training_summary_by_samples.csv"
    summary.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    return summary


# ============================================================
# Plotting repeated metrics
# ============================================================

def plot_repeated_metric(summary, metric_name, ylabel, save_name, logy=False):
    """
    Plot NN and KKT-hPINN mean ± 95% CI versus sample points.
    """

    if summary is None or summary.empty:
        print(f"Skipping {save_name}: summary is empty.")
        return

    required_cols = [
        "num_sample_points",
        f"NN_{metric_name}_mean",
        f"NN_{metric_name}_ci95",
        f"KKThPINN_{metric_name}_mean",
        f"KKThPINN_{metric_name}_ci95",
    ]

    for col in required_cols:
        if col not in summary.columns:
            print(f"Skipping {save_name}: missing column {col}.")
            return

    x = summary["num_sample_points"].to_numpy(dtype=float)

    nn_mean = summary[f"NN_{metric_name}_mean"].to_numpy(dtype=float)
    nn_ci = summary[f"NN_{metric_name}_ci95"].to_numpy(dtype=float)

    kkt_mean = summary[f"KKThPINN_{metric_name}_mean"].to_numpy(dtype=float)
    kkt_ci = summary[f"KKThPINN_{metric_name}_ci95"].to_numpy(dtype=float)

    plt.figure(figsize=(9, 6))

    # NN
    plt.plot(
        x,
        nn_mean,
        marker="o",
        linewidth=2.5,
        label="NN",
    )

    if logy:
        nn_lower = positive_lower_for_log(nn_mean, nn_ci)
    else:
        nn_lower = nn_mean - nn_ci

    plt.fill_between(
        x,
        nn_lower,
        nn_mean + nn_ci,
        alpha=0.2,
        label="NN 95% CI",
    )

    # KKT-hPINN
    plt.plot(
        x,
        kkt_mean,
        marker="o",
        linewidth=2.5,
        label="KKT-hPINN",
    )

    if logy:
        kkt_lower = positive_lower_for_log(kkt_mean, kkt_ci)
    else:
        kkt_lower = kkt_mean - kkt_ci

    plt.fill_between(
        x,
        kkt_lower,
        kkt_mean + kkt_ci,
        alpha=0.2,
        label="KKT-hPINN 95% CI",
    )

    plt.xlabel("Number of sample points")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)

    if logy:
        plt.yscale("log")

    plt.legend()
    plt.tight_layout()

    save_figure(save_name)


# ============================================================
# Deterministic diagnostics
# ============================================================

def load_diagnostics():
    """
    Loads scenario_diagnostics.csv generated by run_all.py.
    These values are deterministic per scenario, so no CI is plotted.
    """

    diag_path = BASE_DIR / "scenario_diagnostics.csv"

    if not diag_path.exists():
        print("Warning: scenario_diagnostics.csv not found. Diagnostic plots will be skipped.")
        return pd.DataFrame()

    diagnostics = pd.read_csv(diag_path)

    if "num_sample_points" not in diagnostics.columns:
        print("Warning: scenario_diagnostics.csv does not have num_sample_points.")
        return pd.DataFrame()

    diagnostics = diagnostics.sort_values("num_sample_points").reset_index(drop=True)

    out_csv = OUT_DIR / "diagnostics_by_samples.csv"
    diagnostics.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    return diagnostics


def plot_diagnostic(diagnostics, col, ylabel, save_name, logy=False):
    """
    Plot deterministic diagnostic metric versus number of sample points.
    """

    if diagnostics is None or diagnostics.empty:
        print(f"Skipping {save_name}: diagnostics are empty.")
        return

    if col not in diagnostics.columns:
        print(f"Skipping {save_name}: column {col} not available.")
        return

    plot_df = diagnostics[["num_sample_points", col]].dropna()

    if plot_df.empty:
        print(f"Skipping {save_name}: no valid values for {col}.")
        return

    x = plot_df["num_sample_points"].to_numpy(dtype=float)
    y = plot_df[col].to_numpy(dtype=float)

    plt.figure(figsize=(9, 6))
    plt.plot(
        x,
        y,
        marker="o",
        linewidth=2.5,
    )

    plt.xlabel("Number of sample points")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)

    if logy:
        plt.yscale("log")

    plt.tight_layout()
    save_figure(save_name)


# ============================================================
# Main
# ============================================================

def main():
    # --------------------------------------------------------
    # Repeated experiment plots
    # --------------------------------------------------------
    experiment_summary = load_experiment_summary()

    plot_repeated_metric(
        experiment_summary,
        metric_name="RMSE",
        ylabel="RMSE",
        save_name="rmse_vs_samples_1d_data_efficiency",
        logy=False,
    )

    plot_repeated_metric(
        experiment_summary,
        metric_name="PL_Violation",
        ylabel="Piecewise-linear constraint violation",
        save_name="pl_violation_vs_samples_1d_data_efficiency",
        logy=True,
    )

    plot_repeated_metric(
        experiment_summary,
        metric_name="Original_Nonlinear_Violation",
        ylabel="Original nonlinear violation",
        save_name="nonlinear_violation_vs_samples_1d_data_efficiency",
        logy=True,
    )

    plot_repeated_metric(
        experiment_summary,
        metric_name="Experiment_Time_sec",
        ylabel="Experiment time (sec)",
        save_name="experiment_time_vs_samples_1d_data_efficiency",
        logy=False,
    )

    # --------------------------------------------------------
    # Training plots
    # --------------------------------------------------------
    training_summary = load_training_summary()

    plot_repeated_metric(
        training_summary,
        metric_name="Training_Time_sec",
        ylabel="Training time (sec)",
        save_name="training_time_vs_samples_1d_data_efficiency",
        logy=False,
    )

    plot_repeated_metric(
        training_summary,
        metric_name="Training_Error",
        ylabel="Final training error",
        save_name="training_error_vs_samples_1d_data_efficiency",
        logy=True,
    )

    # --------------------------------------------------------
    # Diagnostic plots
    # --------------------------------------------------------
    diagnostics = load_diagnostics()

    # Linearization accuracy diagnostics
    plot_diagnostic(
        diagnostics,
        col="linearization_mean_abs_error",
        ylabel="Mean absolute linearization error",
        save_name="linearization_mean_abs_error_vs_samples_1d_data_efficiency",
        logy=True,
    )

    plot_diagnostic(
        diagnostics,
        col="linearization_rmse_error",
        ylabel="RMSE linearization error",
        save_name="linearization_rmse_error_vs_samples_1d_data_efficiency",
        logy=True,
    )

    plot_diagnostic(
        diagnostics,
        col="linearization_max_abs_error",
        ylabel="Max absolute linearization error",
        save_name="linearization_max_abs_error_vs_samples_1d_data_efficiency",
        logy=True,
    )

    # Projection-only diagnostics
    plot_diagnostic(
        diagnostics,
        col="projection_check_output_MAE_scaled",
        ylabel="Projection-check output MAE (scaled)",
        save_name="projection_check_output_mae_scaled_vs_samples_1d_data_efficiency",
        logy=True,
    )

    plot_diagnostic(
        diagnostics,
        col="projection_check_PL_violation",
        ylabel="Projection-check PL violation",
        save_name="projection_check_pl_violation_vs_samples_1d_data_efficiency",
        logy=True,
    )

    plot_diagnostic(
        diagnostics,
        col="projection_check_original_nonlinear_violation",
        ylabel="Projection-check original nonlinear violation",
        save_name="projection_check_nonlinear_violation_vs_samples_1d_data_efficiency",
        logy=True,
    )

    print("\nDone. All plots and summary CSV files were saved in:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()