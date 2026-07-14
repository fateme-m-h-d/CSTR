import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem, t


# ============================================================
# 1D plotting and 95% CI script
# Reads experiment_epoch_errors_nseg_*.csv and scenario_diagnostics.csv.
# Produces plots versus number of Cao regions.
# ============================================================

SEGMENT_SCENARIOS = [1, 2, 3, 5, 11, 30, 55, 90]
BASE_DIR = Path.cwd()


def mean_ci_halfwidth(values, confidence=0.95):
    values = np.asarray(pd.Series(values).dropna().to_numpy(), dtype=float)
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    mean = float(np.mean(values))
    if n < 2:
        return mean, 0.0
    half = float(sem(values) * t.ppf((1 + confidence) / 2.0, n - 1))
    return mean, half


def load_experiment_summary():
    rows = []
    pooled = {"metric": [], "values": []}

    metrics = [
        ("RMSE", "NN_Experiment_RMSE", "KKThPINN_Experiment_RMSE", True),
        ("Experiment_Time_sec", "NN_Experiment_Time_sec", "KKThPINN_Experiment_Time_sec", True),
        ("PL_Violation", "NN_Experiment_VIOL", "KKThPINN_Experiment_VIOL", False),
        ("Original_Nonlinear_Violation", "NN_Experiment_VIOL_NL", "KKThPINN_Experiment_VIOL_NL", True),
    ]

    for nC in SEGMENT_SCENARIOS:
        csv_path = BASE_DIR / f"experiment_epoch_errors_nseg_{nC}.csv"
        if not csv_path.exists():
            print(f"Warning: missing {csv_path.name}; skipping.")
            continue

        df = pd.read_csv(csv_path)
        row = {"nC_regions": nC, "num_regions": nC}

        for metric_name, nn_col, kkt_col, pool_nn in metrics:
            nn_vals = df[nn_col].dropna().to_numpy() if nn_col in df else np.array([])
            kkt_vals = df[kkt_col].dropna().to_numpy() if kkt_col in df else np.array([])

            nn_mean, nn_ci = mean_ci_halfwidth(nn_vals)
            kkt_mean, kkt_ci = mean_ci_halfwidth(kkt_vals)

            row[f"NN_{metric_name}_mean"] = nn_mean
            row[f"NN_{metric_name}_ci95"] = nn_ci
            row[f"KKThPINN_{metric_name}_mean"] = kkt_mean
            row[f"KKThPINN_{metric_name}_ci95"] = kkt_ci

            if pool_nn and len(nn_vals) > 0:
                pooled["metric"].extend([metric_name] * len(nn_vals))
                pooled["values"].extend(nn_vals.tolist())

        rows.append(row)

    summary = pd.DataFrame(rows)

    # Add pooled NN baselines for metrics where the NN should not depend on the
    # number of projection regions. PL violation is intentionally not pooled.
    if len(pooled["values"]) > 0:
        pooled_df = pd.DataFrame(pooled)
        for metric_name, group in pooled_df.groupby("metric"):
            mean, ci = mean_ci_halfwidth(group["values"])
            summary[f"NN_{metric_name}_pooled_mean"] = mean
            summary[f"NN_{metric_name}_pooled_ci95"] = ci

    summary.to_csv("metric_summary_by_segments.csv", index=False)
    print("Saved metric_summary_by_segments.csv")
    return summary


def plot_repeated_metric(summary, metric_name, ylabel, save_name, logy=False, use_pooled_nn=False):
    if summary.empty:
        return

    x = summary["num_regions"].to_numpy(dtype=float)

    plt.figure(figsize=(9, 6))

    if use_pooled_nn and f"NN_{metric_name}_pooled_mean" in summary.columns:
        nn_mean = float(summary[f"NN_{metric_name}_pooled_mean"].iloc[0])
        nn_ci = float(summary[f"NN_{metric_name}_pooled_ci95"].iloc[0])
        plt.plot(x, np.full_like(x, nn_mean), marker="o", linewidth=2.5, label="NN")
        plt.fill_between(x, nn_mean - nn_ci, nn_mean + nn_ci, alpha=0.2, label="NN 95% CI")
    else:
        nn_mean = summary[f"NN_{metric_name}_mean"].to_numpy(dtype=float)
        nn_ci = summary[f"NN_{metric_name}_ci95"].to_numpy(dtype=float)
        plt.plot(x, nn_mean, marker="o", linewidth=2.5, label="NN")
        plt.fill_between(x, nn_mean - nn_ci, nn_mean + nn_ci, alpha=0.2, label="NN 95% CI")

    kkt_mean = summary[f"KKThPINN_{metric_name}_mean"].to_numpy(dtype=float)
    kkt_ci = summary[f"KKThPINN_{metric_name}_ci95"].to_numpy(dtype=float)
    plt.plot(x, kkt_mean, marker="o", linewidth=2.5, label="KKT-hPINN")
    plt.fill_between(x, kkt_mean - kkt_ci, kkt_mean + kkt_ci, alpha=0.2, label="KKT-hPINN 95% CI")

    plt.xlabel("Number of Cao regions")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    if logy:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()
    print(f"Saved {save_name}")


def plot_diagnostic(diagnostics, col, ylabel, save_name, logy=False):
    if diagnostics is None or diagnostics.empty or col not in diagnostics.columns:
        print(f"Skipping {save_name}: column {col} not available.")
        return

    x = diagnostics["num_regions"].to_numpy(dtype=float)
    y = diagnostics[col].to_numpy(dtype=float)

    plt.figure(figsize=(9, 6))
    plt.plot(x, y, marker="o", linewidth=2.5)
    plt.xlabel("Number of Cao regions")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    if logy:
        plt.yscale("log")
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()
    print(f"Saved {save_name}")


def main():
    summary = load_experiment_summary()

    plot_repeated_metric(
        summary,
        "RMSE",
        "RMSE",
        "rmse_vs_regions_1d.png",
        logy=False,
        use_pooled_nn=False,
    )
    plot_repeated_metric(
        summary,
        "Experiment_Time_sec",
        "Experiment time (sec)",
        "time_vs_regions_1d.png",
        logy=False,
        use_pooled_nn=False,
    )
    plot_repeated_metric(
        summary,
        "PL_Violation",
        "Piecewise-linear constraint violation",
        "pl_violation_vs_regions_1d.png",
        logy=True,
        use_pooled_nn=False,
    )
    plot_repeated_metric(
        summary,
        "Original_Nonlinear_Violation",
        "Original nonlinear violation",
        "nonlinear_violation_vs_regions_1d.png",
        logy=True,
        use_pooled_nn=False,
    )

    diag_path = BASE_DIR / "scenario_diagnostics.csv"
    diagnostics = pd.read_csv(diag_path) if diag_path.exists() else pd.DataFrame()

    # Linearization accuracy diagnostics, deterministic per scenario.
    plot_diagnostic(
        diagnostics,
        "linearization_mean_abs_error",
        "Mean absolute linearization error",
        "linearization_mean_abs_error_vs_regions_1d.png",
        logy=True,
    )
    plot_diagnostic(
        diagnostics,
        "linearization_rmse_error",
        "RMSE linearization error",
        "linearization_rmse_error_vs_regions_1d.png",
        logy=True,
    )
    plot_diagnostic(
        diagnostics,
        "linearization_max_abs_error",
        "Max absolute linearization error",
        "linearization_max_abs_error_vs_regions_1d.png",
        logy=True,
    )

    # Projection-only diagnostics, deterministic per scenario.
    plot_diagnostic(
        diagnostics,
        "projection_check_output_MAE_scaled",
        "Projection-check output MAE (scaled)",
        "projection_check_output_mae_scaled_vs_regions_1d.png",
        logy=True,
    )
    plot_diagnostic(
        diagnostics,
        "projection_check_PL_violation",
        "Projection-check PL violation",
        "projection_check_pl_violation_vs_regions_1d.png",
        logy=True,
    )
    plot_diagnostic(
        diagnostics,
        "projection_check_original_nonlinear_violation",
        "Projection-check original nonlinear violation",
        "projection_check_nonlinear_violation_vs_regions_1d.png",
        logy=True,
    )


if __name__ == "__main__":
    main()
