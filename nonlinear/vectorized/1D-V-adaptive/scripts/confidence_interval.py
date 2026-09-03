from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path.cwd()
SUMMARY_PATH = BASE_DIR / "metric_summary_by_segments.csv"

PLOTS = [
    ("RMSE", "RMSE", "rmse_vs_regions_1d.png", False),
    ("Experiment_Time_sec", "Experiment time (sec)", "time_vs_regions_1d.png", False),
    ("Original_Nonlinear_Violation", "Original nonlinear violation", "nonlinear_violation_vs_regions_1d.png", True),
]


def plot_metric(summary, metric_name, ylabel, save_name, logy=False):
    x = summary["num_regions"].to_numpy(dtype=float)
    pooled_mean_col = f"NN_{metric_name}_pooled_mean"
    pooled_ci_col = f"NN_{metric_name}_pooled_ci95"
    if pooled_mean_col in summary and pooled_ci_col in summary:
        nn_mean = float(summary[pooled_mean_col].iloc[-1])
        nn_ci = float(summary[pooled_ci_col].iloc[-1])
    else:
        nn_mean = float(summary[f"NN_{metric_name}_mean"].mean())
        nn_ci = float(summary[f"NN_{metric_name}_ci95"].mean())

    kkt_mean = summary[f"KKThPINN_{metric_name}_mean"].to_numpy(dtype=float)
    kkt_ci = summary[f"KKThPINN_{metric_name}_ci95"].to_numpy(dtype=float)

    plt.figure(figsize=(9, 6))
    plt.plot(x, [nn_mean] * len(x), marker="o", linewidth=2.5, label="NN")
    plt.fill_between(x, nn_mean - nn_ci, nn_mean + nn_ci, alpha=0.2, label="NN 95% CI")
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


def main():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError("Run `python3 -m scripts.run_all` before creating figures.")

    summary = pd.read_csv(SUMMARY_PATH)
    for metric_name, ylabel, save_name, logy in PLOTS:
        plot_metric(summary, metric_name, ylabel, save_name, logy=logy)


if __name__ == "__main__":
    main()
