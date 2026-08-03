from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path.cwd()
SUMMARY_PATH = BASE_DIR / "metric_summary_by_samples.csv"
PLOTS = [
    ("RMSE", "RMSE", "rmse_vs_samples_1d_data_efficiency.png", False),
]


def plot_metric(summary, metric_name, ylabel, filename, log_scale):
    x = summary["num_sample_points"].to_numpy(dtype=float)
    nn_mean = summary[f"NN_{metric_name}_mean"].to_numpy(dtype=float)
    nn_ci = summary[f"NN_{metric_name}_ci95"].to_numpy(dtype=float)
    kkt_mean = summary[f"KKThPINN_{metric_name}_mean"].to_numpy(dtype=float)
    kkt_ci = summary[f"KKThPINN_{metric_name}_ci95"].to_numpy(dtype=float)

    nn_lower = nn_mean - nn_ci
    kkt_lower = kkt_mean - kkt_ci
    if log_scale:
        nn_lower = np.maximum(nn_lower, 1e-18)
        kkt_lower = np.maximum(kkt_lower, 1e-18)

    plt.figure(figsize=(9, 6))
    plt.plot(x, nn_mean, marker="o", linewidth=2.5, label="NN")
    plt.fill_between(
        x, nn_lower, nn_mean + nn_ci, alpha=0.2, label="NN 95% CI"
    )
    plt.plot(x, kkt_mean, marker="o", linewidth=2.5, label="KKT-hPINN")
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
    if log_scale:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / filename, dpi=300)
    plt.close()


def main():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "Run `python3 -m scripts.run_all` before creating figures."
        )

    summary = pd.read_csv(SUMMARY_PATH).sort_values("num_sample_points")
    for metric_name, ylabel, filename, log_scale in PLOTS:
        plot_metric(summary, metric_name, ylabel, filename, log_scale)


if __name__ == "__main__":
    main()
