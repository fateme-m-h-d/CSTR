from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path.cwd()
SUMMARY_PATH = BASE_DIR / "metric_summary_by_samples.csv"
PLOTS = [
    ("RMSE", "RMSE", "rmse_vs_samples_2d.png", False),
]


def plot_metric(summary, metric, ylabel, filename, log_scale):
    x = summary["num_sample_points"].to_numpy(dtype=float)
    plt.figure(figsize=(9, 6))
    for prefix, label in (("NN", "NN"), ("KKThPINN", "KKT-hPINN")):
        mean = summary[f"{prefix}_{metric}_mean"].to_numpy(dtype=float)
        ci = summary[f"{prefix}_{metric}_ci95"].to_numpy(dtype=float)
        lower = mean - ci
        if log_scale:
            lower = np.maximum(lower, 1e-18)
        plt.plot(x, mean, marker="o", linewidth=2.5, label=label)
        plt.fill_between(x, lower, mean + ci, alpha=0.2)
    plt.xlabel("Number of sample points")
    plt.ylabel(ylabel)
    if log_scale:
        plt.yscale("log")
    plt.grid(True, alpha=0.3)
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
    for plot in PLOTS:
        plot_metric(summary, *plot)


if __name__ == "__main__":
    main()
