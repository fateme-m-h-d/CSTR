import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = BASE_DIR / "pinn_2d_mu_sweep_raw.csv"
SUMMARY_PATH = BASE_DIR / "pinn_2d_mu_sweep_summary.csv"
PLOT_PATH = BASE_DIR / "pinn_2d_pareto_validation.png"


def value_label(value):
    return f"{value:g}".replace("-", "m").replace(".", "p").replace("+", "")


def read_report(path):
    frame = pd.read_csv(path, header=None, names=["key", "value"])
    return dict(zip(frame["key"], frame["value"]))


def run(command):
    print("\nRunning:", " ".join(command), flush=True)
    subprocess.run(command, cwd=BASE_DIR, check=True)


def pareto_mask(frame, x_col, y_col):
    values = frame[[x_col, y_col]].to_numpy(dtype=float)
    keep = np.ones(len(values), dtype=bool)
    for i, current in enumerate(values):
        weakly_better = np.all(values <= current, axis=1)
        strictly_better = np.any(values < current, axis=1)
        weakly_better[i] = False
        keep[i] = not np.any(weakly_better & strictly_better)
    return keep


def save_summary(raw):
    summary = (
        raw.groupby(["mu_rxn", "mu_mb"], as_index=False)
        .agg(
            repeats=("run", "count"),
            rmse_mean=("rmse_total", "mean"),
            rmse_std=("rmse_total", "std"),
            violation_total_mean=("violation_original_nonlinear", "mean"),
            violation_total_std=("violation_original_nonlinear", "std"),
            violation_rxn_mean=("violation_rxn", "mean"),
            violation_rxn_std=("violation_rxn", "std"),
            violation_mb_mean=("violation_mb", "mean"),
            violation_mb_std=("violation_mb", "std"),
        )
    )
    summary["is_pareto"] = pareto_mask(
        summary, "violation_total_mean", "rmse_mean"
    )
    summary.to_csv(SUMMARY_PATH, index=False)

    front = summary[summary["is_pareto"]].sort_values(
        "violation_total_mean"
    )
    plt.figure(figsize=(7, 5))
    plt.scatter(
        summary["violation_total_mean"],
        summary["rmse_mean"],
        color="0.7",
        label="All weight pairs",
    )
    plt.plot(
        front["violation_total_mean"],
        front["rmse_mean"],
        "o-",
        color="tab:blue",
        label="Non-dominated front",
    )
    for _, row in front.iterrows():
        plt.annotate(
            f"({row.mu_rxn:g}, {row.mu_mb:g})",
            (row.violation_total_mean, row.rmse_mean),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    plt.xscale("log")
    plt.xlabel("Mean absolute original nonlinear violation (validation)")
    plt.ylabel("RMSE (validation)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mu_rxn_values",
        nargs="+",
        type=float,
        default=[0.0, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0],
    )
    parser.add_argument(
        "--mu_mb_values",
        nargs="+",
        type=float,
        default=[0.0, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dtype", type=int, choices=[32, 64], default=64)
    parser.add_argument("--dataset_path", default="data.csv")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    rows = []
    for mu_rxn in args.mu_rxn_values:
        for mu_mb in args.mu_mb_values:
            model_id = (
                f"PINN2D_rxn{value_label(mu_rxn)}_mb{value_label(mu_mb)}"
            )
            for run_index in range(args.repeats):
                common = [
                    "--model", "PINN",
                    "--model_id", model_id,
                    "--dataset_type", "cstr",
                    "--dataset_path", args.dataset_path,
                    "--batch_size", str(args.batch_size),
                    "--lr", str(args.lr),
                    "--dtype", str(args.dtype),
                    "--mu_rxn", str(mu_rxn),
                    "--mu_mb", str(mu_mb),
                    "--val_ratio", str(args.val_ratio),
                    "--run", str(run_index),
                    "--seed", str(run_index),
                ]
                run([
                    sys.executable, "src/main.py", "--job", "train",
                    "--epochs", str(args.epochs), *common,
                ])
                run([
                    sys.executable, "src/main.py", "--job", "experiment",
                    "--eval_split", "val", *common,
                ])

                report_path = (
                    BASE_DIR / "data" / "tables" / "cstr" / "PINN"
                    / str(args.val_ratio)
                    / f"{model_id}_{args.val_ratio}_{run_index}.csv"
                )
                report = read_report(report_path)
                rows.append({
                    "mu_rxn": mu_rxn,
                    "mu_mb": mu_mb,
                    "run": run_index,
                    "epochs": args.epochs,
                    "dtype": args.dtype,
                    "rmse_total": float(report["rmse_total"]),
                    "violation_original_nonlinear": float(
                        report["violation_original_nonlinear"]
                    ),
                    "violation_rxn": float(report["violation_rxn"]),
                    "violation_mb": float(report["violation_mb"]),
                })
                pd.DataFrame(rows).to_csv(RAW_PATH, index=False)

    raw = pd.DataFrame(rows)
    save_summary(raw)
    print(f"\nSaved {RAW_PATH.name}, {SUMMARY_PATH.name}, and {PLOT_PATH.name}")


if __name__ == "__main__":
    main()
