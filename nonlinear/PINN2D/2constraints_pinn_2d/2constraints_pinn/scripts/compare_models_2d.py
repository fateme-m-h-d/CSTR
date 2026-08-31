import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = BASE_DIR / "nn_pinn_kkt_2d_compare_raw.csv"
SUMMARY_PATH = BASE_DIR / "nn_pinn_kkt_2d_compare_summary.csv"


def read_report(path):
    frame = pd.read_csv(path, header=None, names=["key", "value"])
    return dict(zip(frame["key"], frame["value"]))


def run(command):
    print("\nRunning:", " ".join(command), flush=True)
    start = time.perf_counter()
    subprocess.run(command, cwd=BASE_DIR, check=True)
    return time.perf_counter() - start


def ci95(std, count):
    if count <= 1 or pd.isna(std):
        return 0.0
    return 1.96 * std / np.sqrt(count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu_rxn", type=float, required=True)
    parser.add_argument("--mu_mb", type=float, required=True)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dtype", type=int, choices=[32, 64], default=64)
    parser.add_argument("--dataset_path", default="data.csv")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    settings = [
        ("NN", "NN2D", 0.0, 0.0),
        ("PINN", "PINN2D_selected", args.mu_rxn, args.mu_mb),
        ("KKThPINN", "KKThPINN2D", 0.0, 0.0),
    ]
    rows = []
    for model, model_id, mu_rxn, mu_mb in settings:
        for run_index in range(args.repeats):
            common = [
                "--model", model,
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
            train_time = run([
                sys.executable, "src/main.py", "--job", "train",
                "--epochs", str(args.epochs), *common,
            ])
            run([
                sys.executable, "src/main.py", "--job", "experiment",
                "--eval_split", "test", *common,
            ])

            report_path = (
                BASE_DIR / "data" / "tables" / "cstr" / model
                / str(args.val_ratio)
                / f"{model_id}_{args.val_ratio}_{run_index}.csv"
            )
            report = read_report(report_path)
            rows.append({
                "model": model,
                "run": run_index,
                "mu_rxn": mu_rxn,
                "mu_mb": mu_mb,
                "training_time_sec": train_time,
                "rmse_total": float(report["rmse_total"]),
                "violation_pl": float(report["violation"]),
                "violation_original_nonlinear": float(
                    report["violation_original_nonlinear"]
                ),
                "violation_rxn": float(report["violation_rxn"]),
                "violation_mb": float(report["violation_mb"]),
                "prediction_time_sec": float(report["prediction_time_sec"]),
            })
            pd.DataFrame(rows).to_csv(RAW_PATH, index=False)

    raw = pd.DataFrame(rows)
    metrics = [
        "training_time_sec",
        "rmse_total",
        "violation_pl",
        "violation_original_nonlinear",
        "violation_rxn",
        "violation_mb",
        "prediction_time_sec",
    ]
    aggregations = {"repeats": ("run", "count")}
    for metric in metrics:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
    summary = raw.groupby("model", as_index=False).agg(**aggregations)
    for metric in metrics:
        summary[f"{metric}_ci95"] = summary.apply(
            lambda row: ci95(row[f"{metric}_std"], row["repeats"]), axis=1
        )
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"\nSaved {RAW_PATH.name} and {SUMMARY_PATH.name}")


if __name__ == "__main__":
    main()
