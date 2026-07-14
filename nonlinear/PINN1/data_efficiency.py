import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_command(cmd):
    print("\nRunning:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def read_report_csv(path):
    df = pd.read_csv(path, header=None, names=["key", "value"])
    return dict(zip(df["key"], df["value"]))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sample_sizes", nargs="+", type=int,
                        default=[150, 200, 300, 500, 800, 1000])
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--mu", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--data_seed", type=int, default=0)

    args = parser.parse_args()

    all_rows = []

    for n_points in args.sample_sizes:
        data_file = f"data_pinn_n{n_points}.csv"
        plot_file = f"outputs_vs_Cao_n{n_points}.png"

        # Generate one fixed database for this sample size
        run_command([
            "python", "generate_data.py",
            "--n_total_points", str(n_points),
            "--seed", str(args.data_seed),
            "--out_csv", data_file,
            "--plot_file", plot_file,
        ])

        model_id = f"PINN_mu0p1_n{n_points}"

        for run in range(args.repeats):
            print(f"\n===== n_points={n_points}, run={run} =====")

            train_cmd = [
                "python", "main.py",
                "--job", "train",
                "--model", "PINN",
                "--model_id", model_id,
                "--dataset_type", "cstr",
                "--dataset_path", data_file,
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
                "--mu", str(args.mu),
                "--val_ratio", str(args.val_ratio),
                "--run", str(run),
            ]

            eval_cmd = [
                "python", "main.py",
                "--job", "experiment",
                "--model", "PINN",
                "--model_id", model_id,
                "--dataset_type", "cstr",
                "--dataset_path", data_file,
                "--mu", str(args.mu),
                "--val_ratio", str(args.val_ratio),
                "--run", str(run),
            ]

            run_command(train_cmd)
            run_command(eval_cmd)

            report_path = Path(
                f"./data/tables/cstr/PINN/{args.val_ratio}/"
                f"{model_id}_{args.val_ratio}_{run}.csv"
            )

            report = read_report_csv(report_path)

            row = {
                "n_points": n_points,
                "run": run,
                "mu": args.mu,
                "model_id": model_id,
                "rmse_total": float(report["rmse_total"]),
                "violation": float(report["violation"]),
            }

            all_rows.append(row)

            # Save continuously, so if the server stops, you do not lose everything
            pd.DataFrame(all_rows).to_csv(
                "pinn_data_efficiency_raw.csv",
                index=False,
            )

    raw_df = pd.DataFrame(all_rows)
    raw_df.to_csv("pinn_data_efficiency_raw.csv", index=False)

    summary_df = (
        raw_df
        .groupby("n_points", as_index=False)
        .agg(
            rmse_mean=("rmse_total", "mean"),
            rmse_std=("rmse_total", "std"),
            violation_mean=("violation", "mean"),
            violation_std=("violation", "std"),
            repeats=("run", "count"),
        )
    )

    summary_df["rmse_ci95"] = 1.96 * summary_df["rmse_std"] / np.sqrt(summary_df["repeats"])
    summary_df["violation_ci95"] = 1.96 * summary_df["violation_std"] / np.sqrt(summary_df["repeats"])

    summary_df.to_csv("pinn_data_efficiency_summary.csv", index=False)

    # Plot RMSE vs number of sample points
    plt.figure(figsize=(7, 5))
    plt.errorbar(
        summary_df["n_points"],
        summary_df["rmse_mean"],
        yerr=summary_df["rmse_ci95"],
        marker="o",
        capsize=4,
    )
    plt.xlabel("Number of sample points in database")
    plt.ylabel("Mean experiment RMSE")
    plt.title("PINN data efficiency: RMSE vs sample size")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pinn_data_efficiency_rmse.png", dpi=300)
    plt.close()

    # Plot violation vs number of sample points
    plt.figure(figsize=(7, 5))
    plt.errorbar(
        summary_df["n_points"],
        summary_df["violation_mean"],
        yerr=summary_df["violation_ci95"],
        marker="o",
        capsize=4,
    )
    plt.xlabel("Number of sample points in database")
    plt.ylabel("Mean experiment violation")
    plt.title("PINN data efficiency: violation vs sample size")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pinn_data_efficiency_violation.png", dpi=300)
    plt.close()

    print("\nSaved:")
    print("  pinn_data_efficiency_raw.csv")
    print("  pinn_data_efficiency_summary.csv")
    print("  pinn_data_efficiency_rmse.png")
    print("  pinn_data_efficiency_violation.png")


if __name__ == "__main__":
    main()