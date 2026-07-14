import argparse
import subprocess
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def mu_to_label(mu):
    return str(mu).replace("-", "m").replace(".", "p").replace("+", "")


def read_report_csv(path):
    """
    Your create_report() saves key,value rows.
    This converts that file into a dictionary.
    """
    df = pd.read_csv(path, header=None, names=["key", "value"])
    return dict(zip(df["key"], df["value"]))


def run_command(cmd):
    print("\nRunning:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mus",
        nargs="+",
        type=float,
        default=[0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dataset_path", type=str, default="data.csv")
    parser.add_argument("--val_ratio", type=float, default=0.2)

    args = parser.parse_args()

    all_rows = []

    for mu in args.mus:
        mu_label = mu_to_label(mu)

        for run in range(args.repeats):
            model_id = f"PINN_mu_{mu_label}"

            train_cmd = [
                "python", "main.py",
                "--job", "train",
                "--model", "PINN",
                "--model_id", model_id,
                "--dataset_type", "cstr",
                "--dataset_path", args.dataset_path,
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
                "--mu", str(mu),
                "--val_ratio", str(args.val_ratio),
                "--run", str(run),
            ]

            eval_cmd = [
                "python", "main.py",
                "--job", "experiment",
                "--model", "PINN",
                "--model_id", model_id,
                "--dataset_type", "cstr",
                "--dataset_path", args.dataset_path,
                "--mu", str(mu),
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
                "mu": mu,
                "run": run,
                "model_id": model_id,
                "rmse_total": float(report["rmse_total"]),
                "violation": float(report["violation"]),
            }

            all_rows.append(row)

            pd.DataFrame(all_rows).to_csv(
                "pinn_mu_sweep_results_raw.csv",
                index=False,
            )

    raw_df = pd.DataFrame(all_rows)
    raw_df.to_csv("pinn_mu_sweep_results_raw.csv", index=False)

    summary_df = (
        raw_df
        .groupby("mu", as_index=False)
        .agg(
            rmse_mean=("rmse_total", "mean"),
            rmse_std=("rmse_total", "std"),
            violation_mean=("violation", "mean"),
            violation_std=("violation", "std"),
        )
    )

    summary_df.to_csv("pinn_mu_sweep_summary.csv", index=False)

    # Plot RMSE versus violation
    plt.figure(figsize=(7, 5))
    plt.scatter(summary_df["violation_mean"], summary_df["rmse_mean"])

    for _, row in summary_df.iterrows():
        plt.annotate(
            f"mu={row['mu']:g}",
            (row["violation_mean"], row["rmse_mean"]),
            textcoords="offset points",
            xytext=(5, 5),
        )

    plt.xlabel("Mean violation")
    plt.ylabel("Mean RMSE")
    plt.title("PINN weight trade-off: RMSE vs violation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pinn_mu_tradeoff_rmse_vs_violation.png", dpi=300)
    plt.close()

    print("\nSaved:")
    print("  pinn_mu_sweep_results_raw.csv")
    print("  pinn_mu_sweep_summary.csv")
    print("  pinn_mu_tradeoff_rmse_vs_violation.png")


if __name__ == "__main__":
    main()