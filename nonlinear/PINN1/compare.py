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


def ci95(std, n):
    if n <= 1 or pd.isna(std):
        return 0.0
    return 1.96 * std / np.sqrt(n)


def plot_bar(summary_df, value_col, ci_col, ylabel, out_file):
    plt.figure(figsize=(6, 5))
    x = np.arange(len(summary_df))
    plt.bar(x, summary_df[value_col])
    plt.errorbar(
        x,
        summary_df[value_col],
        yerr=summary_df[ci_col],
        fmt="none",
        capsize=4,
    )
    plt.xticks(x, summary_df["model"])
    plt.ylabel(ylabel)
    # plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_train_vs_inference_violation(summary_df, out_file):
    models = summary_df["model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(7, 5))

    plt.bar(
        x - width / 2,
        summary_df["final_train_violation_mean"],
        width,
        label="Final training violation",
    )
    plt.bar(
        x + width / 2,
        summary_df["inference_violation_mean"],
        width,
        label="Inference violation",
    )

    plt.errorbar(
        x - width / 2,
        summary_df["final_train_violation_mean"],
        yerr=summary_df["final_train_violation_ci95"],
        fmt="none",
        capsize=4,
    )
    plt.errorbar(
        x + width / 2,
        summary_df["inference_violation_mean"],
        yerr=summary_df["inference_violation_ci95"],
        fmt="none",
        capsize=4,
    )

    plt.xticks(x, models)
    plt.ylabel("Mean absolute nonlinear violation")
    # plt.title("Training vs inference violation")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--mu", type=float, default=0.1)
    parser.add_argument("--n_points", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--dataset_path", type=str, default="data.csv")
    parser.add_argument("--regenerate_data", action="store_true")
    args = parser.parse_args()

    if args.regenerate_data:
        run_command([
            "python", "generate_data.py",
            "--n_total_points", str(args.n_points),
            "--seed", "0",
            "--out_csv", args.dataset_path,
            "--plot_file", f"outputs_vs_Cao_n{args.n_points}.png",
        ])
        
    # Build the 30-region PL matrices for KKThPINN
    run_command([
        "python", "linearization.py",
        "--nC_regions", "30",
    ])

    all_rows = []

    model_settings = [
    {
        "model": "NN",
        "model_id": f"NN_n{args.n_points}",
        "mu": 0.0,
    },
    {
        "model": "PINN",
        "model_id": f"PINN_mu0p1_n{args.n_points}",
        "mu": args.mu,
    },
    {
        "model": "KKThPINN",
        "model_id": f"KKThPINN_n{args.n_points}_seg30",
        "mu": 0.0,
    },
]

    for setting in model_settings:
        model = setting["model"]
        model_id = setting["model_id"]
        mu = setting["mu"]

        for run in range(args.repeats):
            print(f"\n===== model={model}, run={run} =====")

            train_cmd = [
                "python", "main.py",
                "--job", "train",
                "--model", model,
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
                "--model", model,
                "--model_id", model_id,
                "--dataset_type", "cstr",
                "--dataset_path", args.dataset_path,
                "--mu", str(mu),
                "--val_ratio", str(args.val_ratio),
                "--run", str(run),
            ]

            run_command(train_cmd)
            run_command(eval_cmd)

            curve_dir = Path(f"./data/learning_curves/cstr/{model}/{args.val_ratio}")
            report_dir = Path(f"./data/tables/cstr/{model}/{args.val_ratio}")

            train_losses = np.load(curve_dir / f"{model_id}_train_losses_run{run}.npy")
            val_losses = np.load(curve_dir / f"{model_id}_val_losses_run{run}.npy")
            train_violations = np.load(curve_dir / f"{model_id}_train_violations_run{run}.npy")
            val_violations = np.load(curve_dir / f"{model_id}_val_violations_run{run}.npy")

            report_path = report_dir / f"{model_id}_{args.val_ratio}_{run}.csv"
            report = read_report_csv(report_path)

            row = {
                "model": model,
                "model_id": model_id,
                "run": run,
                "mu": mu,
                "n_points": args.n_points,

                "final_train_loss": float(train_losses[-1]),
                "final_val_loss": float(val_losses[-1]),
                "final_train_violation": float(train_violations[-1]),
                "final_val_violation": float(val_violations[-1]),

                "inference_rmse": float(report["rmse_total"]),
                "inference_violation": float(report["violation"]),
            }

            all_rows.append(row)

            pd.DataFrame(all_rows).to_csv(
                "nn_pinn_50_compare_raw.csv",
                index=False,
            )

    raw_df = pd.DataFrame(all_rows)
    raw_df.to_csv("nn_pinn_50_compare_raw.csv", index=False)

    summary = (
        raw_df
        .groupby("model", as_index=False)
        .agg(
            repeats=("run", "count"),

            final_train_loss_mean=("final_train_loss", "mean"),
            final_train_loss_std=("final_train_loss", "std"),

            final_val_loss_mean=("final_val_loss", "mean"),
            final_val_loss_std=("final_val_loss", "std"),

            final_train_violation_mean=("final_train_violation", "mean"),
            final_train_violation_std=("final_train_violation", "std"),

            final_val_violation_mean=("final_val_violation", "mean"),
            final_val_violation_std=("final_val_violation", "std"),

            inference_rmse_mean=("inference_rmse", "mean"),
            inference_rmse_std=("inference_rmse", "std"),

            inference_violation_mean=("inference_violation", "mean"),
            inference_violation_std=("inference_violation", "std"),
        )
    )

    for col in [
        "final_train_loss",
        "final_val_loss",
        "final_train_violation",
        "final_val_violation",
        "inference_rmse",
        "inference_violation",
    ]:
        summary[f"{col}_ci95"] = summary.apply(
            lambda r: ci95(r[f"{col}_std"], r["repeats"]),
            axis=1,
        )

    summary.to_csv("nn_pinn_50_compare_summary.csv", index=False)

    plot_bar(
        summary,
        "inference_rmse_mean",
        "inference_rmse_ci95",
        "Mean experiment RMSE",
        # "Inference RMSE: NN vs PINN",
        "nn_pinn_inference_rmse.png",
    )

    plot_bar(
        summary,
        "final_train_loss_mean",
        "final_train_loss_ci95",
        "Final epoch training loss",
        # "Final training loss: NN vs PINN",
        "nn_pinn_final_train_loss.png",
    )

    plot_train_vs_inference_violation(
        summary,
        "nn_pinn_train_vs_inference_violation.png",
    )

    print("\nSaved:")
    print("  nn_pinn_50_compare_raw.csv")
    print("  nn_pinn_50_compare_summary.csv")
    print("  nn_pinn_inference_rmse.png")
    print("  nn_pinn_final_train_loss.png")
    print("  nn_pinn_train_vs_inference_violation.png")


if __name__ == "__main__":
    main()