import argparse
import subprocess
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def value_to_label(x):
    return str(x).replace("-", "m").replace(".", "p").replace("+", "")


def read_report_csv(path):
    df = pd.read_csv(path, header=None, names=["key", "value"])
    return dict(zip(df["key"], df["value"]))


def run_command(cmd):
    print("\nRunning:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    
def plot_pareto(summary_df, mode):
    plt.figure(figsize=(7, 5))

    if mode == "rxn_only":
        x_col = "violation_rxn_mean"
        y_col = "rmse_mean"
        label_col = "mu_rxn"
        xlabel = "Mean reaction violation"
        ylabel = "Mean RMSE"
        out_file = "pinn_rxn_only_pareto_rmse_vs_violation.png"

    elif mode == "mb_only":
        x_col = "violation_mb_mean"
        y_col = "rmse_mean"
        label_col = "mu_mb"
        xlabel = "Mean mass-balance violation"
        ylabel = "Mean RMSE"
        out_file = "pinn_mb_only_pareto_rmse_vs_violation.png"

    elif mode == "scale_both":
        x_col = "violation_mean"
        y_col = "rmse_mean"
        label_col = "label_value"
        xlabel = "Mean total constraint violation"
        ylabel = "Mean RMSE"
        out_file = "pinn_scale_both_pareto_rmse_vs_violation.png"

    else:
        raise ValueError("mode must be rxn_only, mb_only, or scale_both")

    plt.scatter(summary_df[x_col], summary_df[y_col])

    for _, row in summary_df.iterrows():
        if mode == "scale_both":
            label = f"({row['mu_rxn']:g}, {row['mu_mb']:g})"
        else:
            label = f"{label_col}={row[label_col]:g}"

        plt.annotate(
            label,
            (row[x_col], row[y_col]),
            textcoords="offset points",
            xytext=(6, 6),
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("PINN weight trade-off: RMSE vs violation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"Saved plot: {out_file}")


# def plot_tradeoff(summary_df, x_col, y_col, label_col, xlabel, ylabel, out_file):
#     plt.figure(figsize=(7, 5))

#     plt.scatter(summary_df[x_col], summary_df[y_col])

#     for _, row in summary_df.iterrows():
#         plt.annotate(
#             f"{label_col}={row[label_col]:g}",
#             (row[x_col], row[y_col]),
#             textcoords="offset points",
#             xytext=(5, 5),
#         )

#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title("PINN weight trade-off: RMSE vs violation")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(out_file, dpi=300)
#     plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["rxn_only", "mb_only", "scale_both"],
    )

    parser.add_argument(
        "--mus",
        nargs="+",
        type=float,
        default=[0.0, 1e-4, 1e-3, 1e-2, 1e-1, 2e-1, 5e-1, 1.0, 2.0, 5.0, 10.0],
    )

    parser.add_argument("--base_mu_rxn", type=float, default=0.1)
    parser.add_argument("--base_mu_mb", type=float, default=1.0)

    parser.add_argument(
        "--scale_factors",
        nargs="+",
        type=float,
        default=[0.1, 1.0, 10.0],
    )

    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dataset_path", type=str, default="data.csv")
    parser.add_argument("--val_ratio", type=float, default=0.2)

    args = parser.parse_args()
    
    master_raw_file = Path("pinn_2constraint_all_raw.csv")
    master_summary_file = Path("pinn_2constraint_all_summary.csv")


    all_rows = []

    if args.mode == "rxn_only":
        cases = []
        for mu in args.mus:
            cases.append({
                "mu_rxn": mu,
                "mu_mb": 0.0,
                "label_value": mu,
            })

    elif args.mode == "mb_only":
        cases = []
        for mu in args.mus:
            cases.append({
                "mu_rxn": 0.0,
                "mu_mb": mu,
                "label_value": mu,
            })

    else:
        cases = []
        for scale in args.scale_factors:
            cases.append({
                "mu_rxn": args.base_mu_rxn * scale,
                "mu_mb": args.base_mu_mb * scale,
                "label_value": scale,
            })

    for case in cases:
        mu_rxn = case["mu_rxn"]
        mu_mb = case["mu_mb"]

        mu_rxn_label = value_to_label(mu_rxn)
        mu_mb_label = value_to_label(mu_mb)

        model_id = f"PINN_rxn{mu_rxn_label}_mb{mu_mb_label}"

        for run in range(args.repeats):
            print(
                f"\n===== mode={args.mode}, "
                f"mu_rxn={mu_rxn}, mu_mb={mu_mb}, run={run} ====="
            )

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
                "--mu_rxn", str(mu_rxn),
                "--mu_mb", str(mu_mb),
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
                "--mu_rxn", str(mu_rxn),
                "--mu_mb", str(mu_mb),
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
                "mode": args.mode,
                "mu_rxn": mu_rxn,
                "mu_mb": mu_mb,
                "label_value": case["label_value"],
                "run": run,
                "model_id": model_id,
                "rmse_total": float(report["rmse_total"]),
                "violation": float(report["violation"]),
                "violation_rxn": float(report["violation_rxn"]),
                "violation_mb": float(report["violation_mb"]),
            }

            all_rows.append(row)

            # Save current run progress so you do not lose results if the job stops
            current_df = pd.DataFrame(all_rows)

            if master_raw_file.exists():
                old_df = pd.read_csv(master_raw_file)
                combined_df = pd.concat([old_df, current_df], ignore_index=True)
            else:
                combined_df = current_df

            combined_df = combined_df.drop_duplicates(
                subset=["mode", "mu_rxn", "mu_mb", "run"],
                keep="last",
            )

            combined_df.to_csv(master_raw_file, index=False)

        # Load the full master raw file, including previous weight combinations
    raw_df = pd.read_csv(master_raw_file)

    summary_df = (
        raw_df
        .groupby(["mode", "mu_rxn", "mu_mb", "label_value"], as_index=False)
        .agg(
            repeats=("run", "count"),
            rmse_mean=("rmse_total", "mean"),
            rmse_std=("rmse_total", "std"),
            violation_mean=("violation", "mean"),
            violation_std=("violation", "std"),
            violation_rxn_mean=("violation_rxn", "mean"),
            violation_rxn_std=("violation_rxn", "std"),
            violation_mb_mean=("violation_mb", "mean"),
            violation_mb_std=("violation_mb", "std"),
        )
    )

    summary_df.to_csv(master_summary_file, index=False)

    # Plot only the mode you just ran
    summary_mode = summary_df[summary_df["mode"] == args.mode].copy()
    plot_pareto(summary_mode, args.mode)

    # if args.mode == "rxn_only":
    #     plot_tradeoff(
    #         summary_df,
    #         x_col="violation_rxn_mean",
    #         y_col="rmse_mean",
    #         label_col="label_value",
    #         xlabel="Mean reaction violation",
    #         ylabel="Mean RMSE",
    #         out_file="pinn_rxn_weight_tradeoff_rmse_vs_violation.png",
    #     )

    # elif args.mode == "mb_only":
    #     plot_tradeoff(
    #         summary_df,
    #         x_col="violation_mb_mean",
    #         y_col="rmse_mean",
    #         label_col="label_value",
    #         xlabel="Mean mass-balance violation",
    #         ylabel="Mean RMSE",
    #         out_file="pinn_mb_weight_tradeoff_rmse_vs_violation.png",
    #     )

    # else:
    #     plot_tradeoff(
    #         summary_df,
    #         x_col="violation_mean",
    #         y_col="rmse_mean",
    #         label_col="label_value",
    #         xlabel="Mean total constraint violation",
    #         ylabel="Mean RMSE",
    #         out_file="pinn_scale_both_tradeoff_rmse_vs_violation.png",
    #     )

    print("\nSaved:")
    # print(f"  pinn_2constraint_{args.mode}_raw.csv")
    # print(f"  pinn_2constraint_{args.mode}_summary.csv")


if __name__ == "__main__":
    main()