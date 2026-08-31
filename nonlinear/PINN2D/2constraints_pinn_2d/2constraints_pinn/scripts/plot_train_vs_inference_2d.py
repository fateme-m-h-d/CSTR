"""Plot training vs inference nonlinear reaction-constraint violation.

This script does not retrain the models. For every run, it loads the saved
checkpoint and evaluates the original nonlinear reaction residual on the
training set. The corresponding inference/test residual is read from
nn_pinn_kkt_2d_compare_raw.csv.
"""

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"

# The project utilities load region_edges.npz, ABb_matrices.csv, and model
# checkpoints using paths relative to the project root.
os.chdir(BASE_DIR)
sys.path.insert(0, str(SRC_DIR))

from train import load_weights  # noqa: E402
from utils import LoadData, LoadModel, nonlinear_residuals  # noqa: E402


MODEL_IDS = {
    "NN": "NN2D",
    "PINN": "PINN2D_selected",
    "KKThPINN": "KKThPINN2D",
}

MODEL_ORDER = ["KKThPINN", "NN", "PINN"]

DISPLAY_NAMES = {
    "KKThPINN": "PL-KKT-hPINN",
    "NN": "NN",
    "PINN": "PINN",
}


def ci95(std, count):
    """Return the normal-approximation 95% CI half-width."""
    if count <= 1 or pd.isna(std):
        return 0.0
    return 1.96 * std / np.sqrt(count)


def training_reaction_violation(args, data, model_name, model_id, run):
    """Evaluate mean |nonlinear reaction residual| on the training set."""
    args.model = model_name
    args.model_id = model_id
    args.run = int(run)

    model = LoadModel(args, data)
    load_weights(model, model_id, args)
    model.eval()

    absolute_residual_sum = 0.0
    sample_count = 0

    with torch.no_grad():
        for X, _ in data["train_loader"]:
            prediction = model(X)

            # nonlinear_residuals columns:
            #   column 0 = original nonlinear reaction residual
            #   column 1 = mass-balance residual
            residual = nonlinear_residuals(
                X,
                prediction,
                data["scaler"],
            )

            absolute_residual_sum += residual[:, 0].abs().sum().item()
            sample_count += X.shape[0]

    if sample_count == 0:
        raise RuntimeError("The training loader is empty.")

    return absolute_residual_sum / sample_count


def build_summary(raw, args, data):
    required_columns = {"model", "run", "violation_rxn"}
    missing_columns = required_columns.difference(raw.columns)
    if missing_columns:
        raise ValueError(
            "The raw comparison CSV is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    rows = []
    total = len(raw)

    for index, record in enumerate(raw.itertuples(index=False), start=1):
        model_name = record.model
        if model_name not in MODEL_IDS:
            raise ValueError(f"Unexpected model name in CSV: {model_name}")

        model_id = MODEL_IDS[model_name]
        run = int(record.run)

        print(
            f"[{index:03d}/{total:03d}] "
            f"model={model_name}, run={run}",
            flush=True,
        )

        train_violation = training_reaction_violation(
            args,
            data,
            model_name,
            model_id,
            run,
        )

        rows.append(
            {
                "model": model_name,
                "run": run,
                "final_train_violation_rxn": train_violation,
                "inference_violation_rxn": float(record.violation_rxn),
            }
        )

    evaluated = pd.DataFrame(rows)

    summary = (
        evaluated.groupby("model", as_index=False)
        .agg(
            repeats=("run", "count"),
            final_train_violation_rxn_mean=(
                "final_train_violation_rxn",
                "mean",
            ),
            final_train_violation_rxn_std=(
                "final_train_violation_rxn",
                "std",
            ),
            inference_violation_rxn_mean=(
                "inference_violation_rxn",
                "mean",
            ),
            inference_violation_rxn_std=(
                "inference_violation_rxn",
                "std",
            ),
        )
    )

    for metric in [
        "final_train_violation_rxn",
        "inference_violation_rxn",
    ]:
        summary[f"{metric}_ci95"] = summary.apply(
            lambda row: ci95(
                row[f"{metric}_std"],
                row["repeats"],
            ),
            axis=1,
        )

    summary["model"] = pd.Categorical(
        summary["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )
    summary = summary.sort_values("model").reset_index(drop=True)

    return evaluated, summary


def create_plot(summary, png_path, pdf_path):
    x = np.arange(len(summary))
    width = 0.35

    train_mean = summary["final_train_violation_rxn_mean"].to_numpy()
    train_ci = summary["final_train_violation_rxn_ci95"].to_numpy()
    inference_mean = summary["inference_violation_rxn_mean"].to_numpy()
    inference_ci = summary["inference_violation_rxn_ci95"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    ax.bar(
        x - width / 2,
        train_mean,
        width,
        color="#1f77b4",
        label="Final training violation",
    )
    ax.bar(
        x + width / 2,
        inference_mean,
        width,
        color="#ff7f0e",
        label="Inference violation",
    )

    ax.errorbar(
        x - width / 2,
        train_mean,
        yerr=train_ci,
        fmt="none",
        ecolor="#1f77b4",
        capsize=4,
        linewidth=1.2,
    )
    ax.errorbar(
        x + width / 2,
        inference_mean,
        yerr=inference_ci,
        fmt="none",
        ecolor="#ff7f0e",
        capsize=4,
        linewidth=1.2,
    )

    labels = [DISPLAY_NAMES[str(model)] for model in summary["model"]]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean absolute nonlinear reaction-constraint violation")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Plot final training and inference nonlinear reaction-constraint "
            "violations for NN, PINN, and PL-KKT-hPINN."
        )
    )
    parser.add_argument(
        "--raw_csv",
        default="nn_pinn_kkt_2d_compare_raw.csv",
    )
    parser.add_argument("--dataset_path", default="data.csv")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dtype", type=int, choices=[32, 64], default=64)
    parser.add_argument("--input_dim", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--hidden_num", type=int, default=2)
    parser.add_argument("--z0_dim", type=int, default=3)
    parser.add_argument(
        "--output_prefix",
        default="nn_pinn_kkt_2d_train_vs_inference_rxn",
    )
    return parser.parse_args()


def main():
    cli = parse_arguments()

    torch.set_default_dtype(
        torch.float64 if cli.dtype == 64 else torch.float32
    )

    raw_path = Path(cli.raw_csv)
    if not raw_path.is_absolute():
        raw_path = BASE_DIR / raw_path
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw comparison CSV not found: {raw_path}")

    args = SimpleNamespace(
        dataset_type="cstr",
        dataset_path=cli.dataset_path,
        dtype=cli.dtype,
        val_ratio=cli.val_ratio,
        batch_size=cli.batch_size,
        input_dim=cli.input_dim,
        hidden_dim=cli.hidden_dim,
        hidden_num=cli.hidden_num,
        z0_dim=cli.z0_dim,
    )

    data = LoadData(args)
    raw = pd.read_csv(raw_path)
    evaluated, summary = build_summary(raw, args, data)

    output_prefix = BASE_DIR / cli.output_prefix
    raw_output = Path(f"{output_prefix}_raw.csv")
    summary_output = Path(f"{output_prefix}_summary.csv")
    png_output = Path(f"{output_prefix}.png")
    pdf_output = Path(f"{output_prefix}.pdf")

    evaluated.to_csv(raw_output, index=False)
    summary.to_csv(summary_output, index=False)
    create_plot(summary, png_output, pdf_output)

    print("\nSaved:")
    print(f"  {raw_output}")
    print(f"  {summary_output}")
    print(f"  {png_output}")
    print(f"  {pdf_output}")


if __name__ == "__main__":
    main()
