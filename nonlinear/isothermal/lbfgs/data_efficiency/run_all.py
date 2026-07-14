import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1D data-efficiency scenario driver
#
# Fixed input: Cao only
# Fixed constraints per region:
#   1) piecewise-linearized nonlinear reaction constraint
#   2) exact linear mass-balance constraint
#
# Fixed number of Cao regions by default:
#     --nC_regions 30
#
# Vary:
#     --inner_scenarios, i.e., n_inner_per_region
#
# LBFGS default training setup:
#   --optimizer LBFGS --lr 1.0 --batch_size 90 --epochs 5 --dtype 32
# ============================================================

BASE_DIR = Path.cwd()
PYTHON_EXE = sys.executable


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--inner_scenarios",
        type=int,
        nargs="+",
        default=[0, 1, 2, 5, 10, 15, 20, 25],
        help="List of n_inner_per_region values for the data-efficiency test.",
    )
    parser.add_argument(
        "--nC_regions",
        type=int,
        default=30,
        help="Fixed number of Cao regions used for piecewise linearization.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_iterations", type=int, default=50)

    parser.add_argument(
        "--models",
        nargs="+",
        default=["NN", "KKThPINN"],
        help="Models to run. Use --models KKThPINN if you only want KKT-hPINN.",
    )

    # main.py training/evaluation settings
    parser.add_argument("--optimizer", type=str, default="LBFGS")
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=90)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dtype", type=int, default=32, choices=[32, 64])
    parser.add_argument("--model_id", type=str, default="MODELID")
    parser.add_argument("--val_ratio", type=float, default=0.2)

    return parser.parse_args()


def run_cmd(cmd, *, env=None, capture=False):
    print("\n" + " ".join(map(str, cmd)))

    result = subprocess.run(
        list(map(str, cmd)),
        cwd=BASE_DIR,
        env=env,
        text=True,
        capture_output=capture,
        check=True,
    )

    if capture:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)

    return result


def copy_if_exists(src, dst):
    src = BASE_DIR / src
    dst = BASE_DIR / dst

    if src.exists():
        shutil.copy2(src, dst)
        print(f"Copied {src.name} -> {dst.name}")
    else:
        print(f"Warning: {src.name} not found; could not copy.")


def remove_if_exists(path):
    path = BASE_DIR / path

    if path.exists():
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def cleanup_current_outputs():
    """Remove files regenerated in each scenario to avoid stale copies."""

    files_to_remove = [
        "data.csv",
        "data_with_region_info.csv",
        "failed_points_fixed_data.csv",
        "scaled_data.csv",
        "scaler.pkl",
        "ABb_matrices.csv",
        "lin_params.csv",
        "region_edges.npz",
        "sampling_region_edges.npz",
        "linearization_accuracy_detailed.csv",
        "linearization_accuracy_summary.csv",
        "experiment_epoch_errors.csv",
        "training_epoch_errors.csv",
        "outputs_vs_Cao.png",
        "sampling_points_Cao.png",
    ]

    for f in files_to_remove:
        remove_if_exists(f)


def count_rows(csv_path):
    try:
        return int(pd.read_csv(BASE_DIR / csv_path).shape[0])
    except Exception:
        return -1


def compute_linearization_overall():
    detailed_path = BASE_DIR / "linearization_accuracy_detailed.csv"
    summary_path = BASE_DIR / "linearization_accuracy_summary.csv"

    if not detailed_path.exists():
        return {}

    detailed = pd.read_csv(detailed_path)

    out = {
        "linearization_mean_abs_error": float(detailed["abs_linearization_error"].mean()),
        "linearization_rmse_error": float(np.sqrt(np.mean(detailed["linearization_error"] ** 2))),
        "linearization_max_abs_error": float(detailed["abs_linearization_error"].max()),
        "mean_abs_nonlinear_residual_on_data": float(detailed["residual_nonlinear"].abs().mean()),
        "mean_abs_linearized_residual_on_data": float(detailed["residual_linearized"].abs().mean()),
    }

    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        out["linearization_mean_abs_error_region_average"] = float(summary["mean_abs_linearization_error"].mean())
        out["linearization_rmse_error_region_average"] = float(summary["rmse_linearization_error"].mean())

    return out


def parse_projection_check_stdout(stdout):
    results = {}
    pattern = re.compile(r"^([A-Za-z0-9_]+):\s*([-+0-9.eE]+)")

    for line in stdout.splitlines():
        match = pattern.match(line.strip())
        if match:
            key, value = match.groups()
            try:
                results[key] = float(value)
            except ValueError:
                pass

    return results


def run_projection_check(dtype):
    result = run_cmd(
        [
            PYTHON_EXE,
            "projection_check.py",
            "--model",
            "KKThPINN",
            "--model_id",
            "projection_check",
            "--input_dim",
            "1",
            "--z0_dim",
            "3",
            "--dataset_type",
            "cstr",
            "--dataset_path",
            "data.csv",
            "--job",
            "projection_check",
            "--dtype",
            str(dtype),
        ],
        capture=True,
    )

    return parse_projection_check_stdout(result.stdout)


def archive_scenario_files(n_inner):
    # Data files
    copy_if_exists("data.csv", f"data_ninner_{n_inner}.csv")
    copy_if_exists("data_with_region_info.csv", f"data_with_region_info_ninner_{n_inner}.csv")

    # Core linearization files
    copy_if_exists("ABb_matrices.csv", f"ABb_matrices_ninner_{n_inner}.csv")
    copy_if_exists("lin_params.csv", f"lin_params_ninner_{n_inner}.csv")
    copy_if_exists("region_edges.npz", f"region_edges_ninner_{n_inner}.npz")
    copy_if_exists("sampling_region_edges.npz", f"sampling_region_edges_ninner_{n_inner}.npz")

    # Diagnostics
    copy_if_exists("linearization_accuracy_detailed.csv", f"linearization_accuracy_detailed_ninner_{n_inner}.csv")
    copy_if_exists("linearization_accuracy_summary.csv", f"linearization_accuracy_summary_ninner_{n_inner}.csv")

    # Repeated experiment outputs
    copy_if_exists("experiment_epoch_errors.csv", f"experiment_epoch_errors_ninner_{n_inner}.csv")
    copy_if_exists("training_epoch_errors.csv", f"training_epoch_errors_ninner_{n_inner}.csv")


def main():
    args = parse_args()
    diagnostic_rows = []

    print("\n=== Data-efficiency setup ===")
    print(f"Cao input only; fixed nC_regions={args.nC_regions}")
    print("Constraints per region: linearized reaction balance + exact mass balance")
    print(
        f"Training setup: optimizer={args.optimizer}, lr={args.lr}, "
        f"batch_size={args.batch_size}, epochs={args.epochs}, dtype={args.dtype}"
    )
    print(f"Models: {args.models}")
    print(f"inner_scenarios: {args.inner_scenarios}")

    for n_inner in args.inner_scenarios:
        print("\n" + "=" * 70)
        print(
            f"Running data-efficiency scenario: "
            f"nC_regions = {args.nC_regions}, "
            f"n_inner_per_region = {n_inner}"
        )
        print("=" * 70)

        cleanup_current_outputs()

        # 1) Generate data for this n_inner_per_region
        run_cmd(
            [
                PYTHON_EXE,
                "generate_data.py",
                "--nC_regions",
                str(args.nC_regions),
                "--n_inner_per_region",
                str(n_inner),
                "--seed",
                str(args.seed),
                "--out_csv",
                "data.csv",
                "--plot_file",
                f"outputs_vs_Cao_ninner_{n_inner}.png",
                "--sampling_plot_file",
                f"sampling_points_Cao_ninner_{n_inner}.png",
            ]
        )

        num_sample_points = count_rows("data.csv")
        print(f"[INFO] Number of sample points = {num_sample_points}")

        # 2) Build two-constraint linearization for the fixed Cao regions
        run_cmd([PYTHON_EXE, "linearization.py", "--nC_regions", str(args.nC_regions)])

        # Optional sanity check: exactly two constraints per region
        ab = pd.read_csv(BASE_DIR / "ABb_matrices.csv")
        counts = ab.groupby("region_id").size().to_numpy()
        if not np.all(counts == 2):
            raise RuntimeError(
                f"Expected two constraints per region, but got row counts per region: {counts}"
            )
        print(f"[INFO] ABb_matrices.csv has {len(ab)} rows = 2 constraints x {args.nC_regions} regions")

        # 3) Linearization accuracy diagnostic
        run_cmd([PYTHON_EXE, "linearization_accuracy.py"])
        lin_metrics = compute_linearization_overall()

        # 4) Projection-only diagnostic
        proj_metrics = run_projection_check(args.dtype)

        # 5) Run repeated NN/KKT-hPINN experiments with the selected solver settings
        env = os.environ.copy()
        env["SCENARIO_ID"] = f"ninner_{n_inner}"
        env["NUM_ITERATIONS"] = str(args.num_iterations)
        env["PYTHON_EXE"] = PYTHON_EXE
        env["MODEL_ID"] = args.model_id
        env["MAIN_OPTIMIZER"] = args.optimizer
        env["MAIN_LR"] = str(args.lr)
        env["MAIN_BATCH_SIZE"] = str(args.batch_size)
        env["MAIN_EPOCHS"] = str(args.epochs)
        env["MAIN_DTYPE"] = str(args.dtype)
        env["VAL_RATIO"] = str(args.val_ratio)
        env["MODELS_TO_RUN"] = ",".join(args.models)

        run_cmd([PYTHON_EXE, "experiment2.py"], env=env)

        # 6) Save scenario-specific files
        archive_scenario_files(n_inner)

        # 7) Save scenario diagnostics row
        row = {
            "n_inner_per_region": n_inner,
            "nC_regions": args.nC_regions,
            "num_regions": args.nC_regions,
            "num_sample_points": num_sample_points,
            "optimizer": args.optimizer,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "dtype": args.dtype,
            "models": ",".join(args.models),
        }
        row.update(lin_metrics)
        row.update(proj_metrics)
        diagnostic_rows.append(row)

        pd.DataFrame(diagnostic_rows).to_csv("scenario_diagnostics.csv", index=False)
        print("Updated scenario_diagnostics.csv")

    print("\nAll data-efficiency scenarios finished.")
    print("Saved scenario_diagnostics.csv")


if __name__ == "__main__":
    main()
