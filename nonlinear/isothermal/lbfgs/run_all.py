# import os
# import re
# import shutil
# import subprocess
# import sys
# from pathlib import Path

# import numpy as np
# import pandas as pd


# # ============================================================
# # 1D scenario driver
# # Generates one fixed data.csv, then loops over different numbers
# # of Cao segments. For each scenario, it runs:
# #   1) linearization.py
# #   2) linearization_accuracy.py
# #   3) projection_check.py
# #   4) experiment2.py for NN and KKThPINN, 50 repetitions each
# # ============================================================

# SEGMENT_SCENARIOS = [1, 2, 3, 5, 11, 30, 55, 90]
# FIXED_TOTAL_POINTS = 150
# SEED = 0
# NUM_ITERATIONS = 50
# PYTHON_EXE = sys.executable

# BASE_DIR = Path.cwd()


# def run_cmd(cmd, *, env=None, capture=False):
#     print("\n" + " ".join(map(str, cmd)))
#     result = subprocess.run(
#         list(map(str, cmd)),
#         cwd=BASE_DIR,
#         env=env,
#         text=True,
#         capture_output=capture,
#         check=True,
#     )
#     if capture:
#         if result.stdout:
#             print(result.stdout)
#         if result.stderr:
#             print(result.stderr)
#     return result


# def copy_if_exists(src, dst):
#     src = BASE_DIR / src
#     dst = BASE_DIR / dst
#     if src.exists():
#         shutil.copy2(src, dst)
#         print(f"Copied {src.name} -> {dst.name}")


# def compute_linearization_overall():
#     detailed_path = BASE_DIR / "linearization_accuracy_detailed.csv"
#     summary_path = BASE_DIR / "linearization_accuracy_summary.csv"

#     if not detailed_path.exists():
#         return {}

#     detailed = pd.read_csv(detailed_path)
#     out = {
#         "linearization_mean_abs_error": float(detailed["abs_linearization_error"].mean()),
#         "linearization_rmse_error": float(np.sqrt(np.mean(detailed["linearization_error"] ** 2))),
#         "linearization_max_abs_error": float(detailed["abs_linearization_error"].max()),
#         "mean_abs_nonlinear_residual_on_data": float(detailed["residual_nonlinear"].abs().mean()),
#         "mean_abs_linearized_residual_on_data": float(detailed["residual_linearized"].abs().mean()),
#     }

#     if summary_path.exists():
#         summary = pd.read_csv(summary_path)
#         out["linearization_mean_abs_error_region_average"] = float(summary["mean_abs_linearization_error"].mean())
#         out["linearization_rmse_error_region_average"] = float(summary["rmse_linearization_error"].mean())

#     return out


# def parse_projection_check_stdout(stdout):
#     results = {}
#     pattern = re.compile(r"^([A-Za-z0-9_]+):\s*([-+0-9.eE]+)")
#     for line in stdout.splitlines():
#         match = pattern.match(line.strip())
#         if match:
#             key, value = match.groups()
#             try:
#                 results[key] = float(value)
#             except ValueError:
#                 pass
#     return results


# def run_projection_check():
#     result = run_cmd(
#         [
#             PYTHON_EXE, "projection_check.py",
#             "--model", "KKThPINN",
#             "--model_id", "projection_check",
#             "--input_dim", "1",
#             "--z0_dim", "3",
#             "--dataset_type", "cstr",
#             "--dataset_path", "data.csv",
#             "--job", "projection_check",
#             "--dtype", "32",
#         ],
#         capture=True,
#     )
#     return parse_projection_check_stdout(result.stdout)


# def archive_scenario_files(nC):
#     # Core linearization files
#     copy_if_exists("ABb_matrices.csv", f"ABb_matrices_nseg_{nC}.csv")
#     copy_if_exists("lin_params.csv", f"lin_params_nseg_{nC}.csv")
#     copy_if_exists("region_edges.npz", f"region_edges_nseg_{nC}.npz")

#     # Diagnostics
#     copy_if_exists("linearization_accuracy_detailed.csv", f"linearization_accuracy_detailed_nseg_{nC}.csv")
#     copy_if_exists("linearization_accuracy_summary.csv", f"linearization_accuracy_summary_nseg_{nC}.csv")

#     # Repeated experiment outputs
#     copy_if_exists("experiment_epoch_errors.csv", f"experiment_epoch_errors_nseg_{nC}.csv")
#     copy_if_exists("training_epoch_errors.csv", f"training_epoch_errors_nseg_{nC}.csv")


# def main():
#     # 1) Generate one fixed dataset once.
#     run_cmd([
#         PYTHON_EXE, "generate_data.py",
#         "--n_total_points", str(FIXED_TOTAL_POINTS),
#         "--seed", str(SEED),
#         "--out_csv", "data.csv",
#         "--plot_file", "outputs_vs_Cao_fixed.png",
#     ])
#     shutil.copy2(BASE_DIR / "data.csv", BASE_DIR / "data_fixed.csv")
#     print("Copied data.csv -> data_fixed.csv")

#     diagnostic_rows = []

#     # 2) Loop over segment scenarios.
#     for nC in SEGMENT_SCENARIOS:
#         print("\n" + "=" * 70)
#         print(f"Running scenario nC_regions = {nC}")
#         print("=" * 70)

#         # Make sure each scenario starts from the same fixed database.
#         shutil.copy2(BASE_DIR / "data_fixed.csv", BASE_DIR / "data.csv")

#         run_cmd([PYTHON_EXE, "linearization.py", "--nC_regions", str(nC)])
#         run_cmd([PYTHON_EXE, "linearization_accuracy.py"])

#         lin_metrics = compute_linearization_overall()
#         proj_metrics = run_projection_check()

#         env = os.environ.copy()
#         env["SCENARIO_ID"] = f"nseg_{nC}"
#         env["NUM_ITERATIONS"] = str(NUM_ITERATIONS)
#         env["PYTHON_EXE"] = PYTHON_EXE
#         run_cmd([PYTHON_EXE, "experiment2.py"], env=env)

#         archive_scenario_files(nC)

#         row = {"nC_regions": nC, "num_regions": nC}
#         row.update(lin_metrics)
#         row.update(proj_metrics)
#         diagnostic_rows.append(row)

#         pd.DataFrame(diagnostic_rows).to_csv("scenario_diagnostics.csv", index=False)
#         print("Updated scenario_diagnostics.csv")

#     print("\nAll segment scenarios finished.")
#     print("Saved scenario_diagnostics.csv")


# if __name__ == "__main__":
#     main()


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
# 1D scenario driver
# Generates one fixed data.csv, then loops over different numbers
# of Cao segments. For each scenario, it runs:
#   1) linearization.py
#   2) linearization_accuracy.py
#   3) projection_check.py
#   4) experiment2.py for repeated NN / KKThPINN training + testing
#
# The number of Cao regions affects the piecewise linearization of the
# nonlinear reaction constraint through linearization.py. The exact linear
# mass-balance constraint is still included in every region because each
# projection branch needs the complete local ABb matrix.
# ============================================================

DEFAULT_SEGMENT_SCENARIOS = [1, 2, 3, 5, 11, 30, 55, 90]
DEFAULT_FIXED_TOTAL_POINTS = 150
DEFAULT_SEED = 0
DEFAULT_NUM_ITERATIONS = 50

BASE_DIR = Path.cwd()
PYTHON_EXE = sys.executable


def normalize_optimizer(name):
    """Match the names expected by utils.get_optimizer()."""
    key = str(name).strip().lower()
    if key == "adam":
        return "adam"
    if key == "sgd":
        return "SGD"
    if key == "lbfgs":
        return "LBFGS"
    raise ValueError("optimizer must be one of: adam, SGD, LBFGS")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 1D CSTR segment scenarios with configurable training optimizer."
    )

    parser.add_argument(
        "--segments",
        type=int,
        nargs="+",
        default=DEFAULT_SEGMENT_SCENARIOS,
        help="List of Cao segment counts used for linearizing the nonlinear constraint.",
    )
    parser.add_argument("--n_total_points", type=int, default=DEFAULT_FIXED_TOTAL_POINTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num_iterations", type=int, default=DEFAULT_NUM_ITERATIONS)

    # Training options that will be passed through experiment2.py to main.py
    parser.add_argument("--optimizer", type=str, default="LBFGS", help="adam, SGD, or LBFGS")
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=90)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dtype", type=int, default=32, choices=[32, 64])
    parser.add_argument("--model_id", type=str, default="MODELID")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["NN", "KKThPINN"],
        choices=["NN", "KKThPINN"],
        help="Models to run inside experiment2.py. Default keeps your original NN + KKThPINN comparison.",
    )

    args = parser.parse_args()
    args.optimizer = normalize_optimizer(args.optimizer)
    return args


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
            PYTHON_EXE, "projection_check.py",
            "--model", "KKThPINN",
            "--model_id", "projection_check",
            "--input_dim", "1",
            "--z0_dim", "3",
            "--dataset_type", "cstr",
            "--dataset_path", "data.csv",
            "--job", "projection_check",
            "--dtype", str(dtype),
        ],
        capture=True,
    )
    return parse_projection_check_stdout(result.stdout)


def archive_scenario_files(nC):
    # Core linearization files
    copy_if_exists("ABb_matrices.csv", f"ABb_matrices_nseg_{nC}.csv")
    copy_if_exists("lin_params.csv", f"lin_params_nseg_{nC}.csv")
    copy_if_exists("region_edges.npz", f"region_edges_nseg_{nC}.npz")

    # Diagnostics
    copy_if_exists("linearization_accuracy_detailed.csv", f"linearization_accuracy_detailed_nseg_{nC}.csv")
    copy_if_exists("linearization_accuracy_summary.csv", f"linearization_accuracy_summary_nseg_{nC}.csv")

    # Repeated experiment outputs
    copy_if_exists("experiment_epoch_errors.csv", f"experiment_epoch_errors_nseg_{nC}.csv")
    copy_if_exists("training_epoch_errors.csv", f"training_epoch_errors_nseg_{nC}.csv")


def make_experiment_env(args, nC):
    env = os.environ.copy()
    env["SCENARIO_ID"] = f"nseg_{nC}"
    env["NUM_ITERATIONS"] = str(args.num_iterations)
    env["PYTHON_EXE"] = PYTHON_EXE

    # These are read by experiment2.py and forwarded to main.py.
    env["MAIN_OPTIMIZER"] = args.optimizer
    env["MAIN_LR"] = str(args.lr)
    env["MAIN_BATCH_SIZE"] = str(args.batch_size)
    env["MAIN_EPOCHS"] = str(args.epochs)
    env["MAIN_DTYPE"] = str(args.dtype)
    env["MODEL_ID"] = args.model_id
    env["MODELS_TO_RUN"] = ",".join(args.models)
    return env


def print_settings(args):
    print("\n=== run_all settings ===")
    print(f"segments:       {args.segments}")
    print(f"n_total_points: {args.n_total_points}")
    print(f"num_iterations: {args.num_iterations}")
    print(f"models:         {args.models}")
    print(f"optimizer:      {args.optimizer}")
    print(f"lr:             {args.lr}")
    print(f"batch_size:     {args.batch_size}")
    print(f"epochs:         {args.epochs}")
    print(f"dtype:          {args.dtype}")
    print(f"model_id:       {args.model_id}")


def main():
    args = parse_args()
    print_settings(args)

    # 1) Generate one fixed dataset once.
    run_cmd([
        PYTHON_EXE, "generate_data.py",
        "--n_total_points", str(args.n_total_points),
        "--seed", str(args.seed),
        "--out_csv", "data.csv",
        "--plot_file", "outputs_vs_Cao_fixed.png",
    ])
    shutil.copy2(BASE_DIR / "data.csv", BASE_DIR / "data_fixed.csv")
    print("Copied data.csv -> data_fixed.csv")

    diagnostic_rows = []

    # 2) Loop over segment scenarios.
    for nC in args.segments:
        print("\n" + "=" * 70)
        print(f"Running scenario nC_regions = {nC}")
        print("=" * 70)

        # Make sure each scenario starts from the same fixed database.
        shutil.copy2(BASE_DIR / "data_fixed.csv", BASE_DIR / "data.csv")

        # This is the only command that changes the number of linearization segments.
        # It updates lin_params.csv, ABb_matrices.csv, and region_edges.npz.
        run_cmd([PYTHON_EXE, "linearization.py", "--nC_regions", str(nC)])
        run_cmd([PYTHON_EXE, "linearization_accuracy.py"])

        lin_metrics = compute_linearization_overall()
        proj_metrics = run_projection_check(args.dtype)

        env = make_experiment_env(args, nC)
        run_cmd([PYTHON_EXE, "experiment2.py"], env=env)

        archive_scenario_files(nC)

        row = {"nC_regions": nC, "num_regions": nC}
        row.update(lin_metrics)
        row.update(proj_metrics)
        diagnostic_rows.append(row)

        pd.DataFrame(diagnostic_rows).to_csv("scenario_diagnostics.csv", index=False)
        print("Updated scenario_diagnostics.csv")

    print("\nAll segment scenarios finished.")
    print("Saved scenario_diagnostics.csv")


if __name__ == "__main__":
    main()
