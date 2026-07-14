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
# Fixed number of Cao regions:
#     FIXED_NC_REGIONS = 30
#
# Vary:
#     n_inner_per_region
#
# For each scenario:
#   1) generate_data.py
#   2) linearization.py
#   3) linearization_accuracy.py
#   4) projection_check.py
#   5) experiment2.py for NN and KKThPINN, repeated NUM_ITERATIONS times
# ============================================================

INNER_SCENARIOS = [0, 1, 2, 5, 10, 15, 20, 25]


FIXED_NC_REGIONS = 30
SEED = 0
NUM_ITERATIONS = 50

PYTHON_EXE = sys.executable
BASE_DIR = Path.cwd()


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
    """
    Remove files that are regenerated in each scenario.
    This prevents accidentally copying stale files from a previous scenario.
    """

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
    """
    Read linearization_accuracy_detailed.csv and summarize the overall
    linearization error for the current scenario.
    """

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
        out["linearization_mean_abs_error_region_average"] = float(
            summary["mean_abs_linearization_error"].mean()
        )
        out["linearization_rmse_error_region_average"] = float(
            summary["rmse_linearization_error"].mean()
        )

    return out


def parse_projection_check_stdout(stdout):
    """
    projection_check.py prints lines like:
        projection_check_output_MAE_scaled: 1.23e-6

    This function extracts those values into a dictionary.
    """

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


def run_projection_check():
    """
    Run the projection-only diagnostic for the current data.csv,
    ABb_matrices.csv, and region_edges.npz.
    """

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
            "64",
        ],
        capture=True,
    )

    return parse_projection_check_stdout(result.stdout)


def archive_scenario_files(n_inner):
    """
    Save scenario-specific copies of all important files.
    """

    # Data files
    copy_if_exists("data.csv", f"data_ninner_{n_inner}.csv")
    copy_if_exists("data_with_region_info.csv", f"data_with_region_info_ninner_{n_inner}.csv")

    # Core linearization files
    copy_if_exists("ABb_matrices.csv", f"ABb_matrices_ninner_{n_inner}.csv")
    copy_if_exists("lin_params.csv", f"lin_params_ninner_{n_inner}.csv")
    copy_if_exists("region_edges.npz", f"region_edges_ninner_{n_inner}.npz")
    copy_if_exists("sampling_region_edges.npz", f"sampling_region_edges_ninner_{n_inner}.npz")

    # Diagnostics
    copy_if_exists(
        "linearization_accuracy_detailed.csv",
        f"linearization_accuracy_detailed_ninner_{n_inner}.csv",
    )
    copy_if_exists(
        "linearization_accuracy_summary.csv",
        f"linearization_accuracy_summary_ninner_{n_inner}.csv",
    )

    # Repeated experiment outputs
    copy_if_exists(
        "experiment_epoch_errors.csv",
        f"experiment_epoch_errors_ninner_{n_inner}.csv",
    )
    copy_if_exists(
        "training_epoch_errors.csv",
        f"training_epoch_errors_ninner_{n_inner}.csv",
    )


def main():
    diagnostic_rows = []

    for n_inner in INNER_SCENARIOS:
        print("\n" + "=" * 70)
        print(
            f"Running data-efficiency scenario: "
            f"nC_regions = {FIXED_NC_REGIONS}, "
            f"n_inner_per_region = {n_inner}"
        )
        print("=" * 70)

        cleanup_current_outputs()

        # ----------------------------------------------------
        # 1) Generate data for this n_inner_per_region
        # ----------------------------------------------------
        run_cmd(
            [
                PYTHON_EXE,
                "generate_data.py",
                "--nC_regions",
                str(FIXED_NC_REGIONS),
                "--n_inner_per_region",
                str(n_inner),
                "--seed",
                str(SEED),
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

        # ----------------------------------------------------
        # 2) Build linearization for fixed 30 regions
        # ----------------------------------------------------
        run_cmd(
            [
                PYTHON_EXE,
                "linearization.py",
                "--nC_regions",
                str(FIXED_NC_REGIONS),
            ]
        )

        # ----------------------------------------------------
        # 3) Linearization accuracy diagnostic
        # ----------------------------------------------------
        run_cmd([PYTHON_EXE, "linearization_accuracy.py"])
        lin_metrics = compute_linearization_overall()

        # ----------------------------------------------------
        # 4) Projection-only diagnostic
        # ----------------------------------------------------
        proj_metrics = run_projection_check()

        # ----------------------------------------------------
        # 5) Run repeated NN and KKT-hPINN experiments
        # ----------------------------------------------------
        env = os.environ.copy()
        env["SCENARIO_ID"] = f"ninner_{n_inner}"
        env["NUM_ITERATIONS"] = str(NUM_ITERATIONS)
        env["PYTHON_EXE"] = PYTHON_EXE

        run_cmd([PYTHON_EXE, "experiment2.py"], env=env)

        # ----------------------------------------------------
        # 6) Save scenario-specific files
        # ----------------------------------------------------
        archive_scenario_files(n_inner)

        # ----------------------------------------------------
        # 7) Save scenario diagnostics row
        # ----------------------------------------------------
        row = {
            "n_inner_per_region": n_inner,
            "nC_regions": FIXED_NC_REGIONS,
            "num_regions": FIXED_NC_REGIONS,
            "num_sample_points": num_sample_points,
        }

        row.update(lin_metrics)
        row.update(proj_metrics)

        diagnostic_rows.append(row)

        pd.DataFrame(diagnostic_rows).to_csv(
            "scenario_diagnostics.csv",
            index=False,
        )

        print("Updated scenario_diagnostics.csv")

    print("\nAll data-efficiency scenarios finished.")
    print("Saved scenario_diagnostics.csv")


if __name__ == "__main__":
    main()