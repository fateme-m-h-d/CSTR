import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import sem, t

from src.config import INNER_SCENARIOS, N_C_REGIONS, N_REPEATS, SEED

BASE_DIR = Path.cwd()
SUMMARY_PATH = BASE_DIR / "metric_summary_by_samples.csv"

EXPERIMENT_METRICS = [
    ("RMSE", "NN_Experiment_RMSE", "KKThPINN_Experiment_RMSE"),
    ("PL_Violation", "NN_Experiment_VIOL", "KKThPINN_Experiment_VIOL"),
    (
        "Original_Nonlinear_Violation",
        "NN_Experiment_VIOL_NL",
        "KKThPINN_Experiment_VIOL_NL",
    ),
    (
        "Experiment_Time_sec",
        "NN_Experiment_Time_sec",
        "KKThPINN_Experiment_Time_sec",
    ),
]
TRAINING_METRICS = [
    ("Training_Error", "NN_Training_Error", "KKThPINN_Training_Error"),
    (
        "Training_Time_sec",
        "NN_Training_Time_sec",
        "KKThPINN_Training_Time_sec",
    ),
]
FINAL_GENERATED_FILES = [
    "data.csv",
    "ABb_matrices.csv",
    "lin_params.csv",
    "region_edges.npz",
    "failed_points_fixed_data.csv",
]

SCRATCH_FILES = [
    "training_epoch_errors.csv",
    "experiment_epoch_errors.csv",
    "linearization_accuracy_detailed.csv",
    "linearization_accuracy_summary.csv",
]


def run(command, env=None, capture=False):
    return subprocess.run(
        list(map(str, command)),
        cwd=BASE_DIR,
        env=env,
        text=True,
        capture_output=capture,
        check=True,
    )


def mean_ci(values, confidence=0.95):
    values = np.asarray(pd.Series(values).dropna().to_numpy(), dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, 0.0
    halfwidth = float(
        sem(values) * t.ppf((1.0 + confidence) / 2.0, len(values) - 1)
    )
    return mean, halfwidth


def add_metric_summary(row, frame, metric_name, nn_column, kkt_column):
    nn_mean, nn_ci = mean_ci(frame[nn_column])
    kkt_mean, kkt_ci = mean_ci(frame[kkt_column])
    row[f"NN_{metric_name}_mean"] = nn_mean
    row[f"NN_{metric_name}_ci95"] = nn_ci
    row[f"KKThPINN_{metric_name}_mean"] = kkt_mean
    row[f"KKThPINN_{metric_name}_ci95"] = kkt_ci


def linearization_metrics():
    detailed = pd.read_csv("linearization_accuracy_detailed.csv")
    return {
        "linearization_mean_abs_error": float(
            detailed["abs_linearization_error"].mean()
        ),
        "linearization_rmse_error": float(
            np.sqrt(np.mean(detailed["linearization_error"] ** 2))
        ),
        "linearization_max_abs_error": float(
            detailed["abs_linearization_error"].max()
        ),
    }


def projection_metrics(output):
    results = {}
    pattern = re.compile(r"^([A-Za-z0-9_]+):\s*([-+0-9.eE]+)")
    for line in output.splitlines():
        match = pattern.match(line.strip())
        if match:
            key, value = match.groups()
            results[key] = float(value)
    return results


def remove_scratch_files():
    for name in SCRATCH_FILES:
        path = BASE_DIR / name
        if path.exists():
            path.unlink()


def main():
    rows = []

    for n_inner in INNER_SCENARIOS:
        print(f"Running n_inner_per_region={n_inner}")
        run([
            sys.executable,
            "-m", "src.generate_data",
            "--nC_regions", N_C_REGIONS,
            "--n_inner_per_region", n_inner,
            "--seed", SEED,
            "--out_csv", "data.csv",
        ])
        num_samples = len(pd.read_csv(BASE_DIR / "data.csv"))

        run([
            sys.executable,
            "-m", "src.linearization",
            "--nC_regions", N_C_REGIONS,
        ])
        run([sys.executable, "-m", "diagnostics.linearization_accuracy"])
        projection = run([
            sys.executable,
            "-m", "diagnostics.projection_check",
            "--model", "KKThPINN",
            "--model_id", "projection_check",
            "--input_dim", "1",
            "--z0_dim", "3",
            "--dataset_type", "cstr",
            "--dataset_path", "data.csv",
            "--job", "projection_check",
            "--dtype", "64",
        ], capture=True)

        env = os.environ.copy()
        env["SCENARIO_ID"] = f"ninner_{n_inner}"
        env["NUM_ITERATIONS"] = str(N_REPEATS)
        env["PYTHON_EXE"] = sys.executable
        run([sys.executable, "-m", "scripts.experiment2"], env=env)

        experiment = pd.read_csv(BASE_DIR / "experiment_epoch_errors.csv")
        training = pd.read_csv(BASE_DIR / "training_epoch_errors.csv")
        row = {
            "n_inner_per_region": n_inner,
            "nC_regions": N_C_REGIONS,
            "num_regions": N_C_REGIONS,
            "num_sample_points": num_samples,
        }
        for metric in EXPERIMENT_METRICS:
            add_metric_summary(row, experiment, *metric)
        for metric in TRAINING_METRICS:
            add_metric_summary(row, training, *metric)
        row.update(linearization_metrics())
        row.update(projection_metrics(projection.stdout))
        rows.append(row)

        pd.DataFrame(rows).to_csv(SUMMARY_PATH, index=False)
        remove_scratch_files()

    for name in FINAL_GENERATED_FILES:
        path = BASE_DIR / name
        if path.exists():
            path.unlink()

    print(f"Saved {SUMMARY_PATH.name}")


if __name__ == "__main__":
    main()
