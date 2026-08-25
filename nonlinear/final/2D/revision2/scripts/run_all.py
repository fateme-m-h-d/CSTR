import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from src.config import (
    N_C_REGIONS,
    N_REPEATS,
    N_TOTAL_POINTS,
    SEED,
    SEGMENT_SCENARIOS,
)

BASE_DIR = Path.cwd()
SUMMARY_PATH = BASE_DIR / "metric_summary_by_segments.csv"

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

SCRATCH_FILES = [
    BASE_DIR / "training_epoch_errors.csv",
    BASE_DIR / "experiment_epoch_errors.csv",
    BASE_DIR / "data_fixed.csv",
]


def run(command, env=None):
    subprocess.run(
        list(map(str, command)),
        cwd=BASE_DIR,
        env=env,
        check=True,
    )


def mean_ci(values):
    values = np.asarray(pd.Series(values).dropna().to_numpy(), dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), 0.0
    interval = ttest_1samp(values, popmean=0).confidence_interval(0.95)
    return float(values.mean()), float((interval.high - interval.low) / 2)


def add_metric_summary(row, frame, metric_name, nn_column, kkt_column):
    nn_values = frame[nn_column].dropna().to_numpy()
    kkt_values = frame[kkt_column].dropna().to_numpy()
    nn_mean, nn_halfwidth = mean_ci(nn_values)
    kkt_mean, kkt_halfwidth = mean_ci(kkt_values)
    row[f"NN_{metric_name}_mean"] = nn_mean
    row[f"NN_{metric_name}_ci95"] = nn_halfwidth
    row[f"KKThPINN_{metric_name}_mean"] = kkt_mean
    row[f"KKThPINN_{metric_name}_ci95"] = kkt_halfwidth
    return nn_values


def main():
    run([
        sys.executable,
        "-m", "src.generate_data",
        "--n_total_points", N_TOTAL_POINTS,
        "--seed", SEED,
        "--out_csv", "data.csv",
    ])
    shutil.copy2(BASE_DIR / "data.csv", BASE_DIR / "data_fixed.csv")

    rows = []
    pooled_nn = {name: [] for name, _, _ in EXPERIMENT_METRICS}

    for n_t_regions in SEGMENT_SCENARIOS:
        print(
            f"Running nT_regions={n_t_regions}, "
            f"nC_regions={N_C_REGIONS}"
        )
        shutil.copy2(BASE_DIR / "data_fixed.csv", BASE_DIR / "data.csv")
        run([
            sys.executable,
            "-m", "src.linearization",
            "--nT_regions", n_t_regions,
            "--nC_regions", N_C_REGIONS,
        ])

        env = os.environ.copy()
        env["SCENARIO_ID"] = f"nseg_{n_t_regions}"
        env["NUM_ITERATIONS"] = str(N_REPEATS)
        env["PYTHON_EXE"] = sys.executable
        run([sys.executable, "-m", "scripts.experiment2"], env=env)

        experiment = pd.read_csv(BASE_DIR / "experiment_epoch_errors.csv")
        training = pd.read_csv(BASE_DIR / "training_epoch_errors.csv")
        row = {
            "nT_regions": n_t_regions,
            "nC_regions": N_C_REGIONS,
            "num_regions": n_t_regions * N_C_REGIONS,
        }

        for metric_name, nn_column, kkt_column in EXPERIMENT_METRICS:
            nn_values = add_metric_summary(
                row, experiment, metric_name, nn_column, kkt_column
            )
            pooled_nn[metric_name].extend(nn_values.tolist())

        for metric_name, nn_column, kkt_column in TRAINING_METRICS:
            add_metric_summary(
                row, training, metric_name, nn_column, kkt_column
            )
        rows.append(row)

    summary = pd.DataFrame(rows)
    for metric_name, values in pooled_nn.items():
        pooled_mean, pooled_halfwidth = mean_ci(values)
        summary[f"NN_{metric_name}_pooled_mean"] = pooled_mean
        summary[f"NN_{metric_name}_pooled_ci95"] = pooled_halfwidth
    summary.to_csv(SUMMARY_PATH, index=False)

    for path in SCRATCH_FILES:
        if path.exists():
            path.unlink()
    print(f"Saved {SUMMARY_PATH.name}")


if __name__ == "__main__":
    main()
