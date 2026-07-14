import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import sem, t

from src.config import INNER_SCENARIOS, N_C_REGIONS, N_REPEATS, SEED, T_EDGES

BASE_DIR = Path.cwd()
SUMMARY_PATH = BASE_DIR / "metric_summary_by_samples.csv"
METRICS = [
    ("RMSE", "NN_Experiment_RMSE", "KKThPINN_Experiment_RMSE"),
    ("PL_Violation", "NN_Experiment_VIOL", "KKThPINN_Experiment_VIOL"),
    ("Original_Nonlinear_Violation", "NN_Experiment_VIOL_NL", "KKThPINN_Experiment_VIOL_NL"),
    ("Experiment_Time_sec", "NN_Experiment_Time_sec", "KKThPINN_Experiment_Time_sec"),
]


def run(command, env=None):
    subprocess.run(
        list(map(str, command)), cwd=BASE_DIR, env=env, check=True
    )


def mean_ci(values):
    values = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    mean = float(values.mean())
    if len(values) < 2:
        return mean, 0.0
    return mean, float(sem(values) * t.ppf(0.975, len(values) - 1))


def add_summary(row, frame, name, nn_column, kkt_column):
    nn_mean, nn_ci = mean_ci(frame[nn_column])
    kkt_mean, kkt_ci = mean_ci(frame[kkt_column])
    row[f"NN_{name}_mean"] = nn_mean
    row[f"NN_{name}_ci95"] = nn_ci
    row[f"KKThPINN_{name}_mean"] = kkt_mean
    row[f"KKThPINN_{name}_ci95"] = kkt_ci


def remove_artifacts():
    for name in (
        "data.csv", "ABb_matrices.csv", "lin_params.csv", "region_edges.npz",
        "failed_points.csv", "training_epoch_errors.csv", "experiment_epoch_errors.csv",
    ):
        path = BASE_DIR / name
        if path.exists():
            path.unlink()
    for name in ("models", "data"):
        path = BASE_DIR / name
        if path.exists():
            shutil.rmtree(path)


def main():
    rows = []
    for n_inner in INNER_SCENARIOS:
        print(f"Running n_inner_per_region={n_inner}")
        run([
            sys.executable, "-m", "src.generate_data",
            "--n_inner_per_region", n_inner, "--seed", SEED,
        ])
        sample_count = len(pd.read_csv(BASE_DIR / "data.csv"))
        run([sys.executable, "-m", "src.linearization"])

        env = os.environ.copy()
        env["SCENARIO_ID"] = f"ninner_{n_inner}"
        env["NUM_ITERATIONS"] = str(N_REPEATS)
        env["PYTHON_EXE"] = sys.executable
        run([sys.executable, "-m", "scripts.experiment2"], env=env)

        experiment = pd.read_csv(BASE_DIR / "experiment_epoch_errors.csv")
        row = {
            "n_inner_per_region": n_inner,
            "nT_regions": len(T_EDGES) - 1,
            "nC_regions": N_C_REGIONS,
            "num_regions": (len(T_EDGES) - 1) * N_C_REGIONS,
            "num_sample_points": sample_count,
        }
        for metric in METRICS:
            add_summary(row, experiment, *metric)
        rows.append(row)
        pd.DataFrame(rows).to_csv(SUMMARY_PATH, index=False)
        remove_artifacts()

    print(f"Saved {SUMMARY_PATH.name}")


if __name__ == "__main__":
    main()
