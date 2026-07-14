import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import sem, t

from src.config import N_REPEATS, N_TOTAL_POINTS, SEED, SEGMENT_SCENARIOS

PYTHON_EXE = sys.executable
BASE_DIR = Path.cwd()
SUMMARY_PATH = BASE_DIR / "metric_summary_by_segments.csv"


EXPERIMENT_METRICS = [
    ("RMSE", "NN_Experiment_RMSE", "KKThPINN_Experiment_RMSE"),
    ("Experiment_Time_sec", "NN_Experiment_Time_sec", "KKThPINN_Experiment_Time_sec"),
    ("Original_Nonlinear_Violation", "NN_Experiment_VIOL_NL", "KKThPINN_Experiment_VIOL_NL"),
]


SCRATCH_FILES = [
    "training_epoch_errors.csv",
    "experiment_epoch_errors.csv",
    "linearization_accuracy_detailed.csv",
    "linearization_accuracy_summary.csv",
]


def run_cmd(cmd, *, env=None, capture=False):
    return subprocess.run(
        list(map(str, cmd)),
        cwd=BASE_DIR,
        env=env,
        text=True,
        capture_output=capture,
        check=True,
    )


def mean_ci_halfwidth(values, confidence=0.95):
    values = np.asarray(pd.Series(values).dropna().to_numpy(), dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, 0.0
    half = float(sem(values) * t.ppf((1 + confidence) / 2.0, len(values) - 1))
    return mean, half


def compute_linearization_overall():
    detailed_path = BASE_DIR / "linearization_accuracy_detailed.csv"
    if not detailed_path.exists():
        return {}

    detailed = pd.read_csv(detailed_path)
    return {
        "linearization_mean_abs_error": float(detailed["abs_linearization_error"].mean()),
        "linearization_rmse_error": float(np.sqrt(np.mean(detailed["linearization_error"] ** 2))),
        "linearization_max_abs_error": float(detailed["abs_linearization_error"].max()),
    }


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


def run_projection_check():
    result = run_cmd(
        [
            PYTHON_EXE, "-m", "diagnostics.projection_check",
            "--model", "KKThPINN",
            "--model_id", "projection_check",
            "--input_dim", "1",
            "--z0_dim", "3",
            "--dataset_type", "cstr",
            "--dataset_path", "data.csv",
            "--job", "projection_check",
            "--dtype", "64",
        ],
        capture=True,
    )
    return parse_projection_check_stdout(result.stdout)


def summarize_experiment_file(nC):
    df = pd.read_csv(BASE_DIR / "experiment_epoch_errors.csv")
    row = {"nC_regions": nC, "num_regions": nC}
    pooled_nn = {}

    for metric_name, nn_col, kkt_col in EXPERIMENT_METRICS:
        nn_values = df[nn_col] if nn_col in df else []
        nn_mean, nn_ci = mean_ci_halfwidth(nn_values)
        kkt_mean, kkt_ci = mean_ci_halfwidth(df[kkt_col] if kkt_col in df else [])
        row[f"NN_{metric_name}_mean"] = nn_mean
        row[f"NN_{metric_name}_ci95"] = nn_ci
        row[f"KKThPINN_{metric_name}_mean"] = kkt_mean
        row[f"KKThPINN_{metric_name}_ci95"] = kkt_ci
        pooled_nn[metric_name] = np.asarray(
            pd.Series(nn_values).dropna().to_numpy(), dtype=float
        )

    return row, pooled_nn


def remove_scratch_files():
    for name in SCRATCH_FILES:
        path = BASE_DIR / name
        if path.exists():
            path.unlink()


def main():
    run_cmd([
        PYTHON_EXE, "-m", "src.generate_data",
        "--n_total_points", str(N_TOTAL_POINTS),
        "--seed", str(SEED),
        "--out_csv", "data.csv",
    ])
    shutil.copy2(BASE_DIR / "data.csv", BASE_DIR / "data_fixed.csv")

    rows = []
    pooled_nn_values = {metric_name: [] for metric_name, _, _ in EXPERIMENT_METRICS}
    for nC in SEGMENT_SCENARIOS:
        print(f"Running scenario nC_regions={nC}")
        shutil.copy2(BASE_DIR / "data_fixed.csv", BASE_DIR / "data.csv")

        run_cmd([PYTHON_EXE, "-m", "src.linearization", "--nC_regions", str(nC)])
        run_cmd([PYTHON_EXE, "-m", "diagnostics.linearization_accuracy"])

        env = os.environ.copy()
        env["SCENARIO_ID"] = f"nseg_{nC}"
        env["NUM_ITERATIONS"] = str(N_REPEATS)
        env["PYTHON_EXE"] = PYTHON_EXE
        run_cmd([PYTHON_EXE, "-m", "scripts.experiment2"], env=env)

        row, scenario_nn_values = summarize_experiment_file(nC)
        for metric_name, values in scenario_nn_values.items():
            pooled_nn_values[metric_name].extend(values.tolist())

        row.update(compute_linearization_overall())
        row.update(run_projection_check())
        rows.append(row)

        summary = pd.DataFrame(rows)
        for metric_name, values in pooled_nn_values.items():
            pooled_mean, pooled_ci = mean_ci_halfwidth(values)
            summary[f"NN_{metric_name}_pooled_mean"] = pooled_mean
            summary[f"NN_{metric_name}_pooled_ci95"] = pooled_ci
        summary.to_csv(SUMMARY_PATH, index=False)
        remove_scratch_files()

    if (BASE_DIR / "data_fixed.csv").exists():
        (BASE_DIR / "data_fixed.csv").unlink()
    print(f"Saved {SUMMARY_PATH.name}")


if __name__ == "__main__":
    main()
