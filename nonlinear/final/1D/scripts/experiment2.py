import ast
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path.cwd()
TRAINING_CSV_PATH = BASE_DIR / "training_epoch_errors.csv"
EXPERIMENT_CSV_PATH = BASE_DIR / "experiment_epoch_errors.csv"

SCENARIO_ID = os.environ.get("SCENARIO_ID", "default")
NUM_ITERATIONS = int(os.environ.get("NUM_ITERATIONS", "50"))
PYTHON_EXE = os.environ.get("PYTHON_EXE", sys.executable)


def extract_last_epoch_error(output):
    for line in reversed(output.splitlines()):
        if line.startswith("epoch:"):
            parts = line.split()
            values = {}
            i = 0
            while i < len(parts):
                if parts[i].endswith(":") and i + 1 < len(parts):
                    values[parts[i].rstrip(":")] = parts[i + 1]
                    i += 2
                else:
                    i += 1
            return float(values.get("loss_train", np.nan))
    return np.nan


def extract_named_time(output, label):
    for line in output.splitlines():
        if line.strip().startswith(label):
            try:
                return float(line.split(":", 1)[1].replace("s", "").strip())
            except ValueError:
                return np.nan
    return np.nan


def extract_experiment_scores(output):
    for line in reversed(output.splitlines()):
        text = line.strip()
        if text.startswith("{") and text.endswith("}"):
            scores = ast.literal_eval(text)
            return {
                "rmse_total": float(scores.get("rmse_total", np.nan)),
                "violation": float(scores.get("violation", np.nan)),
                "violation_original_nonlinear": float(scores.get("violation_original_nonlinear", np.nan)),
            }
    return {
        "rmse_total": np.nan,
        "violation": np.nan,
        "violation_original_nonlinear": np.nan,
    }


def run_main(model_name, job, run_idx):
    args = [
        PYTHON_EXE, "-m", "src.main",
        "--model", model_name,
        "--model_id", "MODELID",
        "--dataset_type", "cstr",
        "--dataset_path", "data.csv",
        "--job", job,
        "--dtype", "64",
        "--run", str(run_idx),
    ]

    start = time.perf_counter()
    result = subprocess.run(args, capture_output=True, text=True, cwd=BASE_DIR)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"src.main failed for model={model_name}, job={job}, run={run_idx}")

    if job == "train":
        return {
            "loss_train": extract_last_epoch_error(result.stdout),
            "train_time_sec": elapsed,
        }

    scores = extract_experiment_scores(result.stdout)
    eval_time = extract_named_time(result.stdout, "Evaluation time")
    scores["experiment_time_sec"] = eval_time if not np.isnan(eval_time) else elapsed
    return scores


def run_model_experiments(model_name, num_iterations):
    training_errors, training_times = [], []
    experiment_rmse, experiment_viol, experiment_viol_nl, experiment_times = [], [], [], []

    for run_idx in range(1, num_iterations + 1):
        print(f"{model_name} run {run_idx}/{num_iterations}")

        train_res = run_main(model_name, "train", run_idx)
        training_errors.append(float(train_res.get("loss_train", np.nan)))
        training_times.append(float(train_res.get("train_time_sec", np.nan)))

        scores = run_main(model_name, "experiment", run_idx)
        experiment_rmse.append(float(scores.get("rmse_total", np.nan)))
        experiment_viol.append(float(scores.get("violation", np.nan)))
        experiment_viol_nl.append(float(scores.get("violation_original_nonlinear", np.nan)))
        experiment_times.append(float(scores.get("experiment_time_sec", np.nan)))

    return {
        "training_errors": training_errors,
        "training_times": training_times,
        "experiment_rmse": experiment_rmse,
        "experiment_viol": experiment_viol,
        "experiment_viol_nl": experiment_viol_nl,
        "experiment_times": experiment_times,
    }


def main():
    print(f"scenario_id={SCENARIO_ID}, num_iterations={NUM_ITERATIONS}")
    nn_stats = run_model_experiments("NN", NUM_ITERATIONS)
    kkt_stats = run_model_experiments("KKThPINN", NUM_ITERATIONS)

    pd.DataFrame({
        "Iteration": range(1, NUM_ITERATIONS + 1),
        "NN_Training_Error": nn_stats["training_errors"],
        "NN_Training_Time_sec": nn_stats["training_times"],
        "KKThPINN_Training_Error": kkt_stats["training_errors"],
        "KKThPINN_Training_Time_sec": kkt_stats["training_times"],
    }).to_csv(TRAINING_CSV_PATH, index=False)

    pd.DataFrame({
        "NN_Experiment_RMSE": nn_stats["experiment_rmse"],
        "NN_Experiment_VIOL": nn_stats["experiment_viol"],
        "NN_Experiment_VIOL_NL": nn_stats["experiment_viol_nl"],
        "NN_Experiment_Time_sec": nn_stats["experiment_times"],
        "KKThPINN_Experiment_RMSE": kkt_stats["experiment_rmse"],
        "KKThPINN_Experiment_VIOL": kkt_stats["experiment_viol"],
        "KKThPINN_Experiment_VIOL_NL": kkt_stats["experiment_viol_nl"],
        "KKThPINN_Experiment_Time_sec": kkt_stats["experiment_times"],
    }).to_csv(EXPERIMENT_CSV_PATH, index=False)

    print(f"Finished {SCENARIO_ID}")


if __name__ == "__main__":
    main()
