import ast
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path.cwd()
WORK_DIR = BASE_DIR / "_work"
TRAINING_CSV = BASE_DIR / "training_epoch_errors.csv"
EXPERIMENT_CSV = BASE_DIR / "experiment_epoch_errors.csv"

NUM_ITERATIONS = int(os.environ.get("NUM_ITERATIONS", "50"))
PYTHON_EXE = os.environ.get("PYTHON_EXE", "python")
SCENARIO_ID = os.environ.get("SCENARIO_ID", "default")
EPOCHS = int(os.environ.get("EPOCHS", "1000"))

SOURCE_FILES = ["main.py", "train.py", "models.py", "utils.py"]
ARTIFACT_FILES = ["data.csv", "ABb_matrices.csv", "region_edges.npz"]


def prepare_work_dir():
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir()
    for name in SOURCE_FILES:
        shutil.copy2(BASE_DIR / "src" / name, WORK_DIR / name)
    for name in ARTIFACT_FILES:
        shutil.copy2(BASE_DIR / name, WORK_DIR / name)


def extract_last_epoch_error(output):
    for line in reversed(output.splitlines()):
        if not line.startswith("epoch:"):
            continue
        parts = line.split()
        values = {}
        index = 0
        while index < len(parts):
            if parts[index].endswith(":") and index + 1 < len(parts):
                values[parts[index].rstrip(":")] = parts[index + 1]
                index += 2
            else:
                index += 1
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
                "violation_original_nonlinear": float(
                    scores.get("violation_original_nonlinear", np.nan)
                ),
            }
    return {
        "rmse_total": np.nan,
        "violation": np.nan,
        "violation_original_nonlinear": np.nan,
    }


def run_main(model_name, job):
    command = [
        PYTHON_EXE,
        "main.py",
        "--model", model_name,
        "--model_id", "MODELID",
        "--dataset_type", "cstr",
        "--dataset_path", "./data.csv",
        "--job", job,
        "--epochs", str(EPOCHS),
    ]
    start = time.perf_counter()
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=WORK_DIR
    )
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    if job == "train":
        return {
            "loss_train": extract_last_epoch_error(result.stdout),
            "train_time_sec": elapsed,
        }

    scores = extract_experiment_scores(result.stdout)
    eval_time = extract_named_time(result.stdout, "Evaluation time")
    scores["experiment_time_sec"] = (
        eval_time if not np.isnan(eval_time) else elapsed
    )
    return scores


def run_model_experiments(model_name):
    results = {
        "training_errors": [],
        "training_times": [],
        "experiment_rmse": [],
        "experiment_viol": [],
        "experiment_viol_nl": [],
        "experiment_times": [],
    }
    for run_index in range(1, NUM_ITERATIONS + 1):
        print(f"{model_name} run {run_index}/{NUM_ITERATIONS}")
        prepare_work_dir()
        train_result = run_main(model_name, "train")
        scores = run_main(model_name, "experiment")
        results["training_errors"].append(train_result["loss_train"])
        results["training_times"].append(train_result["train_time_sec"])
        results["experiment_rmse"].append(scores["rmse_total"])
        results["experiment_viol"].append(scores["violation"])
        results["experiment_viol_nl"].append(
            scores["violation_original_nonlinear"]
        )
        results["experiment_times"].append(scores["experiment_time_sec"])
    return results


def main():
    print(f"scenario={SCENARIO_ID}, repetitions={NUM_ITERATIONS}")
    nn_results = run_model_experiments("NN")
    kkt_results = run_model_experiments("KKThPINN")

    pd.DataFrame({
        "Iteration": range(1, NUM_ITERATIONS + 1),
        "NN_Training_Error": nn_results["training_errors"],
        "NN_Training_Time_sec": nn_results["training_times"],
        "KKThPINN_Training_Error": kkt_results["training_errors"],
        "KKThPINN_Training_Time_sec": kkt_results["training_times"],
    }).to_csv(TRAINING_CSV, index=False)

    pd.DataFrame({
        "NN_Experiment_RMSE": nn_results["experiment_rmse"],
        "NN_Experiment_VIOL": nn_results["experiment_viol"],
        "NN_Experiment_VIOL_NL": nn_results["experiment_viol_nl"],
        "NN_Experiment_Time_sec": nn_results["experiment_times"],
        "KKThPINN_Experiment_RMSE": kkt_results["experiment_rmse"],
        "KKThPINN_Experiment_VIOL": kkt_results["experiment_viol"],
        "KKThPINN_Experiment_VIOL_NL": kkt_results["experiment_viol_nl"],
        "KKThPINN_Experiment_Time_sec": kkt_results["experiment_times"],
    }).to_csv(EXPERIMENT_CSV, index=False)

    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)


if __name__ == "__main__":
    main()
