import argparse
import ast
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
WORK_DIR = BASE_DIR / "_work"
TRAINING_CSV = BASE_DIR / "training_epoch_errors.csv"
EXPERIMENT_CSV = BASE_DIR / "experiment_epoch_errors.csv"

NUM_ITERATIONS = int(os.environ.get("NUM_ITERATIONS", "50"))
PYTHON_EXE = os.environ.get("PYTHON_EXE", sys.executable)
SCENARIO_ID = os.environ.get("SCENARIO_ID", "default")
EPOCHS = int(os.environ.get("EPOCHS", "1000"))

SOURCE_FILES = ["main.py", "train.py", "models.py", "utils.py"]
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nT_regions", type=int, required=True)
    parser.add_argument("--nC_regions", type=int, default=3)
    parser.add_argument("--noise_level", type=float, default=0.05)
    parser.add_argument("--num_iterations", type=int, default=NUM_ITERATIONS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--clean_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.nT_regions, args.nC_regions, args.num_iterations, args.epochs) < 1:
        parser.error("Region counts, repetitions, and epochs must be positive.")
    if not np.isfinite(args.noise_level) or args.noise_level < 0:
        parser.error("noise_level must be finite and nonnegative.")
    args.clean_csv = args.clean_csv.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.clean_csv.is_file():
        parser.error(f"Clean CSV not found: {args.clean_csv}")
    if args.output_dir.exists():
        parser.error("output_dir already exists; choose a new directory to preserve results.")
    return args


def prepare_work_dir(run_index, args):
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir()
    for name in SOURCE_FILES:
        shutil.copy2(BASE_DIR / "src" / name, WORK_DIR / name)
    shutil.copy2(args.clean_csv, WORK_DIR / "data_clean.csv")

    # Prepare the split and this repeat's noisy measurements in the work folder.
    subprocess.run(
        [
            PYTHON_EXE, str(BASE_DIR / "src" / "prepare_noisy_data.py"),
            "--seed", str(1000 + run_index),
            "--noise_level", str(args.noise_level),
            "--nT_regions", str(args.nT_regions),
            "--nC_regions", str(args.nC_regions),
        ],
        cwd=WORK_DIR, check=True,
    )

    # The unchanged linearization.py reads this repeat's data.csv.
    subprocess.run(
        [
            PYTHON_EXE, str(BASE_DIR / "src" / "linearization.py"),
            "--nT_regions", str(args.nT_regions),
            "--nC_regions", str(args.nC_regions),
        ],
        cwd=WORK_DIR, check=True,
    )
    if run_index == 1:
        for name in ["split_indices.npz", "region_edges.npz"]:
            shutil.copy2(WORK_DIR / name, args.output_dir / name)


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
                "prediction_time_sec": float(
                    scores.get("prediction_time_sec", np.nan)
                ),
            }
    return {
        "rmse_total": np.nan,
        "violation": np.nan,
        "violation_original_nonlinear": np.nan,
        "prediction_time_sec": np.nan,
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


def run_model_experiments(model_name, experiment_args):
    results = {
        "training_errors": [],
        "training_times": [],
        "experiment_rmse": [],
        "experiment_viol": [],
        "experiment_viol_nl": [],
        "experiment_times": [],
        "prediction_times": [],
    }
    for run_index in range(1, NUM_ITERATIONS + 1):
        print(f"{model_name} run {run_index}/{NUM_ITERATIONS}")
        # Matching repetitions of the two models reproduce the same noise.
        prepare_work_dir(run_index, experiment_args)

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
        results["prediction_times"].append(scores["prediction_time_sec"])
    return results


def main():
    global WORK_DIR, TRAINING_CSV, EXPERIMENT_CSV, NUM_ITERATIONS, EPOCHS

    experiment_args = parse_arguments()
    experiment_args.output_dir.mkdir(parents=True)
    # Preserve an immutable reference copy for this condition.
    clean_copy = experiment_args.output_dir / "data_clean.csv"
    shutil.copy2(experiment_args.clean_csv, clean_copy)
    experiment_args.clean_csv = clean_copy

    WORK_DIR = experiment_args.output_dir / "_work"
    TRAINING_CSV = experiment_args.output_dir / "training_epoch_errors.csv"
    EXPERIMENT_CSV = experiment_args.output_dir / "experiment_epoch_errors.csv"
    NUM_ITERATIONS = experiment_args.num_iterations
    EPOCHS = experiment_args.epochs

    print(
        f"Regions={experiment_args.nT_regions} x {experiment_args.nC_regions}; "
        f"noise={experiment_args.noise_level:.1%}; repetitions={NUM_ITERATIONS}"
    )
    nn_results = run_model_experiments("NN", experiment_args)
    kkt_results = run_model_experiments("KKThPINN", experiment_args)

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
        "NN_Prediction_Time_sec": nn_results["prediction_times"],
        "KKThPINN_Prediction_Time_sec": kkt_results["prediction_times"],
    }).to_csv(EXPERIMENT_CSV, index=False)

    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)


if __name__ == "__main__":
    main()
