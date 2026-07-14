# import ast
# import glob
# import os
# import shutil
# import subprocess
# import time
# from pathlib import Path

# import numpy as np
# import pandas as pd


# # ============================================================
# # 1D repeated experiment driver
# # Runs NN and KKThPINN for the current data.csv / ABb_matrices.csv /
# # region_edges.npz that were produced by run_all.py for one nC scenario.
# # ============================================================

# BASE_DIR = Path.cwd()
# TARGET_FOLDER = BASE_DIR / "new1"
# TARGET_FOLDER.mkdir(exist_ok=True)

# # utils.py imports constants from generate_data.py, so generate_data.py
# # must also be copied into the temporary run folder.
# FILES_TO_COPY = [
#     "main.py",
#     "train.py",
#     "models.py",
#     "utils.py",
#     "generate_data.py",
#     "data.csv",
#     "ABb_matrices.csv",
#     "region_edges.npz",
# ]

# TRAINING_CSV_PATH = BASE_DIR / "training_epoch_errors.csv"
# EXPERIMENT_CSV_PATH = BASE_DIR / "experiment_epoch_errors.csv"

# ARCHIVE_ROOT = BASE_DIR / "models_archive"
# RESULTS_ARCHIVE_ROOT = BASE_DIR / "results_archive"
# ARCHIVE_ROOT.mkdir(exist_ok=True)
# RESULTS_ARCHIVE_ROOT.mkdir(exist_ok=True)

# RESULTS_MASTER_CSV = RESULTS_ARCHIVE_ROOT / "results_by_segments_master.csv"
# SOURCE_DATASET = BASE_DIR / "data.csv"
# TARGET_DATASET = TARGET_FOLDER / "data.csv"

# SCENARIO_ID = os.environ.get("SCENARIO_ID", "default")
# NUM_ITERATIONS = int(os.environ.get("NUM_ITERATIONS", "50"))
# PYTHON_EXE = os.environ.get("PYTHON_EXE", "python")


# def _copy_if_exists(src, dst):
#     src = Path(src) if src is not None else None
#     dst = Path(dst)
#     if src is not None and src.exists():
#         dst.parent.mkdir(parents=True, exist_ok=True)
#         shutil.copy2(src, dst)


# def _count_rows(csv_path):
#     try:
#         return int(pd.read_csv(csv_path).shape[0])
#     except Exception:
#         return -1


# def find_latest_model_file(root_dir):
#     root_dir = Path(root_dir)
#     newest_time = -1
#     newest_path = None
#     if not root_dir.exists():
#         return None
#     for p in root_dir.rglob("*.pth"):
#         t = p.stat().st_mtime
#         if t > newest_time:
#             newest_time = t
#             newest_path = p
#     return newest_path


# def make_archive_dir(scenario_id, model_name, run_idx):
#     ts = pd.Timestamp.utcnow().strftime("%Y%m%d-%H%M%S")
#     path = ARCHIVE_ROOT / str(scenario_id) / f"{ts}_run{run_idx:02d}_{model_name}"
#     path.mkdir(parents=True, exist_ok=True)
#     return path


# def archive_current_run(model_name, run_idx, train_err, train_time, exp_rmse, exp_viol, exp_viol_nl, exp_time):
#     n_samples_actual = _count_rows(TARGET_DATASET)
#     arch_dir = make_archive_dir(SCENARIO_ID, model_name, run_idx)

#     latest_pth = find_latest_model_file(TARGET_FOLDER / "models")
#     _copy_if_exists(latest_pth, arch_dir / "model_state.pth")
#     _copy_if_exists(TARGET_DATASET, arch_dir / "data.csv")
#     _copy_if_exists(TARGET_FOLDER / "scaler.pkl", arch_dir / "scaler.pkl")

#     code_snap = arch_dir / "code_snapshot"
#     code_snap.mkdir(exist_ok=True)
#     for file_name in FILES_TO_COPY:
#         src = BASE_DIR / file_name
#         if src.exists():
#             shutil.copy2(src, code_snap / src.name)

#     artifacts_dir = arch_dir / "artifacts"
#     artifacts_dir.mkdir(exist_ok=True)
#     for csv_path in TARGET_FOLDER.glob("*.csv"):
#         shutil.copy2(csv_path, artifacts_dir / f"run{run_idx:02d}_{csv_path.name}")

#     logs_dir = TARGET_FOLDER / "logs"
#     if logs_dir.is_dir():
#         shutil.copytree(logs_dir, arch_dir / "logs", dirs_exist_ok=True)

#     with open(arch_dir / "RUN_INFO.txt", "w", encoding="utf-8") as fp:
#         fp.write(
#             f"scenario_id={SCENARIO_ID}\n"
#             f"model={model_name}\n"
#             f"run_index={run_idx}\n"
#             f"num_samples_actual={n_samples_actual}\n"
#             f"train_error={train_err}\n"
#             f"train_time_sec={train_time}\n"
#             f"experiment_rmse_total={exp_rmse}\n"
#             f"experiment_violation={exp_viol}\n"
#             f"experiment_violation_original_nonlinear={exp_viol_nl}\n"
#             f"experiment_time_sec={exp_time}\n"
#             f"created_utc={pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
#         )
#     print(f"[ARCHIVED] {model_name} run {run_idx} => {arch_dir}")


# def copy_files():
#     for file_name in FILES_TO_COPY:
#         src = BASE_DIR / file_name
#         dst = TARGET_FOLDER / file_name
#         if src.exists():
#             shutil.copy2(src, dst)
#             print(f"Copied {file_name} to {TARGET_FOLDER}")
#         else:
#             print(f"Warning: {file_name} not found.")


# def clear_folder():
#     if not TARGET_FOLDER.exists():
#         return
#     for path in TARGET_FOLDER.iterdir():
#         try:
#             if path.is_file() or path.is_symlink():
#                 path.unlink()
#             elif path.is_dir():
#                 shutil.rmtree(path)
#         except Exception as exc:
#             print(f"Failed to delete {path}. Reason: {exc}")


# def extract_last_epoch_error(output):
#     for line in reversed(output.splitlines()):
#         if line.startswith("epoch:"):
#             parts = line.split()
#             data = {}
#             i = 0
#             while i < len(parts):
#                 if parts[i].endswith(":") and i + 1 < len(parts):
#                     data[parts[i].replace(":", "")] = parts[i + 1]
#                     i += 2
#                 else:
#                     i += 1
#             try:
#                 return float(data["loss_train"])
#             except (KeyError, ValueError):
#                 return np.nan
#     return np.nan


# def extract_named_time(output, label):
#     for line in output.splitlines():
#         s = line.strip()
#         if s.startswith(label):
#             try:
#                 return float(s.split(":", 1)[1].replace("s", "").strip())
#             except Exception:
#                 return np.nan
#     return np.nan


# def extract_experiment_scores(output):
#     rmse_val = viol_val = viol_nl_val = np.nan
#     for line in reversed(output.splitlines()):
#         s = line.strip()
#         if s.startswith("{") and s.endswith("}"):
#             try:
#                 d = ast.literal_eval(s)
#                 rmse_val = float(d.get("rmse_total", np.nan))
#                 viol_val = float(d.get("violation", np.nan))
#                 viol_nl_val = float(d.get("violation_original_nonlinear", np.nan))
#             except Exception:
#                 pass
#             break
#     return {
#         "rmse_total": rmse_val,
#         "violation": viol_val,
#         "violation_original_nonlinear": viol_nl_val,
#     }


# def run_main(model_name, job, run_idx):
#     main_file = TARGET_FOLDER / "main.py"
#     if not main_file.exists():
#         raise FileNotFoundError("main.py not found in the target folder.")

#     args = [
#         PYTHON_EXE, "main.py",
#         "--model", model_name,
#         "--model_id", "MODELID",
#         "--dataset_type", "cstr",
#         "--dataset_path", "./data.csv",
#         "--job", job,
#         "--dtype", "64",
#     ]

#     print(f"Running main.py with model={model_name}, job={job}, run={run_idx} ...")
#     t0 = time.perf_counter()
#     result = subprocess.run(args, capture_output=True, text=True, cwd=TARGET_FOLDER)
#     elapsed = time.perf_counter() - t0

#     logs_root = TARGET_FOLDER / "logs"
#     logs_root.mkdir(exist_ok=True)
#     tag = f"run{run_idx:02d}_{model_name}_{job}"
#     (logs_root / f"{tag}_stdout.txt").write_text(result.stdout, encoding="utf-8")
#     if result.stderr:
#         (logs_root / f"{tag}_stderr.txt").write_text(result.stderr, encoding="utf-8")

#     print("Standard Output:")
#     print(result.stdout)
#     if result.stderr:
#         print("Standard Error:")
#         print(result.stderr)

#     if result.returncode != 0:
#         raise RuntimeError(f"main.py failed for model={model_name}, job={job}, run={run_idx}")

#     if job == "train":
#         return {
#             "loss_train": extract_last_epoch_error(result.stdout),
#             "train_time_sec": elapsed,
#         }
#     if job == "experiment":
#         scores = extract_experiment_scores(result.stdout)
#         eval_time = extract_named_time(result.stdout, "Evaluation time")
#         scores["experiment_time_sec"] = eval_time if not np.isnan(eval_time) else elapsed
#         return scores
#     raise ValueError("job must be train or experiment")


# def run_model_experiments(model_name, num_iterations):
#     training_errors, training_times = [], []
#     experiment_rmse, experiment_viol, experiment_viol_nl, experiment_times = [], [], [], []

#     for i in range(num_iterations):
#         run_idx = i + 1
#         print(f"\n=== Iteration {run_idx}/{num_iterations} for model {model_name} ===\n")
#         copy_files()
#         print(f"[INFO] TARGET_DATASET rows for this run: {_count_rows(TARGET_DATASET)}")

#         train_res = run_main(model_name, "train", run_idx)
#         train_error = float(train_res.get("loss_train", np.nan))
#         train_time = float(train_res.get("train_time_sec", np.nan))
#         training_errors.append(train_error)
#         training_times.append(train_time)

#         sc = run_main(model_name, "experiment", run_idx)
#         rmse_val = float(sc.get("rmse_total", np.nan))
#         viol_val = float(sc.get("violation", np.nan))
#         viol_nl_val = float(sc.get("violation_original_nonlinear", np.nan))
#         exp_time = float(sc.get("experiment_time_sec", np.nan))
#         experiment_rmse.append(rmse_val)
#         experiment_viol.append(viol_val)
#         experiment_viol_nl.append(viol_nl_val)
#         experiment_times.append(exp_time)

#         archive_current_run(model_name, run_idx, train_error, train_time, rmse_val, viol_val, viol_nl_val, exp_time)

#         if i < num_iterations - 1:
#             clear_folder()

#     return {
#         "training_errors": training_errors,
#         "training_times": training_times,
#         "experiment_rmse": experiment_rmse,
#         "experiment_viol": experiment_viol,
#         "experiment_viol_nl": experiment_viol_nl,
#         "experiment_times": experiment_times,
#         "rmse_mean": float(np.nanmean(experiment_rmse)),
#         "viol_mean": float(np.nanmean(experiment_viol)),
#         "viol_nl_mean": float(np.nanmean(experiment_viol_nl)),
#         "train_time_mean": float(np.nanmean(training_times)),
#         "exp_time_mean": float(np.nanmean(experiment_times)),
#     }


# def update_results_master(model_name, scenario_id, stats):
#     if RESULTS_MASTER_CSV.exists():
#         df = pd.read_csv(RESULTS_MASTER_CSV)
#     else:
#         df = pd.DataFrame({"Model": ["NN", "KKThPINN"]})

#     for name in ["NN", "KKThPINN"]:
#         if not df["Model"].eq(name).any():
#             df = pd.concat([df, pd.DataFrame({"Model": [name]})], ignore_index=True)

#     columns = {
#         f"{scenario_id}_RMSE_TOTAL": stats["rmse_mean"],
#         f"{scenario_id}_VIOL": stats["viol_mean"],
#         f"{scenario_id}_VIOL_NL": stats["viol_nl_mean"],
#         f"{scenario_id}_TRAIN_TIME_SEC": stats["train_time_mean"],
#         f"{scenario_id}_EXP_TIME_SEC": stats["exp_time_mean"],
#     }
#     for col in columns:
#         if col not in df.columns:
#             df[col] = np.nan
#     df.loc[df["Model"].eq(model_name), list(columns.keys())] = list(columns.values())
#     df.to_csv(RESULTS_MASTER_CSV, index=False)


# def archive_final_csvs():
#     ts = pd.Timestamp.utcnow().strftime("%Y%m%d-%H%M%S")
#     final_dir = RESULTS_ARCHIVE_ROOT / str(SCENARIO_ID) / ts
#     final_dir.mkdir(parents=True, exist_ok=True)
#     for p in [TRAINING_CSV_PATH, EXPERIMENT_CSV_PATH, RESULTS_MASTER_CSV]:
#         _copy_if_exists(p, final_dir / Path(p).name)
#     print(f"[RESULTS] Final CSVs archived at: {final_dir}")


# def main():
#     print(f"[INFO] scenario_id={SCENARIO_ID}, num_iterations={NUM_ITERATIONS}")
#     print(f"[INFO] SOURCE rows: {_count_rows(SOURCE_DATASET)} | TARGET rows before start: {_count_rows(TARGET_DATASET)}")

#     clear_folder()

#     print("\n******** Running experiments for NN ********\n")
#     nn_stats = run_model_experiments("NN", NUM_ITERATIONS)
#     clear_folder()

#     print("\n******** Running experiments for KKThPINN ********\n")
#     kkt_stats = run_model_experiments("KKThPINN", NUM_ITERATIONS)

#     pd.DataFrame({
#         "Iteration": range(1, NUM_ITERATIONS + 1),
#         "NN_Training_Error": nn_stats["training_errors"],
#         "NN_Training_Time_sec": nn_stats["training_times"],
#         "KKThPINN_Training_Error": kkt_stats["training_errors"],
#         "KKThPINN_Training_Time_sec": kkt_stats["training_times"],
#     }).to_csv(TRAINING_CSV_PATH, index=False)
#     print(f"\nTraining errors saved at: {TRAINING_CSV_PATH}")

#     pd.DataFrame({
#         "NN_Experiment_RMSE": nn_stats["experiment_rmse"],
#         "NN_Experiment_VIOL": nn_stats["experiment_viol"],
#         "NN_Experiment_VIOL_NL": nn_stats["experiment_viol_nl"],
#         "NN_Experiment_Time_sec": nn_stats["experiment_times"],
#         "KKThPINN_Experiment_RMSE": kkt_stats["experiment_rmse"],
#         "KKThPINN_Experiment_VIOL": kkt_stats["experiment_viol"],
#         "KKThPINN_Experiment_VIOL_NL": kkt_stats["experiment_viol_nl"],
#         "KKThPINN_Experiment_Time_sec": kkt_stats["experiment_times"],
#     }).to_csv(EXPERIMENT_CSV_PATH, index=False)
#     print(f"Experiment errors saved at: {EXPERIMENT_CSV_PATH}")

#     update_results_master("NN", SCENARIO_ID, nn_stats)
#     update_results_master("KKThPINN", SCENARIO_ID, kkt_stats)
#     archive_final_csvs()

#     print("\n=== Scenario means ===")
#     print(
#         f"{SCENARIO_ID} | NN: train_time={nn_stats['train_time_mean']:.4f}s, "
#         f"exp_time={nn_stats['exp_time_mean']:.4f}s, RMSE={nn_stats['rmse_mean']:.6e}, "
#         f"VIOL={nn_stats['viol_mean']:.6e}, VIOL_NL={nn_stats['viol_nl_mean']:.6e}"
#     )
#     print(
#         f"{SCENARIO_ID} | KKT: train_time={kkt_stats['train_time_mean']:.4f}s, "
#         f"exp_time={kkt_stats['exp_time_mean']:.4f}s, RMSE={kkt_stats['rmse_mean']:.6e}, "
#         f"VIOL={kkt_stats['viol_mean']:.6e}, VIOL_NL={kkt_stats['viol_nl_mean']:.6e}"
#     )


# if __name__ == "__main__":
#     main()


import ast
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1D repeated experiment driver
# Runs selected models for the current data.csv / ABb_matrices.csv /
# region_edges.npz that were produced by run_all.py for one nC scenario.
#
# Training settings are received from run_all.py through environment variables:
#   MAIN_OPTIMIZER, MAIN_LR, MAIN_BATCH_SIZE, MAIN_EPOCHS, MAIN_DTYPE, MODEL_ID
# ============================================================

BASE_DIR = Path.cwd()
TARGET_FOLDER = BASE_DIR / "new1"
TARGET_FOLDER.mkdir(exist_ok=True)

# utils.py imports constants from generate_data.py, so generate_data.py
# must also be copied into the temporary run folder.
FILES_TO_COPY = [
    "main.py",
    "train.py",
    "models.py",
    "utils.py",
    "generate_data.py",
    "data.csv",
    "ABb_matrices.csv",
    "region_edges.npz",
]

TRAINING_CSV_PATH = BASE_DIR / "training_epoch_errors.csv"
EXPERIMENT_CSV_PATH = BASE_DIR / "experiment_epoch_errors.csv"

ARCHIVE_ROOT = BASE_DIR / "models_archive"
RESULTS_ARCHIVE_ROOT = BASE_DIR / "results_archive"
ARCHIVE_ROOT.mkdir(exist_ok=True)
RESULTS_ARCHIVE_ROOT.mkdir(exist_ok=True)

RESULTS_MASTER_CSV = RESULTS_ARCHIVE_ROOT / "results_by_segments_master.csv"
SOURCE_DATASET = BASE_DIR / "data.csv"
TARGET_DATASET = TARGET_FOLDER / "data.csv"

SCENARIO_ID = os.environ.get("SCENARIO_ID", "default")
NUM_ITERATIONS = int(os.environ.get("NUM_ITERATIONS", "50"))
PYTHON_EXE = os.environ.get("PYTHON_EXE", "python")

MODEL_ID = os.environ.get("MODEL_ID", "MODELID")
MAIN_OPTIMIZER = os.environ.get("MAIN_OPTIMIZER", "adam")
MAIN_LR = float(os.environ.get("MAIN_LR", "1e-4"))
MAIN_BATCH_SIZE = int(os.environ.get("MAIN_BATCH_SIZE", "16"))
MAIN_EPOCHS = int(os.environ.get("MAIN_EPOCHS", "1000"))
MAIN_DTYPE = int(os.environ.get("MAIN_DTYPE", "64"))
VAL_RATIO = float(os.environ.get("VAL_RATIO", "0.2"))
MODELS_TO_RUN = [m.strip() for m in os.environ.get("MODELS_TO_RUN", "NN,KKThPINN").split(",") if m.strip()]


def _copy_if_exists(src, dst):
    src = Path(src) if src is not None else None
    dst = Path(dst)
    if src is not None and src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _count_rows(csv_path):
    try:
        return int(pd.read_csv(csv_path).shape[0])
    except Exception:
        return -1


def find_latest_model_file(root_dir):
    root_dir = Path(root_dir)
    newest_time = -1
    newest_path = None
    if not root_dir.exists():
        return None
    for p in root_dir.rglob("*.pth"):
        t = p.stat().st_mtime
        if t > newest_time:
            newest_time = t
            newest_path = p
    return newest_path


def make_archive_dir(scenario_id, model_name, run_idx):
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d-%H%M%S")
    path = ARCHIVE_ROOT / str(scenario_id) / f"{ts}_run{run_idx:02d}_{model_name}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def archive_current_run(model_name, run_idx, train_err, train_time, exp_rmse, exp_viol, exp_viol_nl, exp_time):
    n_samples_actual = _count_rows(TARGET_DATASET)
    arch_dir = make_archive_dir(SCENARIO_ID, model_name, run_idx)

    latest_pth = find_latest_model_file(TARGET_FOLDER / "models")
    _copy_if_exists(latest_pth, arch_dir / "model_state.pth")
    _copy_if_exists(TARGET_DATASET, arch_dir / "data.csv")
    _copy_if_exists(TARGET_FOLDER / "scaler.pkl", arch_dir / "scaler.pkl")

    code_snap = arch_dir / "code_snapshot"
    code_snap.mkdir(exist_ok=True)
    for file_name in FILES_TO_COPY:
        src = BASE_DIR / file_name
        if src.exists():
            shutil.copy2(src, code_snap / src.name)

    artifacts_dir = arch_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    for csv_path in TARGET_FOLDER.glob("*.csv"):
        shutil.copy2(csv_path, artifacts_dir / f"run{run_idx:02d}_{csv_path.name}")

    logs_dir = TARGET_FOLDER / "logs"
    if logs_dir.is_dir():
        shutil.copytree(logs_dir, arch_dir / "logs", dirs_exist_ok=True)

    with open(arch_dir / "RUN_INFO.txt", "w", encoding="utf-8") as fp:
        fp.write(
            f"scenario_id={SCENARIO_ID}\n"
            f"model={model_name}\n"
            f"run_index={run_idx}\n"
            f"num_samples_actual={n_samples_actual}\n"
            f"optimizer={MAIN_OPTIMIZER}\n"
            f"lr={MAIN_LR}\n"
            f"batch_size={MAIN_BATCH_SIZE}\n"
            f"epochs={MAIN_EPOCHS}\n"
            f"dtype={MAIN_DTYPE}\n"
            f"model_id={MODEL_ID}\n"
            f"train_error={train_err}\n"
            f"train_time_sec={train_time}\n"
            f"experiment_rmse_total={exp_rmse}\n"
            f"experiment_violation={exp_viol}\n"
            f"experiment_violation_original_nonlinear={exp_viol_nl}\n"
            f"experiment_time_sec={exp_time}\n"
            f"created_utc={pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
        )
    print(f"[ARCHIVED] {model_name} run {run_idx} => {arch_dir}")


def copy_files():
    for file_name in FILES_TO_COPY:
        src = BASE_DIR / file_name
        dst = TARGET_FOLDER / file_name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied {file_name} to {TARGET_FOLDER}")
        else:
            print(f"Warning: {file_name} not found.")


def clear_folder():
    if not TARGET_FOLDER.exists():
        return
    for path in TARGET_FOLDER.iterdir():
        try:
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
        except Exception as exc:
            print(f"Failed to delete {path}. Reason: {exc}")


def extract_last_epoch_error(output):
    # This only works when train.py prints epoch lines. For epochs < 50,
    # we fall back to the saved learning-curve .npy file below.
    for line in reversed(output.splitlines()):
        if line.startswith("epoch:"):
            parts = line.split()
            data = {}
            i = 0
            while i < len(parts):
                if parts[i].endswith(":") and i + 1 < len(parts):
                    data[parts[i].replace(":", "")] = parts[i + 1]
                    i += 2
                else:
                    i += 1
            try:
                return float(data["loss_train"])
            except (KeyError, ValueError):
                return np.nan
    return np.nan


def extract_training_loss_from_history(model_name):
    curve_path = (
        TARGET_FOLDER
        / "data"
        / "learning_curves"
        / "cstr"
        / model_name
        / str(VAL_RATIO)
        / f"{MODEL_ID}_train_losses_run0.npy"
    )
    if not curve_path.exists():
        return np.nan
    try:
        values = np.load(curve_path)
        if len(values) == 0:
            return np.nan
        return float(values[-1])
    except Exception:
        return np.nan


def extract_named_time(output, label):
    for line in output.splitlines():
        s = line.strip()
        if s.startswith(label):
            try:
                return float(s.split(":", 1)[1].replace("s", "").strip())
            except Exception:
                return np.nan
    return np.nan


def extract_experiment_scores(output):
    rmse_val = viol_val = viol_nl_val = np.nan
    for line in reversed(output.splitlines()):
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                d = ast.literal_eval(s)
                rmse_val = float(d.get("rmse_total", np.nan))
                viol_val = float(d.get("violation", np.nan))
                viol_nl_val = float(d.get("violation_original_nonlinear", np.nan))
            except Exception:
                pass
            break
    return {
        "rmse_total": rmse_val,
        "violation": viol_val,
        "violation_original_nonlinear": viol_nl_val,
    }


def build_main_args(model_name, job):
    return [
        PYTHON_EXE, "main.py",
        "--model", model_name,
        "--model_id", MODEL_ID,
        "--dataset_type", "cstr",
        "--dataset_path", "./data.csv",
        "--job", job,
        "--optimizer", MAIN_OPTIMIZER,
        "--lr", str(MAIN_LR),
        "--batch_size", str(MAIN_BATCH_SIZE),
        "--epochs", str(MAIN_EPOCHS),
        "--dtype", str(MAIN_DTYPE),
    ]


def run_main(model_name, job, run_idx):
    main_file = TARGET_FOLDER / "main.py"
    if not main_file.exists():
        raise FileNotFoundError("main.py not found in the target folder.")

    args = build_main_args(model_name, job)

    print(f"Running main.py with model={model_name}, job={job}, run={run_idx} ...")
    print("Command:", " ".join(map(str, args)))
    t0 = time.perf_counter()
    result = subprocess.run(args, capture_output=True, text=True, cwd=TARGET_FOLDER)
    elapsed = time.perf_counter() - t0

    logs_root = TARGET_FOLDER / "logs"
    logs_root.mkdir(exist_ok=True)
    tag = f"run{run_idx:02d}_{model_name}_{job}"
    (logs_root / f"{tag}_stdout.txt").write_text(result.stdout, encoding="utf-8")
    if result.stderr:
        (logs_root / f"{tag}_stderr.txt").write_text(result.stderr, encoding="utf-8")

    print("Standard Output:")
    print(result.stdout)
    if result.stderr:
        print("Standard Error:")
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"main.py failed for model={model_name}, job={job}, run={run_idx}")

    if job == "train":
        loss_train = extract_last_epoch_error(result.stdout)
        if np.isnan(loss_train):
            loss_train = extract_training_loss_from_history(model_name)
        return {
            "loss_train": loss_train,
            "train_time_sec": elapsed,
        }
    if job == "experiment":
        scores = extract_experiment_scores(result.stdout)
        eval_time = extract_named_time(result.stdout, "Evaluation time")
        scores["experiment_time_sec"] = eval_time if not np.isnan(eval_time) else elapsed
        return scores
    raise ValueError("job must be train or experiment")


def run_model_experiments(model_name, num_iterations):
    training_errors, training_times = [], []
    experiment_rmse, experiment_viol, experiment_viol_nl, experiment_times = [], [], [], []

    for i in range(num_iterations):
        run_idx = i + 1
        print(f"\n=== Iteration {run_idx}/{num_iterations} for model {model_name} ===\n")
        copy_files()
        print(f"[INFO] TARGET_DATASET rows for this run: {_count_rows(TARGET_DATASET)}")

        train_res = run_main(model_name, "train", run_idx)
        train_error = float(train_res.get("loss_train", np.nan))
        train_time = float(train_res.get("train_time_sec", np.nan))
        training_errors.append(train_error)
        training_times.append(train_time)

        sc = run_main(model_name, "experiment", run_idx)
        rmse_val = float(sc.get("rmse_total", np.nan))
        viol_val = float(sc.get("violation", np.nan))
        viol_nl_val = float(sc.get("violation_original_nonlinear", np.nan))
        exp_time = float(sc.get("experiment_time_sec", np.nan))
        experiment_rmse.append(rmse_val)
        experiment_viol.append(viol_val)
        experiment_viol_nl.append(viol_nl_val)
        experiment_times.append(exp_time)

        archive_current_run(model_name, run_idx, train_error, train_time, rmse_val, viol_val, viol_nl_val, exp_time)

        if i < num_iterations - 1:
            clear_folder()

    return {
        "training_errors": training_errors,
        "training_times": training_times,
        "experiment_rmse": experiment_rmse,
        "experiment_viol": experiment_viol,
        "experiment_viol_nl": experiment_viol_nl,
        "experiment_times": experiment_times,
        "rmse_mean": float(np.nanmean(experiment_rmse)),
        "viol_mean": float(np.nanmean(experiment_viol)),
        "viol_nl_mean": float(np.nanmean(experiment_viol_nl)),
        "train_time_mean": float(np.nanmean(training_times)),
        "exp_time_mean": float(np.nanmean(experiment_times)),
    }


def update_results_master(model_name, scenario_id, stats):
    if RESULTS_MASTER_CSV.exists():
        df = pd.read_csv(RESULTS_MASTER_CSV)
    else:
        df = pd.DataFrame({"Model": ["NN", "KKThPINN"]})

    for name in ["NN", "KKThPINN"]:
        if not df["Model"].eq(name).any():
            df = pd.concat([df, pd.DataFrame({"Model": [name]})], ignore_index=True)

    columns = {
        f"{scenario_id}_RMSE_TOTAL": stats["rmse_mean"],
        f"{scenario_id}_VIOL": stats["viol_mean"],
        f"{scenario_id}_VIOL_NL": stats["viol_nl_mean"],
        f"{scenario_id}_TRAIN_TIME_SEC": stats["train_time_mean"],
        f"{scenario_id}_EXP_TIME_SEC": stats["exp_time_mean"],
    }
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    df.loc[df["Model"].eq(model_name), list(columns.keys())] = list(columns.values())
    df.to_csv(RESULTS_MASTER_CSV, index=False)


def archive_final_csvs():
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d-%H%M%S")
    final_dir = RESULTS_ARCHIVE_ROOT / str(SCENARIO_ID) / ts
    final_dir.mkdir(parents=True, exist_ok=True)
    for p in [TRAINING_CSV_PATH, EXPERIMENT_CSV_PATH, RESULTS_MASTER_CSV]:
        _copy_if_exists(p, final_dir / Path(p).name)
    print(f"[RESULTS] Final CSVs archived at: {final_dir}")


def stats_or_empty(all_stats, model_name):
    if model_name in all_stats:
        return all_stats[model_name]
    n = NUM_ITERATIONS
    return {
        "training_errors": [np.nan] * n,
        "training_times": [np.nan] * n,
        "experiment_rmse": [np.nan] * n,
        "experiment_viol": [np.nan] * n,
        "experiment_viol_nl": [np.nan] * n,
        "experiment_times": [np.nan] * n,
    }


def main():
    print(f"[INFO] scenario_id={SCENARIO_ID}, num_iterations={NUM_ITERATIONS}")
    print(f"[INFO] SOURCE rows: {_count_rows(SOURCE_DATASET)} | TARGET rows before start: {_count_rows(TARGET_DATASET)}")
    print(
        "[INFO] main.py settings: "
        f"optimizer={MAIN_OPTIMIZER}, lr={MAIN_LR}, batch_size={MAIN_BATCH_SIZE}, "
        f"epochs={MAIN_EPOCHS}, dtype={MAIN_DTYPE}, model_id={MODEL_ID}"
    )
    print(f"[INFO] models_to_run={MODELS_TO_RUN}")

    clear_folder()

    all_stats = {}
    for model_name in MODELS_TO_RUN:
        if model_name not in ["NN", "KKThPINN"]:
            raise ValueError(f"Unsupported model in MODELS_TO_RUN: {model_name}")
        print(f"\n******** Running experiments for {model_name} ********\n")
        all_stats[model_name] = run_model_experiments(model_name, NUM_ITERATIONS)
        clear_folder()

    nn_stats = stats_or_empty(all_stats, "NN")
    kkt_stats = stats_or_empty(all_stats, "KKThPINN")

    pd.DataFrame({
        "Iteration": range(1, NUM_ITERATIONS + 1),
        "NN_Training_Error": nn_stats["training_errors"],
        "NN_Training_Time_sec": nn_stats["training_times"],
        "KKThPINN_Training_Error": kkt_stats["training_errors"],
        "KKThPINN_Training_Time_sec": kkt_stats["training_times"],
    }).to_csv(TRAINING_CSV_PATH, index=False)
    print(f"\nTraining errors saved at: {TRAINING_CSV_PATH}")

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
    print(f"Experiment errors saved at: {EXPERIMENT_CSV_PATH}")

    for model_name, stats in all_stats.items():
        update_results_master(model_name, SCENARIO_ID, stats)
    archive_final_csvs()

    print("\n=== Scenario means ===")
    for model_name, stats in all_stats.items():
        print(
            f"{SCENARIO_ID} | {model_name}: train_time={stats['train_time_mean']:.4f}s, "
            f"exp_time={stats['exp_time_mean']:.4f}s, RMSE={stats['rmse_mean']:.6e}, "
            f"VIOL={stats['viol_mean']:.6e}, VIOL_NL={stats['viol_nl_mean']:.6e}"
        )


if __name__ == "__main__":
    main()
