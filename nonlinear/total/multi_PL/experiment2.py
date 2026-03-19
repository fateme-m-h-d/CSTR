import os
import glob
import shutil
import subprocess
import numpy as np
import pandas as pd
import ast
import time

# === Configuration ===
base_dir = os.getcwd()
target_folder = os.path.join(base_dir, "new1")
os.makedirs(target_folder, exist_ok=True)

# only copy the files needed by the 2D code
files_to_copy = ["main.py", "train.py", "models.py", "utils.py", "linearization.py", "data.csv", "ABb_matrices.csv"]

training_csv_path   = os.path.join(base_dir, "training_epoch_errors.csv")
experiment_csv_path = os.path.join(base_dir, "experiment_epoch_errors.csv")

ARCHIVE_ROOT         = os.path.join(base_dir, "models_archive")
RESULTS_ARCHIVE_ROOT = os.path.join(base_dir, "results_archive")
os.makedirs(ARCHIVE_ROOT, exist_ok=True)
os.makedirs(RESULTS_ARCHIVE_ROOT, exist_ok=True)

results_by_samples_csv_path = os.path.join(base_dir, "results_by_samples.csv")
RESULTS_MASTER_CSV = os.path.join(RESULTS_ARCHIVE_ROOT, "results_by_samples_master.csv")

SOURCE_DATASET = os.path.join(base_dir, "data.csv")
TARGET_DATASET = os.path.join(target_folder, "data.csv")


def _copy_if_exists(src: str, dst: str):
    if src and os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def _count_rows(csv_path: str) -> int:
    try:
        df = pd.read_csv(csv_path)
        return int(df.shape[0])
    except Exception:
        return -1


def find_latest_model_file(root_dir: str):
    newest_time, newest_path = -1, None
    for r, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith(".pth"):
                p = os.path.join(r, f)
                t = os.path.getmtime(p)
                if t > newest_time:
                    newest_time, newest_path = t, p
    return newest_path


def make_archive_dir(num_samples: int, model_name: str, run_idx: int) -> str:
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d-%H%M%S")
    leaf = f"{ts}_run{run_idx:02d}_{model_name}"
    path = os.path.join(ARCHIVE_ROOT, str(num_samples), leaf)
    os.makedirs(path, exist_ok=True)
    return path


def archive_current_run(model_name: str, run_idx: int, train_err, exp_rmse, exp_viol, exp_viol_nl):
    n_samples_actual = _count_rows(TARGET_DATASET)
    arch_dir = make_archive_dir(n_samples_actual, model_name, run_idx)

    model_root = os.path.join(target_folder, "models")
    latest_pth = find_latest_model_file(model_root)
    _copy_if_exists(latest_pth, os.path.join(arch_dir, "model_state.pth"))

    _copy_if_exists(TARGET_DATASET, os.path.join(arch_dir, "data.csv"))
    _copy_if_exists(os.path.join(target_folder, "scaler.pkl"), os.path.join(arch_dir, "scaler.pkl"))

    code_snap = os.path.join(arch_dir, "code_snapshot")
    os.makedirs(code_snap, exist_ok=True)
    for f in files_to_copy:
        src = os.path.join(base_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(code_snap, os.path.basename(f)))

    artifacts_dir = os.path.join(arch_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    for csv_path in glob.glob(os.path.join(target_folder, "*.csv")):
        dst_name = f"run{run_idx:02d}_" + os.path.basename(csv_path)
        shutil.copy2(csv_path, os.path.join(artifacts_dir, dst_name))

    logs_dir = os.path.join(target_folder, "logs")
    if os.path.isdir(logs_dir):
        shutil.copytree(logs_dir, os.path.join(arch_dir, "logs"), dirs_exist_ok=True)

    meta_txt = os.path.join(arch_dir, "RUN_INFO.txt")
    with open(meta_txt, "w", encoding="utf-8") as fp:
        fp.write(
            f"model={model_name}\n"
            f"run_index={run_idx}\n"
            f"num_samples_actual={n_samples_actual}\n"
            f"train_error={train_err}\n"
            f"experiment_rmse_total={exp_rmse}\n"
            f"experiment_violation={exp_viol}\n"
            f"experiment_violation_original_nonlinear={exp_viol_nl}\n"
            f"created_utc={pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
        )
    print(f"[ARCHIVED] {model_name} run {run_idx} => {arch_dir}")
    return arch_dir


def copy_files():
    for file in files_to_copy:
        src_file = os.path.join(base_dir, file)
        if os.path.exists(src_file):
            shutil.copy2(src_file, os.path.join(target_folder, file))
            print(f"Copied {file} to {target_folder}")
        else:
            print(f"Warning: {file} not found.")


def run_main(model_name, job, run_idx=None):
    main_file = os.path.join(target_folder, "main.py")
    if not os.path.exists(main_file):
        print("main.py not found in the target folder.")
        return None

    print(f"Running main.py with model={model_name} and job={job} ...")
    args = [
        "python", "main.py",
        "--model", model_name,
        "--model_id", "MODELID",
        "--dataset_type", "cstr",
        "--dataset_path", "./data.csv",
        "--job", job
    ]

    t0 = time.perf_counter()
    result = subprocess.run(args, capture_output=True, text=True, cwd=target_folder)
    elapsed = time.perf_counter() - t0

    logs_root = os.path.join(target_folder, "logs")
    os.makedirs(logs_root, exist_ok=True)
    tag = f"run{run_idx:02d}" if run_idx is not None else "run"
    with open(os.path.join(logs_root, f"{tag}_{job}_stdout.txt"), "w", encoding="utf-8") as f:
        f.write(result.stdout)
    if result.stderr:
        with open(os.path.join(logs_root, f"{tag}_{job}_stderr.txt"), "w", encoding="utf-8") as f:
            f.write(result.stderr)

    print("Standard Output:")
    print(result.stdout)
    if result.stderr:
        print("Standard Error:")
        print(result.stderr)

    if job == "train":
        return {
            "loss_train": extract_last_epoch_error(result.stdout),
            "train_time_sec": elapsed
        }

    elif job == "experiment":
        scores = extract_experiment_scores(result.stdout)
        eval_time = extract_named_time(result.stdout, "Evaluation time")
        scores["experiment_time_sec"] = eval_time if not np.isnan(eval_time) else elapsed
        return scores

    else:
        return None


def extract_experiment_scores(output):
    rmse_val, viol_val, viol_nl_val = np.nan, np.nan, np.nan
    for line in reversed(output.splitlines()):
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                d = ast.literal_eval(s)
                if "rmse_total" in d:
                    rmse_val = float(d["rmse_total"])
                if "violation" in d:
                    viol_val = float(d["violation"])
                if "violation_original_nonlinear" in d:
                    viol_nl_val = float(d["violation_original_nonlinear"])
            except Exception:
                pass
            break
    return {
        "rmse_total": rmse_val,
        "violation": viol_val,
        "violation_original_nonlinear": viol_nl_val
    }

def extract_named_time(output, label):
    for line in output.splitlines():
        s = line.strip()
        if s.startswith(label):
            try:
                return float(s.split(":")[1].replace("s", "").strip())
            except Exception:
                return np.nan
    return np.nan

def extract_last_epoch_error(output):
    lines = output.splitlines()
    for line in reversed(lines):
        if line.startswith("epoch:"):
            parts = line.split()
            data = {}
            i = 0
            while i < len(parts):
                if parts[i].endswith(":"):
                    key = parts[i].replace(":", "")
                    if i + 1 < len(parts):
                        data[key] = parts[i + 1]
                    i += 2
                else:
                    i += 1
            try:
                return float(data["loss_train"])
            except (KeyError, ValueError):
                return None
    return None


def clear_folder():
    for filename in os.listdir(target_folder):
        file_path = os.path.join(target_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                print(f"Deleted file: {file_path}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print(f"Deleted directory: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def run_model_experiments(model_name, num_iterations):
    training_errors = []
    training_times = []

    experiment_rmse = []
    experiment_viol = []
    experiment_viol_nl = []
    experiment_times = []

    for i in range(num_iterations):
        print(f"\n=== Iteration {i+1} for model {model_name} ===\n")
        copy_files()

        print(f"[INFO] TARGET_DATASET rows for this run: {_count_rows(TARGET_DATASET)}")

        train_res = run_main(model_name, "train", run_idx=i+1)
        train_error = float(train_res.get("loss_train", np.nan)) if isinstance(train_res, dict) else float("nan")
        train_time = float(train_res.get("train_time_sec", np.nan)) if isinstance(train_res, dict) else float("nan")

        training_errors.append(train_error)
        training_times.append(train_time)

        sc = run_main(model_name, "experiment", run_idx=i+1)
        rmse_val = float(sc.get("rmse_total", np.nan)) if isinstance(sc, dict) else float("nan")
        viol_val = float(sc.get("violation", np.nan)) if isinstance(sc, dict) else float("nan")
        viol_nl  = float(sc.get("violation_original_nonlinear", np.nan)) if isinstance(sc, dict) else float("nan")
        exp_time = float(sc.get("experiment_time_sec", np.nan)) if isinstance(sc, dict) else float("nan")

        experiment_rmse.append(rmse_val)
        experiment_viol.append(viol_val)
        experiment_viol_nl.append(viol_nl)
        experiment_times.append(exp_time)

        archive_current_run(model_name, i+1, train_error, rmse_val, viol_val, viol_nl)

        if i < num_iterations - 1:
            clear_folder()

    rmse_mean = float(np.nanmean(experiment_rmse)) if len(experiment_rmse) else float("nan")
    viol_mean = float(np.nanmean(experiment_viol)) if len(experiment_viol) else float("nan")
    viol_nl_mean = float(np.nanmean(experiment_viol_nl)) if len(experiment_viol_nl) else float("nan")
    train_time_mean = float(np.nanmean(training_times)) if len(training_times) else float("nan")
    exp_time_mean = float(np.nanmean(experiment_times)) if len(experiment_times) else float("nan")

    return (
        training_errors, training_times,
        experiment_rmse, experiment_viol, experiment_viol_nl, experiment_times,
        rmse_mean, viol_mean, viol_nl_mean,
        train_time_mean, exp_time_mean
    )
    
    
def update_results_by_samples_csv(model_name: str, num_samples: int,
                                  rmse_mean: float, viol_mean: float, viol_nl_mean: float,
                                  train_time_mean: float, exp_time_mean: float):
    num_samples = int(num_samples)
    col_rmse = f"{num_samples}_RMSE_TOTAL"
    col_viol = f"{num_samples}_VIOL"
    col_viol_nl = f"{num_samples}_VIOL_NL"
    col_train_time = f"{num_samples}_TRAIN_TIME_SEC"
    col_exp_time = f"{num_samples}_EXP_TIME_SEC"

    if os.path.exists(RESULTS_MASTER_CSV):
        df = pd.read_csv(RESULTS_MASTER_CSV)
    else:
        df = pd.DataFrame({"Model": ["NN", "KKThPINN"]})

    for name in ("NN", "KKThPINN"):
        if not (df["Model"] == name).any():
            df = pd.concat([df, pd.DataFrame({"Model": [name]})], ignore_index=True)

    for c in (col_rmse, col_viol, col_viol_nl, col_train_time, col_exp_time):
        if c not in df.columns:
            df[c] = np.nan

    df.loc[df["Model"].eq(model_name), [col_rmse, col_viol, col_viol_nl, col_train_time, col_exp_time]] = [
        rmse_mean, viol_mean, viol_nl_mean, train_time_mean, exp_time_mean
    ]

    def _key(c):
        if c == "Model":
            return (-1, "")
        size = int(c.split("_")[0])
        metric = "_".join(c.split("_")[1:])
        return (size, metric)

    ordered = ["Model"] + sorted([c for c in df.columns if c != "Model"], key=_key)
    df = df[ordered]

    df.to_csv(RESULTS_MASTER_CSV, index=False)
    df.to_csv(results_by_samples_csv_path, index=False)
    print(f"[SUMMARY] Updated master: {RESULTS_MASTER_CSV} | mirror: {results_by_samples_csv_path}")
    
    
def archive_final_csvs():
    actual_rows = _count_rows(TARGET_DATASET)
    if actual_rows <= 0:
        actual_rows = _count_rows(SOURCE_DATASET)

    ts = pd.Timestamp.utcnow().strftime("%Y%m%d-%H%M%S")
    final_dir = os.path.join(RESULTS_ARCHIVE_ROOT, str(actual_rows), ts)
    os.makedirs(final_dir, exist_ok=True)
    for p in [training_csv_path, experiment_csv_path, results_by_samples_csv_path, RESULTS_MASTER_CSV]:
        _copy_if_exists(p, os.path.join(final_dir, os.path.basename(p)))
    print(f"[RESULTS] Final CSVs archived at: {final_dir}")


def main():
    num_iterations = 30

    print(f"[INFO] SOURCE rows: {_count_rows(SOURCE_DATASET)} | current TARGET rows: {_count_rows(TARGET_DATASET)}")

    print("\n******** Running experiments for NN ********\n")
    (
        nn_train, nn_train_times,
        nn_rmse, nn_viol, nn_viol_nl, nn_exp_times,
        nn_rmse_mean, nn_viol_mean, nn_viol_nl_mean,
        nn_train_time_mean, nn_exp_time_mean
    ) = run_model_experiments("NN", num_iterations)

    clear_folder()

    print("\n******** Running experiments for KKThPINN ********\n")
    (
        kkt_train, kkt_train_times,
        kkt_rmse, kkt_viol, kkt_viol_nl, kkt_exp_times,
        kkt_rmse_mean, kkt_viol_mean, kkt_viol_nl_mean,
        kkt_train_time_mean, kkt_exp_time_mean
    ) = run_model_experiments("KKThPINN", num_iterations)

    pd.DataFrame({
        "Iteration": range(1, num_iterations + 1),
        "NN_Training_Error": nn_train,
        "NN_Training_Time_sec": nn_train_times,
        "KKThPINN_Training_Error": kkt_train,
        "KKThPINN_Training_Time_sec": kkt_train_times
    }).to_csv(training_csv_path, index=False)
    print(f"\nTraining errors saved at: {training_csv_path}")

    pd.DataFrame({
        "NN_Experiment_RMSE": nn_rmse,
        "NN_Experiment_VIOL": nn_viol,
        "NN_Experiment_VIOL_NL": nn_viol_nl,
        "NN_Experiment_Time_sec": nn_exp_times,
        "KKThPINN_Experiment_RMSE": kkt_rmse,
        "KKThPINN_Experiment_VIOL": kkt_viol,
        "KKThPINN_Experiment_VIOL_NL": kkt_viol_nl,
        "KKThPINN_Experiment_Time_sec": kkt_exp_times
    }).to_csv(experiment_csv_path, index=False)
    print(f"Experiment errors saved at: {experiment_csv_path}")

    final_rows = _count_rows(TARGET_DATASET)
    if final_rows <= 0:
        final_rows = _count_rows(SOURCE_DATASET)

    update_results_by_samples_csv(
        "NN", final_rows,
        nn_rmse_mean, nn_viol_mean, nn_viol_nl_mean,
        nn_train_time_mean, nn_exp_time_mean
    )
    update_results_by_samples_csv(
        "KKThPINN", final_rows,
        kkt_rmse_mean, kkt_viol_mean, kkt_viol_nl_mean,
        kkt_train_time_mean, kkt_exp_time_mean
    )

    archive_final_csvs()

    print("\n=== Cross-sample means ===")
    print(
        f"samples={final_rows} | NN: "
        f"TRAIN_TIME(mean)={nn_train_time_mean:.4f} s, "
        f"EXP_TIME(mean)={nn_exp_time_mean:.4f} s, "
        f"RMSE_total(mean)={nn_rmse_mean:.6e}, "
        f"VIOL(mean)={nn_viol_mean:.6e}, "
        f"VIOL_NL(mean)={nn_viol_nl_mean:.6e}"
    )
    print(
        f"samples={final_rows} | KKT: "
        f"TRAIN_TIME(mean)={kkt_train_time_mean:.4f} s, "
        f"EXP_TIME(mean)={kkt_exp_time_mean:.4f} s, "
        f"RMSE_total(mean)={kkt_rmse_mean:.6e}, "
        f"VIOL(mean)={kkt_viol_mean:.6e}, "
        f"VIOL_NL(mean)={kkt_viol_nl_mean:.6e}"
    )

if __name__ == "__main__":
    main()