import os
import glob
import shutil
import subprocess
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
import torch
import pickle
import time

from scipy.optimize import fsolve
from scipy import stats as st

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # Force CPU

# === Configuration ===
base_dir = os.getcwd()   # must be run from within total/
target_folder = os.path.join(base_dir, "new1")
os.makedirs(target_folder, exist_ok=True)

files_to_copy = ["main.py", "train.py", "models.py", "utils.py", "curves.py",
                 "data.csv", "scaler_utils.py", "scaler.pkl", "load_data.py"]

# Working CSVs written here (root of project)
training_csv_path   = os.path.join(base_dir, "training_epoch_errors.csv")
experiment_csv_path = os.path.join(base_dir, "experiment_epoch_errors.csv")

# A convenient mirror of the master (see below)
results_by_samples_csv_path = os.path.join(base_dir, "results_by_samples.csv")

# === NEVER-DELETE ARCHIVE ROOTS ===
ARCHIVE_ROOT          = os.path.join(base_dir, "models_archive")
RESULTS_ARCHIVE_ROOT  = os.path.join(base_dir, "results_archive")
os.makedirs(ARCHIVE_ROOT, exist_ok=True)
os.makedirs(RESULTS_ARCHIVE_ROOT, exist_ok=True)

# === Master cross-sample CSV that persists and accumulates columns ===
RESULTS_MASTER_CSV = os.path.join(RESULTS_ARCHIVE_ROOT, "results_by_samples_master.csv")

# Single source-of-truth paths for dataset we intend to use/copy
SOURCE_DATASET = os.path.join(base_dir, "data.csv")      # copied into target_folder each run
TARGET_DATASET = os.path.join(target_folder, "data.csv")  # actually consumed by main.py


# ------------------------------- Utilities -------------------------------
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
    ts = time.strftime("%Y%m%d-%H%M%S")
    leaf = f"{ts}_run{run_idx:02d}_{model_name}"
    path = os.path.join(ARCHIVE_ROOT, str(num_samples), leaf)
    os.makedirs(path, exist_ok=True)
    return path


# ===== Archive one run (now labels folder with the ACTUAL rows in new1/data.csv) =====
def archive_current_run(model_name: str, run_idx: int, train_err, exp_err):
    # IMPORTANT: count rows from the dataset main.py actually used
    n_samples_actual = _count_rows(TARGET_DATASET)
    arch_dir  = make_archive_dir(n_samples_actual, model_name, run_idx)

    # model state
    model_root = os.path.join(target_folder, "models")
    latest_pth = find_latest_model_file(model_root)
    _copy_if_exists(latest_pth, os.path.join(arch_dir, "model_state.pth"))

    # core inputs
    _copy_if_exists(TARGET_DATASET,                       os.path.join(arch_dir, "data.csv"))
    _copy_if_exists(os.path.join(target_folder, "scaler.pkl"), os.path.join(arch_dir, "scaler.pkl"))

    # code snapshot
    code_snap = os.path.join(arch_dir, "code_snapshot")
    os.makedirs(code_snap, exist_ok=True)
    for f in files_to_copy:
        src = os.path.join(base_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(code_snap, os.path.basename(f)))

    # archive any CSVs produced during this run
    artifacts_dir = os.path.join(arch_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    for csv_path in glob.glob(os.path.join(target_folder, "*.csv")):
        dst_name = f"run{run_idx:02d}_" + os.path.basename(csv_path)
        shutil.copy2(csv_path, os.path.join(artifacts_dir, dst_name))

    # logs (if any)
    logs_dir = os.path.join(target_folder, "logs")
    if os.path.isdir(logs_dir):
        shutil.copytree(logs_dir, os.path.join(arch_dir, "logs"), dirs_exist_ok=True)

    # metadata
    meta_txt = os.path.join(arch_dir, "RUN_INFO.txt")
    with open(meta_txt, "w", encoding="utf-8") as fp:
        fp.write(
            f"model={model_name}\n"
            f"run_index={run_idx}\n"
            f"num_samples_actual={n_samples_actual}\n"
            f"train_error={train_err}\n"
            f"experiment_rmse={exp_err}\n"
            f"created_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
        )

    print(f"[ARCHIVED] {model_name} run {run_idx} => {arch_dir}")
    return arch_dir


# ------------------------------- Ground truth helpers -------------------------------
Cao = 1; Cbo = 2; Cco = 0
V = 10; Q = 1; tau = V/Q
Afo = 10e12; Eaf = 90000
Aro = 10e10; Ear = 80000
R = 8.314

def equations(variables, T):
    Cc, Cb, Ca = variables
    kf = Afo * np.exp(-Eaf/(R*T))
    kr = Aro * np.exp(-Ear/(R*T))
    eq1 = Cao - Ca + -kf*Ca*(Cb**2)*tau + kr*(Cao-Ca+Cbo-Cb)*tau
    eq2 = Cbo - Cb + -2*kf*Ca*(Cb**2)*tau + 2*kr*(Cao-Ca+Cbo-Cb)*tau
    eq3 = Cc - Cao + Ca - Cbo + Cb
    return [eq1, eq2, eq3]

def get_ground_truth(n=800):
    T_values   = np.linspace(280, 600, n)
    Ca_values  = np.ones(n) * Cao
    Cb_values  = np.ones(n) * Cbo
    Cc_values  = np.ones(n) * Cco
    initial_guess = [Cco, Cbo, Cao]
    for i, T in enumerate(T_values):
        solution, _, ier, mesg = fsolve(equations, initial_guess, args=(T,), full_output=True, xtol=1.0e-11)
        if ier == 1:
            Cc_values[i], Cb_values[i], Ca_values[i] = solution
        else:
            print(f"Solver did not converge for T = {T}. Message: {mesg}")
    kf_arr = Afo * np.exp(-Eaf/(R*T_values))
    kr_arr = Aro * np.exp(-Ear/(R*T_values))
    f = kr_arr*(Cao-Ca_values+Cbo-Cb_values+Cco)
    g = kf_arr*Ca_values*(Cb_values**2)
    return T_values, Ca_values, Cb_values, Cc_values, f, g


# ------------------------------- Parsers -------------------------------
def extract_experiment_scores(output):
    rmse_val, viol_val = None, None
    lines = output.splitlines()
    for line in reversed(lines):
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                d = ast.literal_eval(s)
                rmse_val = d.get("rmse3", d.get("rmse_inner", d.get("rmse_total")))
                viol_val = d.get("violation", None)
            except Exception:
                pass
            break
    return {"rmse": None if rmse_val is None else float(rmse_val),
            "violation": None if viol_val is None else float(viol_val)}

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


# ------------------------------- Run helpers -------------------------------
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
    result = subprocess.run(args, capture_output=True, text=True, cwd=target_folder)

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
    
    if job == 'train':
        return extract_last_epoch_error(result.stdout)
    elif job == 'experiment':
        sc = extract_experiment_scores(result.stdout)
        print(f"Experiment RMSE3 (stdout): {sc['rmse']}, violation: {sc['violation']}")
        sc["stdout"] = result.stdout
        return sc
    else:
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


def _band_rmse3(T, Y_pred, Ca_true, Cb_true, Cc_true):
    T = T.reshape(-1)
    Yt = np.stack([Ca_true, Cb_true, Cc_true], axis=1)
    err = Y_pred[:, :3] - Yt
    def rmse_on(mask):
        if mask.sum() == 0:
            return float("nan")
        return float(np.sqrt(np.mean(err[mask]**2)))
    low_mask  = (T <= 300.0)
    high_mask = (T >= 520.0)
    return rmse_on(low_mask), rmse_on(high_mask)

def _welch_and_d(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    t, p = st.ttest_ind(a, b, equal_var=False)
    sp = np.sqrt(0.5*(a.var(ddof=1) + b.var(ddof=1)))
    d = (b.mean() - a.mean()) / sp if sp > 0 else 0.0
    return t, p, d, a.mean(), b.mean(), a.var(ddof=1), b.var(ddof=1)


def get_predictions(model_name):
    scaler_path = os.path.join(target_folder, 'scaler.pkl')
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        print("Scaler file not found at:", scaler_path)
        return None, None

    input_dim = 1; hidden_dim = 32; hidden_num = 2; z0_dim = 5; z0_inner_dim = 3 

    if model_name == "NN":
        model_path = "./new1/models/cstr/NN/0.2/MODELID_0.2_0.pth"
        model_type = "NN"
        A, B, b = None, None, None
        from load_data import load_saved_model, make_prediction
        model = load_saved_model(model_path, model_type,
                                 input_dim, hidden_dim, hidden_num, z0_inner_dim,
                                 z0_dim, A, B, b)
    elif model_name == "KKThPINN":
        model_path = "./new1/models/cstr/KKThPINN/0.2/MODELID_0.2_0.pth"
        model_type = "KKT"
        A = torch.tensor([[0]], dtype=torch.float64)
        B = torch.tensor([[1, 0, 0, -10, 10]], dtype=torch.float64)
        b = torch.tensor([1], dtype=torch.float64)
        A = A.double(); B = B.double(); b = b.double()
        from load_data import load_saved_model, make_prediction
        model = load_saved_model(model_path, model_type,
                                 input_dim, hidden_dim, hidden_num, z0_inner_dim,
                                 z0_dim, A, B, b)
    else:
        print("Unknown model name:", model_name)
        return None, None

    model = model.double()
    new_temperatures = np.linspace(280, 600, 800)
    predictions = make_prediction(model, scaler, new_temperatures)
    return new_temperatures.reshape(-1, 1), predictions


def run_model_experiments(model_name, num_iterations):
    """
    Runs num_iterations. Uses experiment stdout only (RMSE3 + violation).
    Returns:
      training_errors, experiment_rmse, experiment_viol, band_low_rmse3, band_high_rmse3, mean_rmse3, mean_violation
    """
    training_errors    = []
    experiment_rmse    = []
    experiment_viol    = []
    band_low_rmse3     = []
    band_high_rmse3    = []
    
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_ground_truth(ax)
    
    for i in range(num_iterations):
        print(f"\n=== Iteration {i+1} for model {model_name} ===\n")
        copy_files()  # ensures TARGET_DATASET is a fresh copy of SOURCE_DATASET

        # Read the ACTUAL dataset rows from TARGET_DATASET just before training this run
        rows_target = _count_rows(TARGET_DATASET)
        print(f"[INFO] TARGET_DATASET rows for this run: {rows_target}")

        train_error = run_main(model_name, "train", run_idx=i+1)
        if train_error is None:
            train_error = float('nan')
        training_errors.append(train_error)
        
        sc = run_main(model_name, "experiment", run_idx=i+1)
        rmse_val = sc.get("rmse", float('nan')) if isinstance(sc, dict) else float('nan')
        viol_val = sc.get("violation", float('nan')) if isinstance(sc, dict) else float('nan')
        experiment_rmse.append(rmse_val)
        experiment_viol.append(viol_val)
        
        # Optional: predictions for plotting & band RMSE (kept)
        X_test, Y_pred = get_predictions(model_name)
        if X_test is not None and Y_pred is not None:
            n = X_test.shape[0]
            T_vals, Ca_vals, Cb_vals, Cc_vals, _, _ = get_ground_truth(n=n)
            low_rmse, high_rmse = _band_rmse3(X_test.squeeze(), Y_pred, Ca_vals, Cb_vals, Cc_vals)
            band_low_rmse3.append(low_rmse); band_high_rmse3.append(high_rmse)
            plot_predictions_on_axes(ax, X_test, Y_pred, run_number=i+1, model_name=model_name)
        else:
            band_low_rmse3.append(float('nan')); band_high_rmse3.append(float('nan'))

        # Archive this run under the ACTUAL row count seen
        archive_current_run(model_name, i+1, train_error, rmse_val)
        
        if i < num_iterations - 1:
            clear_folder()
        
    ax.set_title(f"Predictions Over {num_iterations} Runs ({model_name})")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Concentration (mol/L)")
    plot_path = os.path.join(base_dir, f"{model_name}_all_runs_plot.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved combined plot at {plot_path}")
    plt.close()
    
    mean_rmse3     = float(np.nanmean(experiment_rmse)) if len(experiment_rmse) else float('nan')
    mean_violation = float(np.nanmean(experiment_viol)) if len(experiment_viol) else float('nan')
    
    return training_errors, experiment_rmse, experiment_viol, band_low_rmse3, band_high_rmse3, mean_rmse3, mean_violation


def plot_predictions_on_axes(ax, X, Y_pred, run_number, model_name):
    if len(X.shape) > 1:
        X = X.squeeze(-1)
    colors = ['blue', 'red', 'green']
    labels = ['Ca', 'Cb', 'Cc']
    for i in range(3):
        ax.plot(X, Y_pred[:, i], color=colors[i], linestyle='-', label=f"Run {run_number} {model_name} {labels[i]} (pred)")

def plot_ground_truth(ax):
    n = 800
    T_vals, Ca_vals, Cb_vals, Cc_vals, f, g = get_ground_truth(n=n)
    ax.plot(T_vals, Ca_vals, 'b--', label='Ground truth Ca')
    ax.plot(T_vals, Cb_vals, 'r--', label='Ground truth Cb')
    ax.plot(T_vals, Cc_vals, 'g--', label='Ground truth Cc')


# === cross-sample summary CSV (RMSE3 + VIOL means) that ACCUMULATES columns ===
def update_results_by_samples_csv(model_name: str, num_samples: int, rmse3_mean: float, violation_mean: float):
    num_samples = int(num_samples)
    col_rmse = f"{num_samples}_RMSE3"
    col_viol = f"{num_samples}_VIOL"

    # Load or create the master frame
    if os.path.exists(RESULTS_MASTER_CSV):
        df = pd.read_csv(RESULTS_MASTER_CSV)
    else:
        df = pd.DataFrame({"Model": ["NN", "KKThPINN"]})

    # Ensure both model rows exist (idempotent)
    for name in ("NN", "KKThPINN"):
        if not (df["Model"] == name).any():
            df = pd.concat([df, pd.DataFrame({"Model": [name]})], ignore_index=True)

    # Add columns for this sample size if missing
    for c in (col_rmse, col_viol):
        if c not in df.columns:
            df[c] = np.nan

    # Update the correct row
    mask = df["Model"].eq(model_name)
    df.loc[mask, [col_rmse, col_viol]] = [rmse3_mean, violation_mean]

    # Optional: keep columns ordered numerically by sample size (Model first)
    def _key(c):
        if c == "Model": return (-1, "")
        size, metric = c.split("_")
        return (int(size), metric)
    ordered_cols = ["Model"] + sorted([c for c in df.columns if c != "Model"], key=_key)
    df = df[ordered_cols]

    # Write back to the durable master AND mirror to working copy
    df.to_csv(RESULTS_MASTER_CSV, index=False)
    df.to_csv(results_by_samples_csv_path, index=False)
    print(f"[SUMMARY] Updated master: {RESULTS_MASTER_CSV} | mirror: {results_by_samples_csv_path}")


# === Archive the final combined CSVs into a timestamped folder (uses ACTUAL rows at the end) ===
def archive_final_csvs():
    """
    Copy the combined CSVs (and cross-sample CSVs) into a permanent,
    timestamped folder under results_archive/<num_samples_detected>/.
    """
    # Detect final TARGET rows (actual dataset used)
    num_samples_detected = _count_rows(TARGET_DATASET)
    ts = time.strftime("%Y%m%d-%H%M%S")
    final_dir = os.path.join(RESULTS_ARCHIVE_ROOT, str(num_samples_detected), ts)
    os.makedirs(final_dir, exist_ok=True)

    for p in [training_csv_path, experiment_csv_path, results_by_samples_csv_path, RESULTS_MASTER_CSV]:
        _copy_if_exists(p, os.path.join(final_dir, os.path.basename(p)))

    print(f"[RESULTS] Final CSVs archived at: {final_dir}")


# === Main Execution ===
def main():
    num_iterations = 50

    # Informational: what we plan to run with (source rows) and what's in target now
    src_rows   = _count_rows(SOURCE_DATASET)
    tgt_rows_0 = _count_rows(TARGET_DATASET)
    print(f"[INFO] SOURCE_DATASET rows: {src_rows} | current TARGET_DATASET rows (pre-run): {tgt_rows_0}")

    print("\n******** Running experiments for NN ********\n")
    nn_train, nn_rmse_list, nn_viol_list, nn_low, nn_high, nn_rmse_mean, nn_viol_mean = run_model_experiments("NN", num_iterations)
    clear_folder()
    
    print("\n******** Running experiments for KKThPINN ********\n")
    kkt_train, kkt_rmse_list, kkt_viol_list, kkt_low, kkt_high, kkt_rmse_mean, kkt_viol_mean = run_model_experiments("KKThPINN", num_iterations)
    
    # Write combined CSVs (working copies in project root)
    pd.DataFrame({
        "Iteration": range(1, num_iterations + 1),
        "NN_Training_Error": nn_train,
        "KKThPINN_Training_Error": kkt_train
    }).to_csv(training_csv_path, index=False)
    print(f"\nTraining errors saved at: {training_csv_path}")
    
    pd.DataFrame({
        "NN_Experiment_RMSE": nn_rmse_list,
        "NN_Experiment_VIOL": nn_viol_list,
        "KKThPINN_Experiment_RMSE": kkt_rmse_list,
        "KKThPINN_Experiment_VIOL": kkt_viol_list
    }).to_csv(experiment_csv_path, index=False)
    print(f"Experiment errors saved at: {experiment_csv_path}")

    # Detect ACTUAL rows after runs (should match what was used)
    final_rows = _count_rows(TARGET_DATASET)
    if final_rows <= 0:
        # Fallback to source if target missing
        final_rows = _count_rows(SOURCE_DATASET)

    # Update cross-sample summary (accumulates columns like 30_*, 100_*)
    update_results_by_samples_csv("NN",       final_rows, nn_rmse_mean,  nn_viol_mean)
    update_results_by_samples_csv("KKThPINN", final_rows, kkt_rmse_mean, kkt_viol_mean)

    # Persist the combined CSVs in a permanent, timestamped folder labeled by ACTUAL rows
    archive_final_csvs()

    print("\n=== Cross-sample metrics for this run ===")
    print(f"samples={final_rows} | NN:       RMSE3(mean of 50)={nn_rmse_mean:.6e},  violation(mean)={nn_viol_mean:.6e}")
    print(f"samples={final_rows} | KKThPINN: RMSE3(mean of 50)={kkt_rmse_mean:.6e}, violation(mean)={kkt_viol_mean:.6e}")


if __name__ == "__main__":
    main()
