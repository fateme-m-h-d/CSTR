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

from scipy.optimize import fsolve  # for ground-truth solver

# === Configuration ===
base_dir = os.getcwd()   # must be run from within total/
target_folder = os.path.join(base_dir, "new1")
os.makedirs(target_folder, exist_ok=True)

# List of source files to be copied into the folder
files_to_copy = ["main.py", "train.py", "models.py", "utils.py", "curves.py", "data.csv", "scaler.pkl"]

# CSV file paths (working copies in project root)
training_csv_path   = os.path.join(base_dir, "training_epoch_errors.csv")
experiment_csv_path = os.path.join(base_dir, "experiment_epoch_errors.csv")

# === Archives and master cross-sample CSV ===
ARCHIVE_ROOT         = os.path.join(base_dir, "models_archive")
RESULTS_ARCHIVE_ROOT = os.path.join(base_dir, "results_archive")
os.makedirs(ARCHIVE_ROOT, exist_ok=True)
os.makedirs(RESULTS_ARCHIVE_ROOT, exist_ok=True)

# Mirror of the master for convenience
results_by_samples_csv_path = os.path.join(base_dir, "results_by_samples.csv")
# Durable master that accumulates columns across reruns
RESULTS_MASTER_CSV = os.path.join(RESULTS_ARCHIVE_ROOT, "results_by_samples_master.csv")

SOURCE_DATASET = os.path.join(base_dir, "data.csv")        # copied each run
TARGET_DATASET = os.path.join(target_folder, "data.csv")   # actually consumed by main.py


# ------------------------------- Helpers: file ops & counting -------------------------------
def _copy_if_exists(src: str, dst: str):
    if src and os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True
        )
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


# ===== Archive one run (labels folder with ACTUAL rows in new1/data.csv) =====
def archive_current_run(model_name: str, run_idx: int, train_err, exp_rmse, exp_viol, exp_viol_nl):
    n_samples_actual = _count_rows(TARGET_DATASET)
    arch_dir  = make_archive_dir(n_samples_actual, model_name, run_idx)

    # model state
    model_root = os.path.join(target_folder, "models")
    latest_pth = find_latest_model_file(model_root)
    _copy_if_exists(latest_pth, os.path.join(arch_dir, "model_state.pth"))

    # core inputs used
    _copy_if_exists(TARGET_DATASET,                         os.path.join(arch_dir, "data.csv"))
    _copy_if_exists(os.path.join(target_folder, "scaler.pkl"), os.path.join(arch_dir, "scaler.pkl"))

    # code snapshot
    code_snap = os.path.join(arch_dir, "code_snapshot")
    os.makedirs(code_snap, exist_ok=True)
    for f in files_to_copy:
        src = os.path.join(base_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(code_snap, os.path.basename(f)))

    # run artifacts (any CSVs produced under target_folder)
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
            f"experiment_rmse_total={exp_rmse}\n"
            f"experiment_violation={exp_viol}\n"
            f"experiment_violation_original_nonlinear={exp_viol_nl}\n"  # NEW
            f"created_utc={pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
        )
    print(f"[ARCHIVED] {model_name} run {run_idx} => {arch_dir}")
    return arch_dir


# ------------------------------- 1) Ground Truth Setup -------------------------------
Cao = 1 #mol/L
Cbo = 2 #mol/L
Cco = 0 #mol/L
V = 10 #L
Q = 1 #L/s
tau = V/Q #s

Afo = 10e12
Eaf = 90000 #J/mol
Aro = 10e10
Ear = 80000 #J/mol
R = 8.314 #J/mol

def equations(variables, T):
    Cc, Cb, Ca = variables
    kf = Afo * np.exp(-Eaf/(R*T)) # forward
    kr = Aro * np.exp(-Ear/(R*T)) # reverse
    eq1 = Cao - Ca + -kf*Ca*(Cb**2)*tau + kr*(Cao-Ca+Cbo-Cb)*tau
    eq2 = Cbo - Cb + -2*kf*Ca*(Cb**2)*tau + 2*kr*(Cao-Ca+Cbo-Cb)*tau
    eq3 = Cc - Cao + Ca - Cbo + Cb
    return [eq1, eq2, eq3]

def get_ground_truth(n=30):
    T_values   = np.linspace(280, 600, n)
    Ca_values  = np.ones(n) * Cao
    Cb_values  = np.ones(n) * Cbo
    Cc_values  = np.ones(n) * Cco
    initial_guess = [Cco, Cbo, Cao]
    for i, T in enumerate(T_values):
        solution, _, ier, mesg = fsolve(equations, initial_guess, args=(T,), full_output=True)
        if ier == 1:
            Cc_values[i], Cb_values[i], Ca_values[i] = solution
        else:
            print(f"Solver did not converge for T = {T}. Message: {mesg}")
    return T_values, Ca_values, Cb_values, Cc_values


# === Helper Functions ===
def copy_files():
    """Copy all necessary files into the target folder."""
    for file in files_to_copy:
        src_file = os.path.join(base_dir, file)
        if os.path.exists(src_file):
            shutil.copy2(src_file, os.path.join(target_folder, file))
            print(f"Copied {file} to {target_folder}")
        else:
            print(f"Warning: {file} not found.")

def run_main(model_name, job, run_idx=None):
    """
    Run main.py with model_name and job ('train' or 'experiment').
    Returns: 
      - for 'train' -> float loss_train
      - for 'experiment' -> dict {'rmse_total': float, 'violation': float}
    Also writes logs under new1/logs/.
    """
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

    # persist logs
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
        return extract_experiment_scores(result.stdout)
    else:
        return None

def extract_experiment_scores(output):
    """
    Parse the final printed dict and return:
    {'rmse_total': float|nan, 'violation': float|nan,
     'violation_original_nonlinear': float|nan}
    """
    rmse_val, viol_val, viol_nl_val = np.nan, np.nan, np.nan
    for line in reversed(output.splitlines()):
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                d = ast.literal_eval(s)
                if "rmse_total" in d: rmse_val = float(d["rmse_total"])
                if "violation"  in d: viol_val = float(d["violation"])
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

def extract_last_epoch_error(output):
    """
    Extract 'loss_train' from the last 'epoch:' line.
    """
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
    """Empty the target folder by deleting its contents."""
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
    """
    Perform num_iterations:
      - copy files
      - train and record loss
      - experiment and record rmse_total + violation
      - archive this run (model + artifacts)
      - clear folder between runs
    Also plots predictions across runs.
    Returns training_errors, experiment_rmse, experiment_viol, rmse_mean, viol_mean.
    """
    training_errors = []
    experiment_rmse  = []
    experiment_viol  = []
    experiment_viol_nl = []   # NEW
    
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_ground_truth(ax)
    
    for i in range(num_iterations):
        print(f"\n=== Iteration {i+1} for model {model_name} ===\n")
        copy_files()  # refresh TARGET_DATASET etc.

        print(f"[INFO] TARGET_DATASET rows for this run: {_count_rows(TARGET_DATASET)}")
        
        train_error = run_main(model_name, "train", run_idx=i+1)
        if train_error is None:
            train_error = float('nan')
        training_errors.append(train_error)
        
        sc = run_main(model_name, "experiment", run_idx=i+1)
        rmse_val = float(sc.get("rmse_total", np.nan)) if isinstance(sc, dict) else float('nan')
        viol_val = float(sc.get("violation",  np.nan)) if isinstance(sc, dict) else float('nan')
        viol_nl  = float(sc.get("violation_original_nonlinear", np.nan)) if isinstance(sc, dict) else float('nan')

        experiment_rmse.append(rmse_val)
        experiment_viol.append(viol_val)
        experiment_viol_nl.append(viol_nl)
        
        # predictions for plotting
        X_test, Y_pred = get_predictions(model_name)
        if X_test is not None and Y_pred is not None:
            plot_predictions_on_axes(ax, X_test, Y_pred, run_number=i+1, model_name=model_name)

        # archive this run (saves model, csvs, code, logs + violation in metadata)
        archive_current_run(model_name, i+1, train_error, rmse_val, viol_val, viol_nl)
        
        if i < num_iterations - 1:
            clear_folder()
        
    ax.set_title(f"Predictions Over {num_iterations} Runs ({model_name})")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Concentration (mol/L)")
    plot_path = os.path.join(base_dir, f"{model_name}_all_runs_plot.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved combined plot at {plot_path}")
    plt.close()
    
    rmse_mean = float(np.nanmean(experiment_rmse)) if len(experiment_rmse) else float('nan')
    viol_mean = float(np.nanmean(experiment_viol)) if len(experiment_viol) else float('nan')
    viol_nl_mean = float(np.nanmean(experiment_viol_nl)) if experiment_viol_nl else float('nan')
    return training_errors, experiment_rmse, experiment_viol, experiment_viol_nl, rmse_mean, viol_mean, viol_nl_mean


def plot_predictions_on_axes(ax, X, Y_pred, run_number, model_name):
    if len(X.shape) > 1:
        X = X.squeeze(-1)
    colors = ['blue', 'red', 'green']
    labels = ['Ca', 'Cb', 'Cc']
    for i in range(3):
        ax.plot(X, Y_pred[:, i],
                color=colors[i],
                linestyle='-',
                label=f"Run {run_number} {model_name} {labels[i]} (pred)")

def plot_ground_truth(ax):
    n = 30
    T_vals, Ca_vals, Cb_vals, Cc_vals = get_ground_truth(n=n)
    ax.plot(T_vals, Ca_vals, 'b--', label='Ground truth Ca')
    ax.plot(T_vals, Cb_vals, 'r--', label='Ground truth Cb')
    ax.plot(T_vals, Cc_vals, 'g--', label='Ground truth Cc')


def get_predictions(model_name):
    """Load saved model and predict along 280..600K for plotting."""
    scaler_path = os.path.join(target_folder, 'scaler.pkl')
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        print("Scaler file not found at:", scaler_path)
        return None, None

    input_dim = 1
    hidden_dim = 32
    hidden_num = 2
    z0_dim = 3

    if model_name == "NN":
        model_path = "./new1/models/cstr/NN/0.2/MODELID_0.2_0.pth"
        model_type = "NN"
        A_list, B_list, b_list = None, None, None
    elif model_name == "KKThPINN":
        model_path = "./new1/models/cstr/KKThPINN/0.2/MODELID_0.2_0.pth"
        model_type = "KKT"
        # === piece-wise linearization params (your values) ===
        A_list = [
            torch.tensor([[-0.00301071551554214]], dtype=torch.float64),
            torch.tensor([[-0.0287912896000818 ]], dtype=torch.float64),
            torch.tensor([[-0.0589977043951539 ]], dtype=torch.float64),
            torch.tensor([[-0.175928980736293 ]], dtype=torch.float64),
            torch.tensor([[-2.22034336021516  ]], dtype=torch.float64),
            torch.tensor([[-19.068831993137   ]], dtype=torch.float64),
            torch.tensor([[-67.0529314068883 ]], dtype=torch.float64),
            torch.tensor([[-147.195966751539 ]], dtype=torch.float64),
            torch.tensor([[-242.760891902504 ]], dtype=torch.float64),
            torch.tensor([[-356.302755070124 ]], dtype=torch.float64),
            torch.tensor([[-496.399591699859 ]], dtype=torch.float64),
            torch.tensor([[-650.139480396354 ]], dtype=torch.float64),
        ]
        B_list = [
            torch.tensor([[-1.02825709223637,   -0.0282570922363733, 0]], dtype=torch.float64),
            torch.tensor([[-1.54923960097239,   -0.549239600972388,  0]], dtype=torch.float64),
            torch.tensor([[-6.41562952963354,   -5.41562952963354,  0]], dtype=torch.float64),
            torch.tensor([[-47.4935472673711,   -46.4935472673712,  0]], dtype=torch.float64),
            torch.tensor([[-1002.53593524019,   -1001.53593524029,  0]], dtype=torch.float64),
            torch.tensor([[-11478.9269831455,   -11477.9269831524,  0]], dtype=torch.float64),
            torch.tensor([[-48213.8668993521,   -48212.866899362,   0]], dtype=torch.float64),
            torch.tensor([[-119069.962944269,   -119068.962945171,  0]], dtype=torch.float64),
            torch.tensor([[-212313.514625447,   -212312.514623723,  0]], dtype=torch.float64),
            torch.tensor([[-331422.775549942,   -331421.775550256,  0]], dtype=torch.float64),
            torch.tensor([[-487637.901829002,   -487636.901838071,  0]], dtype=torch.float64),
            torch.tensor([[-668298.267688647,   -668297.267673516,  0]], dtype=torch.float64),
        ]
        b_list = [
            torch.tensor([-1.93327845562254 ], dtype=torch.float64),
            torch.tensor([-11.1707510081181 ], dtype=torch.float64),
            torch.tensor([-29.674961918774  ], dtype=torch.float64),
            torch.tensor([-131.782655072763 ], dtype=torch.float64),
            torch.tensor([-2204.88922143194 ], dtype=torch.float64),
            torch.tensor([-22375.8507664129 ], dtype=torch.float64),
            torch.tensor([-87505.3569136546 ], dtype=torch.float64),
            torch.tensor([-206400.451290504 ], dtype=torch.float64),
            torch.tensor([-357218.284591596 ], dtype=torch.float64),
            torch.tensor([-544832.512463095 ], dtype=torch.float64),
            torch.tensor([-785540.247635705 ], dtype=torch.float64),
            torch.tensor([-1058769.64213322 ], dtype=torch.float64),
        ]
        A_list = [A.double() for A in A_list]
        B_list = [B.double() for B in B_list]
        b_list = [b.double() for b in b_list]
    else:
        print("Unknown model name:", model_name)
        return None, None

    from load_data import load_saved_model, make_prediction
    model = load_saved_model(model_path, model_type, input_dim, hidden_dim, hidden_num, z0_dim, A_list, B_list, b_list)
    model = model.double()

    new_temperatures = np.linspace(280, 600, 30)
    predictions = make_prediction(model, scaler, new_temperatures)
    return new_temperatures.reshape(-1, 1), predictions


# === Cross-sample master CSV updater (adds columns, never overwrites old sample sizes) ===
def update_results_by_samples_csv(model_name: str, num_samples: int,
                                  rmse_mean: float, viol_mean: float, viol_nl_mean: float):
    num_samples = int(num_samples)
    col_rmse = f"{num_samples}_RMSE_TOTAL"
    col_viol = f"{num_samples}_VIOL"
    col_viol_nl = f"{num_samples}_VIOL_NL"   # NEW

    if os.path.exists(RESULTS_MASTER_CSV):
        df = pd.read_csv(RESULTS_MASTER_CSV)
    else:
        df = pd.DataFrame({"Model": ["NN", "KKThPINN"]})

    for name in ("NN", "KKThPINN"):
        if not (df["Model"] == name).any():
            df = pd.concat([df, pd.DataFrame({"Model": [name]})], ignore_index=True)

    for c in (col_rmse, col_viol, col_viol_nl):   # include new column
        if c not in df.columns:
            df[c] = np.nan

    df.loc[df["Model"].eq(model_name), [col_rmse, col_viol, col_viol_nl]] = \
        [rmse_mean, viol_mean, viol_nl_mean]

    def _key(c):
        if c == "Model": return (-1, "")
        size = int(c.split("_")[0])
        metric = "_".join(c.split("_")[1:])
        return (size, metric)
    ordered = ["Model"] + sorted([c for c in df.columns if c != "Model"], key=_key)
    df = df[ordered]

    df.to_csv(RESULTS_MASTER_CSV, index=False)
    df.to_csv(results_by_samples_csv_path, index=False)
    print(f"[SUMMARY] Updated master: {RESULTS_MASTER_CSV} | mirror: {results_by_samples_csv_path}")



def archive_final_csvs():
    """Snapshot final CSVs under results_archive/<actual_rows>/<timestamp>/."""
    actual_rows = _count_rows(TARGET_DATASET)
    if actual_rows <= 0:
        actual_rows = _count_rows(SOURCE_DATASET)

    ts = pd.Timestamp.utcnow().strftime("%Y%m%d-%H%M%S")
    final_dir = os.path.join(RESULTS_ARCHIVE_ROOT, str(actual_rows), ts)
    os.makedirs(final_dir, exist_ok=True)
    for p in [training_csv_path, experiment_csv_path, results_by_samples_csv_path, RESULTS_MASTER_CSV]:
        _copy_if_exists(p, os.path.join(final_dir, os.path.basename(p)))
    print(f"[RESULTS] Final CSVs archived at: {final_dir}")


# === Main Execution ===
def main():
    num_iterations = 50 # Perform 50 cycles for each model

    print(f"[INFO] SOURCE rows: {_count_rows(SOURCE_DATASET)} | current TARGET rows: {_count_rows(TARGET_DATASET)}")

    # Run experiments for NN
    print("\n******** Running experiments for NN ********\n")
    nn_train, nn_rmse, nn_viol, nn_viol_nl, nn_rmse_mean, nn_viol_mean, nn_viol_nl_mean = run_model_experiments("NN", num_iterations)
    
    # Clear folder before next model
    clear_folder()
    
    # Run experiments for KKThPINN (KKT)
    print("\n******** Running experiments for KKThPINN ********\n")
    kkt_train, kkt_rmse, kkt_viol, kkt_viol_nl, kkt_rmse_mean, kkt_viol_mean, kkt_viol_nl_mean = run_model_experiments("KKThPINN", num_iterations)
    
    # Save per-model combined errors
    pd.DataFrame({
        "Iteration": range(1, num_iterations + 1),
        "NN_Training_Error": nn_train,
        "KKThPINN_Training_Error": kkt_train
    }).to_csv(training_csv_path, index=False)
    print(f"\nTraining errors saved at: {training_csv_path}")
    
    pd.DataFrame({
        "NN_Experiment_RMSE": nn_rmse,
        "NN_Experiment_VIOL": nn_viol,
        "NN_Experiment_VIOL_NL": nn_viol_nl,           # NEW
        "KKThPINN_Experiment_RMSE": kkt_rmse,
        "KKThPINN_Experiment_VIOL": kkt_viol,
        "KKThPINN_Experiment_VIOL_NL": kkt_viol_nl     # NEW
    }).to_csv(experiment_csv_path, index=False)
    print(f"Experiment errors saved at: {experiment_csv_path}")

    # Use ACTUAL rows in target to key the cross-sample update
    final_rows = _count_rows(TARGET_DATASET)
    if final_rows <= 0:
        final_rows = _count_rows(SOURCE_DATASET)

    update_results_by_samples_csv("NN",       final_rows, nn_rmse_mean,  nn_viol_mean,  nn_viol_nl_mean)
    update_results_by_samples_csv("KKThPINN", final_rows, kkt_rmse_mean, kkt_viol_mean, kkt_viol_nl_mean)


    # Snapshot all CSVs to a timestamped folder
    archive_final_csvs()

    print("\n=== Cross-sample means ===")
    print(f"samples={final_rows} | NN:  RMSE_total(mean of {num_iterations})={nn_rmse_mean:.6e},  VIOL(mean)={nn_viol_mean:.6e}, VIOL_NL(mean)={nn_viol_nl_mean:.6e}")
    print(f"samples={final_rows} | KKT: RMSE_total(mean of {num_iterations})={kkt_rmse_mean:.6e}, VIOL(mean)={kkt_viol_mean:.6e}, VIOL_NL(mean)={kkt_viol_nl_mean:.6e}")


if __name__ == "__main__":
    main()
