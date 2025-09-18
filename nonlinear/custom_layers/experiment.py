import os
import shutil
import subprocess
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
import torch
import pickle

from scipy.optimize import fsolve  # for ground-truth solver

from scipy import stats as st

# === Configuration ===
# base_dir = "./CSTR/nonlinear/total"
base_dir = os.getcwd()   # must be run from within total/
target_folder = os.path.join(base_dir, "new1")
os.makedirs(target_folder, exist_ok=True)

# List of source files to be copied into the folder
files_to_copy = ["main.py", "train.py", "models.py", "utils.py", "curves.py", "data.csv", "scaler.pkl", "scaler_utils.py"]

# CSV file paths for saving errors
training_csv_path = os.path.join(base_dir, "training_epoch_errors.csv")
experiment_csv_path = os.path.join(base_dir, "experiment_epoch_errors.csv")

# -------------------------------
# 1) Ground Truth Setup
# -------------------------------
# Global constants for CSTR

Cao = 1 #mol/L
Cbo = 2 #mol/L
Cco = 0 #mol/L
V = 10 #L
Q = 1 #L/s
tau = V/Q #s

#Parameters to tuning to obtain "aggressive" non-linearity
Afo = 10e12
Eaf = 90000 #J/mol
Aro = 10e10
Ear = 80000 #J/mol
R = 8.314 #J/mol

def equations(variables, T):
    Cc, Cb, Ca = variables
    kf = Afo * np.exp(-Eaf/(R*T)) #Arrhenius eqn for forward reaction
    kr = Aro * np.exp(-Ear/(R*T)) #arrhenius eqn for reverse reaction

    eq1 = Cao - Ca + -kf*Ca*(Cb**2)*tau + kr*(Cao-Ca+Cbo-Cb)*tau
    eq2 = Cbo - Cb + -2*kf*Ca*(Cb**2)*tau + 2*kr*(Cao-Ca+Cbo-Cb)*tau
    eq3=Cc-Cao+Ca-Cbo+Cb
    return [eq1, eq2, eq3]

def get_ground_truth(n=30):
    """
    Uses fsolve to compute Ca, Cb, Cc over a range of T from 280..600 K.
    Returns (T_values, Ca_values, Cb_values, Cc_values).
    """
    T_values   = np.linspace(280, 600, n)
    Ca_values  = np.ones(n) * Cao
    Cb_values  = np.ones(n) * Cbo
    Cc_values  = np.ones(n) * Cco

    initial_guess = [Cco, Cbo, Cao]  # [Cc, Cb, Ca]

    for i, T in enumerate(T_values):
        solution, infodict, ier, mesg = fsolve(equations, initial_guess, args=(T,), full_output=True, xtol= 1.0e-11)
        if ier == 1:  # ier == 1 indicates successful convergence
            Cc_values[i], Cb_values[i], Ca_values[i] = solution
        else:
            print(f"Solver did not converge for T = {T}. Message: {mesg}")
            
    kf_arr = Afo * np.exp(-Eaf/(R*T_values))
    kr_arr = Aro * np.exp(-Ear/(R*T_values))
    f = kr_arr*(Cao-Ca_values+Cbo-Cb_values+Cco)
    g = kf_arr*Ca_values*(Cb_values**2)

    return T_values, Ca_values, Cb_values, Cc_values, f, g

# === Helper Functions ===

def copy_files():
    """Copy all necessary files into the target folder."""
    for file in files_to_copy:
        src_file = os.path.join(base_dir, file)
        if os.path.exists(src_file):
            shutil.copy(src_file, os.path.join(target_folder, file))
            print(f"Copied {file} to {target_folder}")
        else:
            print(f"Warning: {file} not found.")

def run_main(model_name, job):
    """
    Run main.py from within the target folder using the specified model_name and job.
    The job argument should be either 'train' or 'experiment'.
    Returns the parsed 'loss_train' value (float) from the final epoch line, or None.
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

    print("Standard Output:")
    print(result.stdout)
    if result.stderr:
        print("Standard Error:")
        print(result.stderr)
    
    # Extract the error (loss_train) from the output
    if job == 'train':
        error = extract_last_epoch_error(result.stdout)
    elif job == 'experiment':
        error = extract_experiment_error(result.stdout)
        print(f"Experiment RMSE3: {error}")
    else:
        error = None
    return error

def extract_experiment_error(output):
    lines = output.splitlines()
    for line in reversed(lines):
        if line.strip().startswith("{") and line.strip().endswith("}"):
            try:
                scores = ast.literal_eval(line.strip())
                return float(scores.get("rmse_inner", scores.get("rmse_total")))
            except (SyntaxError, KeyError, ValueError):
                return None
    return None

def extract_last_epoch_error(output):
    """
    Extract 'loss_train' from the last line in the output that starts with 'epoch:'.
    The output lines are expected to look like:
      epoch: 00050 loss_train: 0.0000 violation_train: 0.0000 loss_val: 0.0000 violation_val: 0.0000
    """
    lines = output.splitlines()
    # Iterate from the end to find the last line that starts with 'epoch:'
    for line in reversed(lines):
        if line.startswith("epoch:"):
            # Split the line into tokens
            parts = line.split()
            data = {}
            i = 0
            while i < len(parts):
                if parts[i].endswith(":"):
                    key = parts[i].replace(":", "")  # e.g., "loss_train"
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
    For the given model_name, perform num_iterations cycles.
    In each cycle:
      1. Copy files.
      2. Run training (job='train') and record training error.
      3. Run experiment (job='experiment') and record experiment error.
      4. Clear the folder (except after the last iteration).
    Returns two lists: training_errors and experiment_errors.
    """
    training_errors = []
    experiment_errors = []
    band_low_rmse3    = []   # <— NEW
    band_high_rmse3   = []   # <— NEW
    
    # Create a figure/axes for all runs (so we can keep adding lines)
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 2) Plot ground truth in dashed lines
    plot_ground_truth(ax)
    
    for i in range(num_iterations):
        print(f"\n=== Iteration {i+1} for model {model_name} ===\n")
        copy_files()
        
        # Training run
        train_error = run_main(model_name, "train")
        if train_error is None:
            train_error = float('nan')
        training_errors.append(train_error)
        
        # Experiment run (with unseen data)
        exp_error = run_main(model_name, "experiment")
        if exp_error is None:
            exp_error = float('nan')
        experiment_errors.append(exp_error)
        
        
        # 1) Retrieve predictions to plot them:
        #    We’ll do a custom function that calls main.py in "plot_mode" or
        #    just reuse "experiment" if you modify the code to return predicted arrays.
        #    For demonstration, let's assume you have a function get_predictions(...).
        X_test, Y_pred = get_predictions(model_name)

        # 2) Plot them on the same figure
        plot_predictions_on_axes(ax, X_test, Y_pred, run_number=i+1, model_name=model_name)
        
        
        # Ground truth on the same grid length
        n = X_test.shape[0]
        T_vals, Ca_vals, Cb_vals, Cc_vals, _, _ = get_ground_truth(n=n)

        # Compute banded RMSE3 and store
        low_rmse, high_rmse = _band_rmse3(X_test.squeeze(), Y_pred, Ca_vals, Cb_vals, Cc_vals)
        band_low_rmse3.append(low_rmse)
        band_high_rmse3.append(high_rmse)
        
        
        # Clear the folder for the next iteration (unless it's the last iteration)
        if i < num_iterations - 1:
            clear_folder()
        
        # After all runs, finalize the figure
    ax.set_title(f"Predictions Over {num_iterations} Runs ({model_name})")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Concentration (mol/L)")
    #ax.legend()
    plot_path = os.path.join(base_dir, f"{model_name}_all_runs_plot.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved combined plot at {plot_path}")
    plt.close()
    
    return training_errors, experiment_errors, band_low_rmse3, band_high_rmse3


def plot_predictions_on_axes(ax, X, Y_pred, run_number, model_name):
    """
    Plots the three output concentrations vs. the single input temperature X.
    ax: the matplotlib axes object
    X: shape [N] or [N,1], the input temperature
    Y_true: shape [N, 3], the ground-truth concentrations
    Y_pred: shape [N, 3], the predicted concentrations
    """
    # Make sure X is 1D
    if len(X.shape) > 1:
        X = X.squeeze(-1)

    # For each of the 3 outputs, pick a color or style
    colors = ['blue', 'red', 'green']
    labels = ['Ca', 'Cb', 'Cc']  # or whatever you prefer

    for i in range(3):
        ax.plot(X, Y_pred[:, i],
                color=colors[i],
                linestyle='-',
                label=f"Run {run_number} {model_name} {labels[i]} (pred)")
        
def plot_ground_truth(ax):
    """
    Plots the ground truth curves for Ca, Cb, Cc over T=280..600 on the given axes.
    """
    n = 30
    T_vals, Ca_vals, Cb_vals, Cc_vals, f, g = get_ground_truth(n=n)

    # Plot in dashed lines
    ax.plot(T_vals, Ca_vals, 'b--', label='Ground truth Ca')
    ax.plot(T_vals, Cb_vals, 'r--', label='Ground truth Cb')
    ax.plot(T_vals, Cc_vals, 'g--', label='Ground truth Cc')
    

def _band_rmse3(T, Y_pred, Ca_true, Cb_true, Cc_true):
    """
    T:       (N,) or (N,1) temperatures
    Y_pred:  (N,>=3) model predictions (first 3 columns are Ca,Cb,Cc)
    *_true:  (N,) ground-truth arrays
    returns: (rmse_low, rmse_high)
    """
    T = T.reshape(-1)
    Yt = np.stack([Ca_true, Cb_true, Cc_true], axis=1)
    err = Y_pred[:, :3] - Yt  # (N,3)

    def rmse_on(mask):
        if mask.sum() == 0:
            return float("nan")
        return float(np.sqrt(np.mean(err[mask]**2)))

    low_mask  = (T <= 300.0)
    high_mask = (T >= 520.0)
    return rmse_on(low_mask), rmse_on(high_mask)


def _welch_and_d(a, b):
    """
    Welch's t-test and Cohen's d (pooled SD) for 1-D arrays 'a' and 'b'.
    Returns (t, p, d, mean_a, mean_b, var_a, var_b).
    """
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    t, p = st.ttest_ind(a, b, equal_var=False)
    # Cohen's d with simple pooled SD (works fine for quick effect size)
    sp = np.sqrt(0.5*(a.var(ddof=1) + b.var(ddof=1)))
    d = (b.mean() - a.mean()) / sp if sp > 0 else 0.0
    return t, p, d, a.mean(), b.mean(), a.var(ddof=1), b.var(ddof=1)


def get_predictions(model_name):
    """
    In practice, you might do something like:
      1. Load the model from disk
      2. Load the test set
      3. Predict
      4. Return (X_test, Y_test, Y_pred)
    For demonstration, we just return fake data below.
    """
    # Example shapes: X_test [N,1], Y_test [N,3], Y_pred [N,3]
    # In your real code, you'd replicate what evaluate_model does, but store predictions.
    scaler_path = os.path.join(os.getcwd(), 'scaler.pkl')
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        print("Scaler file not found at:", scaler_path)
        return None, None

    # Set model hyperparameters
    input_dim = 1
    hidden_dim = 32
    hidden_num = 2
    z0_dim = 5
    z0_inner_dim = 3 

    # Choose model path and parameters based on model_name
    if model_name == "NN":
        model_path = "./new1/models/cstr/NN/0.2/MODELID_0.2_0.pth"
        model_type = "NN"
        # A, B, b = None, None, None
        A, B, b = None, None, None
        z0_inner_dim = 3
        from load_data import load_saved_model, make_prediction
        model = load_saved_model(model_path, model_type,
                             input_dim, hidden_dim, hidden_num,z0_inner_dim,
                             z0_dim, A, B, b)
    elif model_name == "KKThPINN":
        model_path = "./new1/models/cstr/KKThPINN/0.2/MODELID_0.2_0.pth"
        model_type = "KKT"  # Use "KKT" to load KKThPINN
        # A = torch.tensor([[0]])
        # B = torch.tensor([[1, 1, 1]])
        # b = torch.tensor([3])
        
        A = torch.tensor([[0]], dtype=torch.float64)
        B = torch.tensor([[1, 0, 0, -10, 10]], dtype=torch.float64)
        b = torch.tensor([1], dtype=torch.float64)
        
        # A = A.float()
        # B = B.float()
        # b = b.float()
        
        A = A.double()
        B = B.double()
        b = b.double()
        
        from load_data import load_saved_model, make_prediction
        model = load_saved_model(model_path, model_type,
                             input_dim, hidden_dim, hidden_num, z0_inner_dim,
                             z0_dim, A, B, b)   

    else:
        print("Unknown model name:", model_name)
        return None, None

    # # Import the model-loading functions from load_data.py
    # from load_data import load_saved_model, make_prediction

    # # Load the model
    # model = load_saved_model(model_path, model_type, input_dim, hidden_dim, hidden_num, z0_inner_dim, z0_dim, A, B, b)
    model = model.double()
    # model.to(device)
    
    # Define the range of temperatures for prediction (e.g., from 280K to 600K)
    new_temperatures = np.linspace(280, 600, 30)  # 300 points
    # Make predictions using your make_prediction function
    predictions = make_prediction(model, scaler, new_temperatures)
    
    # Return X_test as a column vector and Y_pred
    return new_temperatures.reshape(-1, 1), predictions


# === Main Execution ===

def main():
    num_iterations = 50  # Perform 50 cycles for each model
    
    # Run experiments for NN
    print("\n******** Running experiments for NN ********\n")
    nn_training_errors, nn_experiment_errors, nn_low, nn_high = run_model_experiments("NN", num_iterations)
    
    # Clear folder before next model (optional)
    clear_folder()
    
    # Run experiments for KKThPINN (KKT)
    print("\n******** Running experiments for KKThPINN ********\n")
    kkt_training_errors, kkt_experiment_errors, kkt_low, kkt_high = run_model_experiments("KKThPINN", num_iterations)
    
    # Save training errors for both models in one CSV file
    training_df = pd.DataFrame({
        "Iteration": range(1, num_iterations + 1),
        "NN_Training_Error": nn_training_errors,
        "KKThPINN_Training_Error": kkt_training_errors
    })
    training_df.to_csv(training_csv_path, index=False)
    print(f"\nTraining errors saved at: {training_csv_path}")
    
    # Save experiment errors for both models in another CSV file
    experiment_df = pd.DataFrame({
        
        "NN_Experiment_Error": nn_experiment_errors,
        "KKThPINN_Experiment_Error": kkt_experiment_errors
    })
    experiment_df.to_csv(experiment_csv_path, index=False)
    print(f"Experiment errors saved at: {experiment_csv_path}")
    
    
    # ---- NEW: banded RMSE₃ summary & Welch tests ----
    nn_low  = np.array(nn_low, dtype=float);   kkt_low  = np.array(kkt_low, dtype=float)
    nn_high = np.array(nn_high, dtype=float);  kkt_high = np.array(kkt_high, dtype=float)

    def _print_band(name, a, b):
        t, p, d, ma, mb, va, vb = _welch_and_d(a, b)
        print(f"\n[{name}] RMSE₃  —  NN: mean={ma:.6f}, var={va:.6e} | KKThPINN: mean={mb:.6f}, var={vb:.6e}")
        print(f"Welch t-test: t={t:.3f}, p={p:.4f} | Cohen's d={d:.3f}")
        if p < 0.05:
            print("=> Significant difference.")
        else:
            print("=> No significant difference.")

    _print_band("LOW (T ≤ 300 K)",  nn_low,  kkt_low)
    _print_band("HIGH (T ≥ 520 K)", nn_high, kkt_high)

if __name__ == "__main__":
    main()
