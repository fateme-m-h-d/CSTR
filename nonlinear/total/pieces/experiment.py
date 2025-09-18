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

# === Configuration ===
# base_dir = "./CSTR/nonlinear/total"
base_dir = os.getcwd()   # must be run from within total/
target_folder = os.path.join(base_dir, "new1")
os.makedirs(target_folder, exist_ok=True)

# List of source files to be copied into the folder
files_to_copy = ["main.py", "train.py", "models.py", "utils.py", "curves.py", "data.csv", "scaler.pkl"]

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
        solution, infodict, ier, mesg = fsolve(equations, initial_guess, args=(T,), full_output=True)
        if ier == 1:  # ier == 1 indicates successful convergence
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
    else:
        error = None
    return error

def extract_experiment_error(output):
    lines = output.splitlines()
    for line in reversed(lines):
        if line.strip().startswith("{") and line.strip().endswith("}"):
            try:
                scores = ast.literal_eval(line.strip())
                return float(scores['rmse_total'])
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
    
    return training_errors, experiment_errors


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
    T_vals, Ca_vals, Cb_vals, Cc_vals = get_ground_truth(n=n)

    # Plot in dashed lines
    ax.plot(T_vals, Ca_vals, 'b--', label='Ground truth Ca')
    ax.plot(T_vals, Cb_vals, 'r--', label='Ground truth Cb')
    ax.plot(T_vals, Cc_vals, 'g--', label='Ground truth Cc')

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
    z0_dim = 3

    # Choose model path and parameters based on model_name
    if model_name == "NN":
        model_path = "./new1/models/cstr/NN/0.2/MODELID_0.2_0.pth"
        model_type = "NN"
        # A, B, b = None, None, None
        A_list, B_list, b_list = None, None, None
    elif model_name == "KKThPINN":
        model_path = "./new1/models/cstr/KKThPINN/0.2/MODELID_0.2_0.pth"
        model_type = "KKT"  # Use "KKT" to load KKThPINN
        # A = torch.tensor([[0]])
        # B = torch.tensor([[1, 1, 1]])
        # b = torch.tensor([3])
        
        A_list = [
            torch.tensor([[- 0.00301071551554214]], dtype=torch.float64),
            torch.tensor([[- 0.0287912896000818]], dtype=torch.float64),
            torch.tensor([[- 0.0589977043951539]], dtype=torch.float64),
            torch.tensor([[- 0.175928980736293]], dtype=torch.float64),
            torch.tensor([[- 2.22034336021516]], dtype=torch.float64),
            torch.tensor([[- 19.068831993137]], dtype=torch.float64),
            torch.tensor([[- 67.0529314068883]], dtype=torch.float64),
            torch.tensor([[- 147.195966751539]], dtype=torch.float64),
            torch.tensor([[- 242.760891902504]], dtype=torch.float64),
            torch.tensor([[- 356.302755070124]], dtype=torch.float64),
            torch.tensor([[- 496.399591699859]], dtype=torch.float64),
            torch.tensor([[- 650.139480396354]], dtype=torch.float64)
            
        ]
        B_list = [
            torch.tensor([[ -1.02825709223637, - 0.0282570922363733, 0]], dtype=torch.float64),
            torch.tensor([[ -1.54923960097239, - 0.549239600972388, 0]], dtype=torch.float64),
            torch.tensor([[-6.41562952963354, - 5.41562952963354, 0]], dtype=torch.float64),
            torch.tensor([[-47.4935472673711, - 46.4935472673712, 0]], dtype=torch.float64),
            torch.tensor([[-1002.53593524019, - 1001.53593524029, 0]], dtype=torch.float64),
            torch.tensor([[-11478.9269831455, - 11477.9269831524, 0]], dtype=torch.float64),
            torch.tensor([[-48213.8668993521, - 48212.866899362, 0]], dtype=torch.float64),
            torch.tensor([[ -119069.962944269, - 119068.962945171, 0]], dtype=torch.float64),
            torch.tensor([[-212313.514625447, - 212312.514623723, 0]], dtype=torch.float64),
            torch.tensor([[-331422.775549942, - 331421.775550256, 0]], dtype=torch.float64),
            torch.tensor([[-487637.901829002, - 487636.901838071, 0]], dtype=torch.float64),
            torch.tensor([[-668298.267688647, - 668297.267673516, 0]], dtype=torch.float64)
            
        ]
        b_list = [
            torch.tensor([-1.93327845562254], dtype=torch.float64),
            torch.tensor([-11.1707510081181], dtype=torch.float64),
            torch.tensor([-29.674961918774], dtype=torch.float64),
            torch.tensor([-131.782655072763], dtype=torch.float64),
            torch.tensor([-2204.88922143194], dtype=torch.float64),
            torch.tensor([-22375.8507664129], dtype=torch.float64),
            torch.tensor([-87505.3569136546], dtype=torch.float64),
            torch.tensor([-206400.451290504], dtype=torch.float64),
            torch.tensor([-357218.284591596], dtype=torch.float64),
            torch.tensor([-544832.512463095], dtype=torch.float64),
            torch.tensor([-785540.247635705], dtype=torch.float64),
            torch.tensor([-1058769.64213322], dtype=torch.float64)
        ]
        
        A_list = [A.double() for A in A_list]
        B_list = [B.double() for B in B_list]
        b_list = [b.double() for b in b_list]

    else:
        print("Unknown model name:", model_name)
        return None, None

    # Import the model-loading functions from load_data.py
    from load_data import load_saved_model, make_prediction

    # Load the model
    model = load_saved_model(model_path, model_type, input_dim, hidden_dim, hidden_num, z0_dim, A_list, B_list, b_list)
    
    # Define the range of temperatures for prediction (e.g., from 280K to 600K)
    new_temperatures = np.linspace(280, 600, 30)  # 300 points
    # Make predictions using your make_prediction function
    predictions = make_prediction(model, scaler, new_temperatures)
    
    # Return X_test as a column vector and Y_pred
    return new_temperatures.reshape(-1, 1), predictions


# === Main Execution ===

def main():
    num_iterations = 50 # Perform 50 cycles for each model
    
    # Run experiments for NN
    print("\n******** Running experiments for NN ********\n")
    nn_training_errors, nn_experiment_errors = run_model_experiments("NN", num_iterations)
    
    # Clear folder before next model (optional)
    clear_folder()
    
    # Run experiments for KKThPINN (KKT)
    print("\n******** Running experiments for KKThPINN ********\n")
    kkt_training_errors, kkt_experiment_errors = run_model_experiments("KKThPINN", num_iterations)
    
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

if __name__ == "__main__":
    main()
