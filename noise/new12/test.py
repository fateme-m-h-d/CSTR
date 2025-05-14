import os
import shutil
import subprocess
import numpy as np
import pandas as pd

# === Configuration ===
base_dir = "C:/Users/Fateme/Desktop/Research/CSTR/noise/new12"
target_folder = os.path.join(base_dir, "new22")
os.makedirs(target_folder, exist_ok=True)

# List of source files to be copied into the folder
files_to_copy = ["main.py", "train.py", "models.py", "utils.py", "curves.py", "data.csv"]

# CSV file paths for saving errors
training_csv_path = os.path.join(base_dir, "2_training_epoch_errors.csv")
experiment_csv_path = os.path.join(base_dir, "2_experiment_epoch_errors.csv")

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
    last_epoch_error = extract_last_epoch_error(result.stdout)
    return last_epoch_error

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

def run_model_experiments(model_name):
    """
    For the given model_name, run training and experiment once.
    1. Copy files.
    2. Run training (job='train') and record training error.
    3. Run experiment (job='experiment') and record experiment error.
    Returns the training error and experiment error.
    """
    copy_files()
    
    # Training run
    train_error = run_main(model_name, "train")
    if train_error is None:
        train_error = float('nan')
    
    # Experiment run (with unseen data)
    exp_error = run_main(model_name, "experiment")
    if exp_error is None:
        exp_error = float('nan')
    
    return train_error, exp_error

# === Main Execution ===

def main():
    # Run experiments for NN (only one iteration)
    print("\n******** Running experiments for NN ********\n")
    nn_train_error, nn_exp_error = run_model_experiments("KKThPINN")
    
    # Save training error for NN in a CSV file
    training_df = pd.DataFrame({
        "Model": ["KKThPINN"],
        "Training_Error": [nn_train_error]
    })
    training_df.to_csv(training_csv_path, index=False)
    print(f"\nTraining error saved at: {training_csv_path}")
    
    # Save experiment error for NN in another CSV file
    experiment_df = pd.DataFrame({
        "Model": ["KKThPINN"],
        "Experiment_Error": [nn_exp_error]
    })
    experiment_df.to_csv(experiment_csv_path, index=False)
    print(f"Experiment error saved at: {experiment_csv_path}")

if __name__ == "__main__":
    main()
