import os
import shutil
import subprocess
import numpy as np
import pandas as pd

# === Configuration ===
base_dir = "C:/Users/Fateme/Desktop/Research/CSTR/noise/new12"
target_folder = os.path.join(base_dir, "new20")
os.makedirs(target_folder, exist_ok=True)

# List of source files to be copied into the folder
files_to_copy = ["main.py", "train.py", "models.py", "utils.py", "curves.py", "data.csv"]

# CSV file paths for saving errors
training_csv_path = os.path.join(base_dir, "ONE_training_epoch_errors.csv")
experiment_csv_path = os.path.join(base_dir, "ONE_experiment_epoch_errors.csv")

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
        
        # Clear the folder for the next iteration (unless it's the last iteration)
        if i < num_iterations - 1:
            clear_folder()
    
    return training_errors, experiment_errors

# === Main Execution ===

def main():
    num_iterations = 50  # Perform 10 cycles for each model
    
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
        "Iteration": range(1, num_iterations + 1),
        "NN_Experiment_Error": nn_experiment_errors,
        "KKThPINN_Experiment_Error": kkt_experiment_errors
    })
    experiment_df.to_csv(experiment_csv_path, index=False)
    print(f"Experiment errors saved at: {experiment_csv_path}")

if __name__ == "__main__":
    main()
