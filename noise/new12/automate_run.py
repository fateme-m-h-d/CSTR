# import os
# import shutil
# import subprocess
# import numpy as np
# import pandas as pd

# # Configuration
# base_dir = "C:/Users/Fateme/Desktop/Research/CSTR/noise/new12"
# target_folder = os.path.join(base_dir, "new20")
# os.makedirs(target_folder, exist_ok=True)

# # List of source files to be copied into the folder
# files_to_copy = ["main.py", "train.py", "models.py", "utils.py", "curves.py", "data.csv"]

# # Path to the CSV file where errors will be saved
# error_csv_path = os.path.join(base_dir, "epoch_errors.csv")

# def copy_files():
#     """Copy all necessary files into the target folder."""
#     for file in files_to_copy:
#         src_file = os.path.join(base_dir, file)
#         if os.path.exists(src_file):
#             shutil.copy(src_file, os.path.join(target_folder, file))
#             print(f"Copied {file} to {target_folder}")
#         else:
#             print(f"Warning: {file} not found.")

# def run_main(iteration):
#     """Run the main.py file from within the target folder."""
#     main_file = os.path.join(target_folder, "main.py")
#     if os.path.exists(main_file):
#         print("Running main.py ...")
#         # Using subprocess.run to execute the script.
#         args = [
#             "python", "main.py",
#             "--model", "NN",
#             "--model_id", "MODELID",
#             "--dataset_type", "cstr",
#             "--dataset_path", "./data.csv",
#             "--job", "train"
#         ]
#         result = subprocess.run(args, capture_output=True, text=True, cwd=target_folder)
#         print("Standard Output:")
#         print(result.stdout)
#         if result.stderr:
#             print("Standard Error:")
#             print(result.stderr)
        
#         # Extract the error of the last epoch from the output (assuming it's in the stdout)
#         last_epoch_error = extract_last_epoch_error(result.stdout)
#         if last_epoch_error is not None:
#             save_error_to_csv(iteration, last_epoch_error)
#     else:
#         print("main.py not found in the target folder.")

# def extract_last_epoch_error(output):
#     """Extract the error of the last epoch from the output."""
#     # Implement the logic to extract the error from the output
#     # This is a placeholder implementation and should be adjusted based on the actual output format
#     lines = output.splitlines()
#     for line in reversed(lines):
#         if "Epoch" in line and "Error" in line:
#             # Assuming the line format is "Epoch X: Error Y"
#             parts = line.split(":")
#             if len(parts) == 2:
#                 try:
#                     return float(parts[1].strip())
#                 except ValueError:
#                     pass
#     return None

# def save_error_to_csv(iteration, error):
#     """Save the error to the CSV file."""
#     if not os.path.exists(error_csv_path):
#         # Create the CSV file with headers if it doesn't exist
#         with open(error_csv_path, "w") as f:
#             f.write("Iteration,Error\n")
    
#     # Append the error to the CSV file
#     with open(error_csv_path, "a") as f:
#         f.write(f"{iteration},{error}\n")

# def clear_folder():
#     """Empty the target folder by deleting its contents."""
#     for filename in os.listdir(target_folder):
#         file_path = os.path.join(target_folder, filename)
#         try:
#             if os.path.isfile(file_path) or os.path.islink(file_path):
#                 os.unlink(file_path)
#                 print(f"Deleted file: {file_path}")
#             elif os.path.isdir(file_path):
#                 shutil.rmtree(file_path)
#                 print(f"Deleted directory: {file_path}")
#         except Exception as e:
#             print(f"Failed to delete {file_path}. Reason: {e}")

# # Define how many times you want to run the cycle
# num_iterations = 2  # Adjust as needed

# for i in range(num_iterations):
#     print(f"\n=== Iteration {i+1} ===\n")
#     copy_files()  # Copy the original files into the target folder
    
#     run_main(i + 1)    # Run the main file from the target folder
    
#     if i < num_iterations - 1:
#         clear_folder()  # Empty the folder before the next iteration

# # Load final epoch loss values from 5 runs for KKThPINN and NN models
# # loss_KKThPINN = [
# #     np.load(os.path.join(target_folder, f'./data/learning_curves/cstr/KKThPINN/0.2/MODELID_train_losses_run{i}.npy'))[-1]
# #     for i in range(1)
# # ]
# loss_NN = [
#     np.load(os.path.join(target_folder, f'C:/Users/Fateme/Desktop/Research/CSTR/noise/new12/new20/data/learning_curves/cstr/NN/0.2/MODELID_train_losses_run0.npy'))[-1]
#     for i in range(1)
# ]

# # print("Final epoch loss values for KKThPINN:", loss_KKThPINN)
# print("Final epoch loss values for NN:", loss_NN)


import os
import shutil
import subprocess
import numpy as np
import pandas as pd

# Configuration
base_dir = "C:/Users/Fateme/Desktop/Research/CSTR/noise/new12"
target_folder = os.path.join(base_dir, "new20")
os.makedirs(target_folder, exist_ok=True)

# List of source files to be copied into the folder
files_to_copy = ["main.py", "train.py", "models.py", "utils.py", "curves.py", "data.csv"]

# Path to the CSV file where combined errors will be saved
error_csv_path = os.path.join(base_dir, "epoch_errors.csv")

def copy_files():
    """Copy all necessary files into the target folder."""
    for file in files_to_copy:
        src_file = os.path.join(base_dir, file)
        if os.path.exists(src_file):
            shutil.copy(src_file, os.path.join(target_folder, file))
            print(f"Copied {file} to {target_folder}")
        else:
            print(f"Warning: {file} not found.")

def run_main(model_name):
    """
    Run main.py from within the target folder using the specified model_name.
    Returns the parsed 'loss_train' value (float) from the final epoch line,
    or None if not found.
    """
    main_file = os.path.join(target_folder, "main.py")
    if not os.path.exists(main_file):
        print("main.py not found in the target folder.")
        return None
    
    print(f"Running main.py with model={model_name} ...")
    args = [
        "python", "main.py",
        "--model", model_name,
        "--model_id", "MODELID",
        "--dataset_type", "cstr",
        "--dataset_path", "./data.csv",
        "--job", "train"
    ]
    result = subprocess.run(args, capture_output=True, text=True, cwd=target_folder)

    # Print subprocess output for debugging
    print("Standard Output:")
    print(result.stdout)
    if result.stderr:
        print("Standard Error:")
        print(result.stderr)
    
    # Extract the error (loss_train) of the last epoch from the output
    last_epoch_error = extract_last_epoch_error(result.stdout)
    return last_epoch_error

def extract_last_epoch_error(output):
    """
    Extract 'loss_train' from the last line in the output that starts with 'epoch:'.
    The output lines typically look like:
      epoch: 00050 loss_train: 0.0000 violation_train: 0.0000 loss_val: 0.0000 violation_val: 0.0000
    """
    lines = output.splitlines()
    # Iterate from the end to find the last line that starts with 'epoch:'
    for line in reversed(lines):
        if line.startswith("epoch:"):
            # Example:
            # "epoch: 00050 loss_train: 0.0000 violation_train: 0.0000 loss_val: 0.0000 violation_val: 0.0000"
            parts = line.split()
            # parts might be:
            # ["epoch:", "00050", "loss_train:", "0.0000", "violation_train:", "0.0000",
            #  "loss_val:", "0.0000", "violation_val:", "0.0000"]
            
            data = {}
            i = 0
            while i < len(parts):
                if parts[i].endswith(":"):
                    key = parts[i].replace(":", "")  # e.g., "loss_train"
                    if i + 1 < len(parts):
                        val = parts[i + 1]
                        data[key] = val
                    i += 2
                else:
                    i += 1
            
            # Now data might look like:
            # {
            #   "epoch": "00050",
            #   "loss_train": "0.0000",
            #   "violation_train": "0.0000",
            #   "loss_val": "0.0000",
            #   "violation_val": "0.0000"
            # }
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

def run_experiments(model_name, num_iterations):
    """
    Run main.py for the specified model_name for num_iterations times.
    Returns a list of error values (one per iteration).
    """
    errors = []
    for i in range(num_iterations):
        print(f"\n=== Iteration {i+1} for model {model_name} ===\n")
        copy_files()
        err = run_main(model_name)
        if err is None:
            err = float('nan')  # or 0.0, or however you'd like to handle missing
        errors.append(err)
        
        # Clear folder unless we're on the last iteration
        if i < num_iterations - 1:
            clear_folder()
    return errors

# ----------------------------------------------------
# Main Execution: Compare NN vs. KKT over 20 iterations
# ----------------------------------------------------
num_iterations = 50

# 1) Run 20 iterations for NN
nn_errors = run_experiments("NN", num_iterations)

# 2) Run 20 iterations for KKT
kkt_errors = run_experiments("KKThPINN", num_iterations)

# Create a DataFrame with Iteration, NN_Error, KKT_Error
df = pd.DataFrame({
    "Iteration": range(1, num_iterations + 1),
    "NN_Error": nn_errors,
    "KKT_Error": kkt_errors
})

# Save the combined results to CSV
df.to_csv(error_csv_path, index=False)
print(f"\nAll done! CSV saved at: {error_csv_path}")
