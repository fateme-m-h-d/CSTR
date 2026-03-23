import subprocess
import shutil
import os
import sys

SCENARIOS = [5, 8, 10, 15, 20, 40, 60, 80, 100]

for n in SCENARIOS:
    print("=" * 60)
    print(f"Running scenario n_inner_per_region = {n}")
    print("=" * 60)

    # Generate new data.csv and ABb_matrices.csv for this 1D scenario
    subprocess.run(
        [sys.executable, "linearization.py", "--n_inner_per_region", str(n), "--seed", "0"],
        check=True
    )

    # Keep copies of scenario-specific inputs
    if os.path.exists("data.csv"):
        shutil.copy2("data.csv", f"data_ninner_{n}.csv")

    if os.path.exists("ABb_matrices.csv"):
        shutil.copy2("ABb_matrices.csv", f"ABb_matrices_ninner_{n}.csv")

    if os.path.exists("lin_params.csv"):
        shutil.copy2("lin_params.csv", f"lin_params_ninner_{n}.csv")

    # Run the experiment workflow
    subprocess.run(
        [sys.executable, "experiment2.py"],
        check=True
    )

    # Save scenario-specific outputs
    if os.path.exists("results_by_samples.csv"):
        shutil.copy2("results_by_samples.csv", f"results_by_samples_ninner_{n}.csv")

    if os.path.exists("experiment_epoch_errors.csv"):
        shutil.copy2("experiment_epoch_errors.csv", f"experiment_epoch_errors_ninner_{n}.csv")

    if os.path.exists("training_epoch_errors.csv"):
        shutil.copy2("training_epoch_errors.csv", f"training_epoch_errors_ninner_{n}.csv")

print("All scenarios finished.")