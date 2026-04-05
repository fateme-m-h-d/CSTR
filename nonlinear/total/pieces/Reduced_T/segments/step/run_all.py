import subprocess
import shutil
import os
import sys

SEGMENT_SCENARIOS = [1, 2, 3, 5, 7, 9, 11]
TOTAL_POINTS = 150   # choose any fixed number >= union of all centers/edges
SEED = 0

# -------------------------------------------------
# Build one shared dataset once
# -------------------------------------------------
print("=" * 60)
print("Building one shared dataset for all segment scenarios")
print("=" * 60)

subprocess.run(
    [
        sys.executable, "generate_data.py",
        "--n_total_points", str(TOTAL_POINTS),
        "--seed", str(SEED)
    ],
    check=True
)

if os.path.exists("data.csv"):
    shutil.copy2("data.csv", "data_shared.csv")

# -------------------------------------------------
# For each scenario: same data, different linearization
# -------------------------------------------------
for nseg in SEGMENT_SCENARIOS:
    print("=" * 60)
    print(f"Running scenario n_segments = {nseg}")
    print("=" * 60)

    subprocess.run(
        [
            sys.executable, "linearization.py",
            "--nT_regions", str(nseg)
        ],
        check=True
    )

    if os.path.exists("ABb_matrices.csv"):
        shutil.copy2("ABb_matrices.csv", f"ABb_matrices_nseg_{nseg}.csv")

    if os.path.exists("lin_params.csv"):
        shutil.copy2("lin_params.csv", f"lin_params_nseg_{nseg}.csv")

    # same data every time, but save a copy anyway for bookkeeping
    if os.path.exists("data.csv"):
        shutil.copy2("data.csv", f"data_nseg_{nseg}.csv")

    subprocess.run(
        [sys.executable, "experiment2.py"],
        check=True
    )

    if os.path.exists("results_by_samples.csv"):
        shutil.copy2("results_by_samples.csv", f"results_by_samples_nseg_{nseg}.csv")

    if os.path.exists("experiment_epoch_errors.csv"):
        shutil.copy2("experiment_epoch_errors.csv", f"experiment_epoch_errors_nseg_{nseg}.csv")

    if os.path.exists("training_epoch_errors.csv"):
        shutil.copy2("training_epoch_errors.csv", f"training_epoch_errors_nseg_{nseg}.csv")

print("All scenarios finished.")