import subprocess
import shutil
import os
import sys

SEGMENT_SCENARIOS = [1, 2, 3, 5, 7, 9, 11]   # number of T segments
NC_REGIONS = 3
FIXED_TOTAL_POINTS = 170
SEED = 0

# 1) Generate one fixed dataset once
subprocess.run(
    [sys.executable, "generate_data.py",
     "--n_total_points", str(FIXED_TOTAL_POINTS),
     "--seed", str(SEED),
     "--out_csv", "data.csv"],
    check=True
)

shutil.copy2("data.csv", "data_fixed.csv")

# 2) Loop over segment scenarios
for nT in SEGMENT_SCENARIOS:
    print("=" * 60)
    print(f"Running scenario nT_regions = {nT}, nC_regions = {NC_REGIONS}")
    print("=" * 60)

    subprocess.run(
        [sys.executable, "linearization.py",
         "--nT_regions", str(nT),
         "--nC_regions", str(NC_REGIONS)],
        check=True
    )

    subprocess.run(
        [sys.executable, "experiment2.py"],
        check=True
    )

    total_regions = nT * NC_REGIONS

    shutil.copy2("ABb_matrices.csv", f"ABb_matrices_nseg_{nT}.csv")
    shutil.copy2("lin_params.csv", f"lin_params_nseg_{nT}.csv")
    shutil.copy2("region_edges.npz", f"region_edges_nseg_{nT}.npz")

    if os.path.exists("experiment_epoch_errors.csv"):
        shutil.copy2("experiment_epoch_errors.csv", f"experiment_epoch_errors_nseg_{nT}.csv")

    if os.path.exists("training_epoch_errors.csv"):
        shutil.copy2("training_epoch_errors.csv", f"training_epoch_errors_nseg_{nT}.csv")

    # optional mirror with total regions in name too
    if os.path.exists("experiment_epoch_errors.csv"):
        shutil.copy2("experiment_epoch_errors.csv", f"experiment_epoch_errors_regions_{total_regions}.csv")

print("All segment scenarios finished.")