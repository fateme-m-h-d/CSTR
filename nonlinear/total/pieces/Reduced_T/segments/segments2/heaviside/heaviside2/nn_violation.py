import os
import glob
import numpy as np
import pandas as pd


TABLE_DIR = "./data/tables/cstr/NN/0.2"
OUT_PATH = "nn_violation_baseline.txt"


def main():
    paths = sorted(glob.glob(os.path.join(TABLE_DIR, "*.csv")))

    if len(paths) == 0:
        raise RuntimeError(f"No NN report CSV files found in: {TABLE_DIR}")

    values = []

    for path in paths:
        df = pd.read_csv(path, header=None, names=["key", "value"])

        row = df.loc[df["key"] == "violation_original_nonlinear", "value"]

        if len(row) == 0:
            print(f"Skipping {path}: no violation_original_nonlinear found")
            continue

        value = float(row.iloc[0])

        if np.isfinite(value):
            values.append(value)

    if len(values) == 0:
        raise RuntimeError("No valid NN violation_original_nonlinear values found.")

    baseline = float(np.mean(values))

    np.savetxt(OUT_PATH, np.array([baseline]))

    print("\n=== NN violation baseline ===")
    print(f"Number of trained NN runs used: {len(values)}")
    print(f"Mean NN original nonlinear violation: {baseline:.8e}")
    print(f"Saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()