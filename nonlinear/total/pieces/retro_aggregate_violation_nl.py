# retro_aggregate_violation_nl.py
# Aggregate per-run nonlinear-violation results (NO re-eval).
# Reads:   retro_violation_original_nonlinear.csv  (per-run rows)
# Updates: results_by_samples_master.csv           (adds <N>_VIOL_NL mean columns)

from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path.cwd()
RETRO = BASE / "retro_violation_original_nonlinear.csv"
MASTER = BASE / "results_by_samples_master.csv"

def main():
    if not RETRO.exists():
        raise FileNotFoundError(f"Per-run CSV not found: {RETRO}")

    retro = pd.read_csv(RETRO)
    if retro.empty:
        print("[AGG] Retro CSV is empty. Nothing to aggregate.")
        return

    required = {"num_samples", "model", "violation_original_nonlinear"}
    missing = required - set(retro.columns)
    if missing:
        raise ValueError(f"Retro CSV missing required columns: {missing}")

    # ensure types and keep only needed cols
    retro = retro[["model", "num_samples", "violation_original_nonlinear"]].copy()
    retro["num_samples"] = retro["num_samples"].astype(int)

    # Load/create master
    if MASTER.exists():
        master = pd.read_csv(MASTER)
    else:
        master = pd.DataFrame({"Model": ["NN", "KKThPINN"]})

    # Ensure both model rows exist (won't duplicate existing)
    for m in ["NN", "KKThPINN"]:
        if not (master["Model"] == m).any():
            master = pd.concat([master, pd.DataFrame({"Model": [m]})], ignore_index=True)

    # Compute per-(model, num_samples) mean; pandas mean skips NaNs
    grouped = (
        retro.groupby(["model", "num_samples"])["violation_original_nonlinear"]
             .mean()
             .reset_index()
    )

    # Update/add <N>_VIOL_NL columns with the means
    for _, row in grouped.iterrows():
        model = row["model"]
        n = int(row["num_samples"])
        col = f"{n}_VIOL_NL"
        if col not in master.columns:
            master[col] = np.nan
        master.loc[master["Model"].eq(model), col] = float(row["violation_original_nonlinear"])

    # Nice column order: Model first, then by increasing N, then metric name
    def sort_key(c: str):
        if c == "Model":
            return (-1, "")
        if "_" not in c:
            return (10**9, c)
        try:
            n, rest = c.split("_", 1)
            return (int(n), rest)
        except Exception:
            return (10**9, c)

    cols = ["Model"] + sorted([c for c in master.columns if c != "Model"], key=sort_key)
    master = master[cols]

    master.to_csv(MASTER, index=False)
    print(f"[AGG] Updated {MASTER} with *_VIOL_NL means from {RETRO}")

if __name__ == "__main__":
    main()
