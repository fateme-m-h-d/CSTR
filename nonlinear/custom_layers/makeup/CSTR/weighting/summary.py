import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


RESULT_DIR = Path("./data/tables/cstr/KKThPINN/0.2")
VAL_RATIO = 0.2
RUNS = 10

CASES = {
    "softplus_eps": "KKThPINN_softplus_eps",
    "relu": "KKThPINN_relu",
}


def read_one_result(path):
    df = pd.read_csv(path, header=None, names=["key", "value"])
    d = dict(zip(df["key"], df["value"]))

    return {
        "rmse_total": float(d["rmse_total"]),
        "rmse_inner": float(d["rmse_inner"]),
        "violation": float(d["violation"]),
    }


rows = []

for case_name, model_id in CASES.items():
    for run in range(RUNS):
        path = RESULT_DIR / f"{model_id}_{VAL_RATIO}_{run}.csv"

        if not path.exists():
            print(f"Missing: {path}")
            continue

        result = read_one_result(path)
        result["case"] = case_name
        result["model_id"] = model_id
        result["run"] = run
        rows.append(result)


df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError("No result files were found. Check RESULT_DIR, model_id names, and RUNS.")


# Save all per-run results
df.to_csv("all_runs_results.csv", index=False)


# Summary: mean and standard deviation
summary = (
    df.groupby("case")[["rmse_total", "rmse_inner", "violation"]]
    .agg(["mean", "std", "min", "max"])
)

summary.to_csv("summary_results.csv")

print("\nPer-run results:")
print(df)

print("\nSummary:")
print(summary)


# -------------------------
# Visualization 1: RMSE
# -------------------------
rmse_stats = df.groupby("case")["rmse_total"].agg(["mean", "std"])

plt.figure(figsize=(6, 4))
plt.bar(rmse_stats.index, rmse_stats["mean"], yerr=rmse_stats["std"], capsize=5)
plt.ylabel("RMSE total")
plt.xlabel("z4 activation")
plt.title("Average RMSE over runs")
plt.tight_layout()
plt.savefig("rmse_total_comparison.png", dpi=300)
plt.close()


# -------------------------
# Visualization 2: violation
# -------------------------
viol_stats = df.groupby("case")["violation"].agg(["mean", "std"])

plt.figure(figsize=(6, 4))
plt.bar(viol_stats.index, viol_stats["mean"], yerr=viol_stats["std"], capsize=5)
plt.ylabel("Mean nonlinear violation")
plt.xlabel("z4 activation")
plt.title("Average violation over runs")
plt.yscale("log")
plt.tight_layout()
plt.savefig("violation_comparison.png", dpi=300)
plt.close()


# -------------------------
# Visualization 3: per-run RMSE
# -------------------------
plt.figure(figsize=(7, 4))

for case_name in df["case"].unique():
    sub = df[df["case"] == case_name]
    plt.plot(sub["run"], sub["rmse_total"], marker="o", label=case_name)

plt.xlabel("Run")
plt.ylabel("RMSE total")
plt.title("RMSE across runs")
plt.legend()
plt.tight_layout()
plt.savefig("rmse_total_per_run.png", dpi=300)
plt.close()


# -------------------------
# Visualization 4: per-run violation
# -------------------------
plt.figure(figsize=(7, 4))

for case_name in df["case"].unique():
    sub = df[df["case"] == case_name]
    plt.plot(sub["run"], sub["violation"], marker="o", label=case_name)

plt.xlabel("Run")
plt.ylabel("Mean nonlinear violation")
plt.yscale("log")
plt.title("Violation across runs")
plt.legend()
plt.tight_layout()
plt.savefig("violation_per_run.png", dpi=300)
plt.close()


print("\nSaved:")
print("all_runs_results.csv")
print("summary_results.csv")
print("rmse_total_comparison.png")
print("violation_comparison.png")
print("rmse_total_per_run.png")
print("violation_per_run.png")