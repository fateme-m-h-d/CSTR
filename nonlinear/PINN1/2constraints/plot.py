import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

summary = pd.read_csv("nn_pinn_50_compare_summary.csv")

models = summary["model"].tolist()
x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(7, 5))

plt.bar(
    x - width / 2,
    summary["final_train_violation_mean"],
    width,
    label="Final training violation",
)

plt.bar(
    x + width / 2,
    summary["inference_violation_mean"],
    width,
    label="Inference violation",
)

plt.errorbar(
    x - width / 2,
    summary["final_train_violation_mean"],
    yerr=summary["final_train_violation_ci95"],
    fmt="none",
    capsize=4,
)

plt.errorbar(
    x + width / 2,
    summary["inference_violation_mean"],
    yerr=summary["inference_violation_ci95"],
    fmt="none",
    capsize=4,
)

plt.xticks(x, models)
plt.ylabel("Mean absolute constraint violation")
plt.yscale("log")
plt.legend()
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("nn_pinn_train_vs_inference_violation_relabelled.pdf", dpi=300)
plt.close()