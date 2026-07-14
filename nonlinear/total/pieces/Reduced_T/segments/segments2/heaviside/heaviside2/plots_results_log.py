import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULT_ROOT = Path("scenario_results")
OUTDIR = RESULT_ROOT / "plots_log"
OUTDIR.mkdir(parents=True, exist_ok=True)

summary_mean_std_path = RESULT_ROOT / "summary_mean_std.csv"
summary_all_repeats_path = RESULT_ROOT / "summary_all_repeats.csv"

# ------------------------------------------------------------
# Load summary
# ------------------------------------------------------------
if summary_mean_std_path.exists():
    df = pd.read_csv(summary_mean_std_path)
    already_mean_std = True
    print(f"Loaded: {summary_mean_std_path}")
elif summary_all_repeats_path.exists():
    raw = pd.read_csv(summary_all_repeats_path)
    already_mean_std = False
    print(f"Loaded: {summary_all_repeats_path}")

    group_cols = ["segmentation_type", "n_regions"]

    metric_cols = [
        c for c in raw.columns
        if c not in ["segmentation_type", "n_regions", "repeat", "model_id"]
    ]

    df = (
        raw.groupby(group_cols)[metric_cols]
        .agg(["mean", "std", "min", "max"])
    )

    df.columns = [f"{metric}_{stat}" for metric, stat in df.columns]
    df = df.reset_index()

    df.to_csv(summary_mean_std_path, index=False)
    print(f"Saved computed mean/std summary to: {summary_mean_std_path}")
else:
    raise FileNotFoundError(
        "Could not find scenario_results/summary_mean_std.csv "
        "or scenario_results/summary_all_repeats.csv"
    )

# ------------------------------------------------------------
# Metrics to plot
# ------------------------------------------------------------
metrics = [
    ("projection_projection_mae_scaled", "Projection MAE, scaled"),
    ("projection_projection_rmse_scaled", "Projection RMSE, scaled"),
    ("projection_violation", "Projection normalized nonlinear violation"),
    ("projection_violation_original_nonlinear", "Projection original nonlinear violation"),

    ("experiment_rmse_total", "Experiment RMSE"),
    ("experiment_violation", "Experiment normalized nonlinear violation"),
    ("experiment_violation_original_nonlinear", "Experiment original nonlinear violation"),
]

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
for metric, ylabel in metrics:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    if mean_col not in df.columns:
        print(f"Skipping {metric}: column not found")
        continue

    plt.figure(figsize=(7, 5))

    for segmentation_type, group in df.groupby("segmentation_type"):
        group = group.sort_values("n_regions")

        x = group["n_regions"].to_numpy()
        y = group[mean_col].to_numpy()

        if std_col in group.columns:
            yerr = group[std_col].to_numpy()
        else:
            yerr = None

        # log scale cannot show zero or negative values
        y = np.where(y > 0, y, np.nan)

        plt.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=4,
            label=segmentation_type,
        )

    plt.xlabel("Number of regions")
    plt.ylabel(ylabel)
    plt.title(ylabel + " vs number of regions")
    plt.yscale("log")   # IMPORTANT
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = OUTDIR / f"{metric}_logy.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")

print("\nDone. Log-scale plots saved to:")
print(OUTDIR.resolve())