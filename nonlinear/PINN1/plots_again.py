import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import zipfile

# ---------------- files ----------------
raw_csv = Path("nn_pinn_50_compare_raw.csv")
summary_csv = Path("nn_pinn_50_compare_summary.csv")

raw = pd.read_csv(raw_csv)
summary = pd.read_csv(summary_csv)

# Preferred method order
method_order = ["NN", "PINN", "KKThPINN"]
summary["model"] = pd.Categorical(summary["model"], categories=method_order, ordered=True)
summary = summary.sort_values("model").reset_index(drop=True)

# ---------------- plotting helper ----------------
def plot_bar_from_summary(
    df,
    mean_col,
    ci_col,
    ylabel,
    save_stem,
    logy=False,
):
    fig, ax = plt.subplots(figsize=(5.2, 4.0))

    x = np.arange(len(df))
    means = df[mean_col].to_numpy()
    errors = df[ci_col].to_numpy()

    ax.bar(x, means, yerr=errors, capsize=5, edgecolor="black", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(df["model"].astype(str).tolist())
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Method")

    if logy:
        ax.set_yscale("log")

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)

    # No title, as requested
    fig.tight_layout()

    pdf_path = Path(f"{save_stem}.pdf")
    png_path = Path(f"{save_stem}.png")

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return pdf_path, png_path

# ---------------- plots ----------------
files = []

files.extend(plot_bar_from_summary(
    summary,
    mean_col="inference_violation_mean",
    ci_col="inference_violation_ci95",
    ylabel="Mean absolute violation",
    save_stem="mean_absolute_violation_vs_methods",
    logy=True,
))

files.extend(plot_bar_from_summary(
    summary,
    mean_col="inference_rmse_mean",
    ci_col="inference_rmse_ci95",
    ylabel="Inference RMSE",
    save_stem="inference_rmse_vs_methods",
    logy=False,
))

files.extend(plot_bar_from_summary(
    summary,
    mean_col="final_train_loss_mean",
    ci_col="final_train_loss_ci95",
    ylabel="Training loss",
    save_stem="training_loss_vs_methods",
    logy=False,
))

# ---------------- save reusable script ----------------
script_text = r'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

raw_csv = Path("nn_pinn_50_compare_raw.csv")
summary_csv = Path("nn_pinn_50_compare_summary.csv")

raw = pd.read_csv(raw_csv)
summary = pd.read_csv(summary_csv)

method_order = ["NN", "PINN", "KKThPINN"]
summary["model"] = pd.Categorical(summary["model"], categories=method_order, ordered=True)
summary = summary.sort_values("model").reset_index(drop=True)

def plot_bar_from_summary(df, mean_col, ci_col, ylabel, save_stem, logy=False):
    fig, ax = plt.subplots(figsize=(5.2, 4.0))

    x = np.arange(len(df))
    means = df[mean_col].to_numpy()
    errors = df[ci_col].to_numpy()

    ax.bar(x, means, yerr=errors, capsize=5, edgecolor="black", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(df["model"].astype(str).tolist())
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Method")

    if logy:
        ax.set_yscale("log")

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)

    # No title
    fig.tight_layout()
    fig.savefig(f"{save_stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{save_stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

plot_bar_from_summary(
    summary,
    mean_col="inference_violation_mean",
    ci_col="inference_violation_ci95",
    ylabel="Mean absolute violation",
    save_stem="mean_absolute_violation_vs_methods",
    logy=True,
)

plot_bar_from_summary(
    summary,
    mean_col="inference_rmse_mean",
    ci_col="inference_rmse_ci95",
    ylabel="Inference RMSE",
    save_stem="inference_rmse_vs_methods",
    logy=False,
)

plot_bar_from_summary(
    summary,
    mean_col="final_train_loss_mean",
    ci_col="final_train_loss_ci95",
    ylabel="Training loss",
    save_stem="training_loss_vs_methods",
    logy=False,
)
'''
script_path = Path("plot_nn_pinn_results.py")
script_path.write_text(script_text)

files.append(script_path)

# ---------------- zip all outputs ----------------
zip_path = Path("nn_pinn_plots_no_titles.zip")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for file in files:
        zf.write(file, arcname=file.name)

print("Created files:")
for file in files:
    print(file.name)
print(zip_path.name)
