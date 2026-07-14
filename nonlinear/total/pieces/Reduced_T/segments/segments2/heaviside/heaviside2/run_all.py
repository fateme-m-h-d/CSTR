# run_segment_comparison.py

import ast
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


# ============================================================
# USER CONFIG
# ============================================================

N_REPEATS = 10

SEGMENT_SCENARIOS = [1, 2, 3, 5, 7, 9, 11, 30, 90]

# Which segmentation types to compare
# SEGMENTATION_TYPES = ["uniform", "adaptive"]
SEGMENTATION_TYPES = ["uniform"]

# For adaptive_linearization.py
ADAPTIVE_SPLIT_METRIC = "mean_abs_f_pl"

# Model/training settings
DATASET_TYPE = "cstr"
DATASET_PATH = "data.csv"
VAL_RATIO = 0.2

HIDDEN_DIM = 32
HIDDEN_NUM = 2
Z0_DIM = 3
INPUT_DIM = 1

EPOCHS = 1000
LR = 1e-4
BATCH_SIZE = 16
DTYPE = 64

RESULT_ROOT = Path("scenario_results")


# ============================================================
# HELPERS
# ============================================================

def run_cmd(cmd, log_path):
    """
    Run a command, save stdout/stderr to log_path, and return stdout text.
    Stops if the command fails.
    """
    print("\n" + "=" * 80)
    print("Running:")
    print(" ".join(cmd))
    print("=" * 80)

    result = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout)

    print(result.stdout)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with return code {result.returncode}.\n"
            f"See log: {log_path}"
        )

    return result.stdout


def extract_last_dict(stdout_text):
    """
    Extract the last printed Python dictionary from stdout.
    This works for outputs like:
        {'rmse_total': ..., 'violation': ...}
    """
    matches = re.findall(r"\{[^{}]*\}", stdout_text)

    if not matches:
        return {}

    for item in reversed(matches):
        try:
            parsed = ast.literal_eval(item)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return {}


def save_dict(d, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path.with_suffix(".json"), "w") as f:
        json.dump(d, f, indent=2)

    pd.DataFrame([d]).to_csv(path.with_suffix(".csv"), index=False)


def safe_copy(src, dst):
    src = Path(src)
    dst = Path(dst)

    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def archive_common_files(outdir):
    """
    Save the linearization files used for this scenario.
    """
    for fname in [
        "region_edges.npz",
        "lin_params.csv",
        "ABb_matrices.csv",
        "adaptive_segmentation_summary.csv",
        "linearization_accuracy_detailed.csv",
        "linearization_accuracy_summary.csv",
        "scaled_data.csv",
        "scaler.pkl",
    ]:
        safe_copy(fname, outdir / "artifacts" / fname)


def archive_model_outputs(outdir, model_id):
    """
    Save model checkpoint, learning curves, and report CSVs.
    """
    model_path = (
        Path("models")
        / DATASET_TYPE
        / "KKThPINN"
        / str(VAL_RATIO)
        / f"{model_id}_{VAL_RATIO}_0.pth"
    )

    safe_copy(model_path, outdir / "artifacts" / "model.pth")

    table_path = (
        Path("data")
        / "tables"
        / DATASET_TYPE
        / "KKThPINN"
        / str(VAL_RATIO)
        / f"{model_id}_{VAL_RATIO}_0.csv"
    )

    safe_copy(table_path, outdir / "artifacts" / "experiment_report.csv")

    curve_dir = (
        Path("data")
        / "learning_curves"
        / DATASET_TYPE
        / "KKThPINN"
        / str(VAL_RATIO)
    )

    for suffix in [
        "train_losses_run0.npy",
        "val_losses_run0.npy",
        "train_violations_run0.npy",
        "val_violations_run0.npy",
    ]:
        safe_copy(
            curve_dir / f"{model_id}_{suffix}",
            outdir / "artifacts" / f"{model_id}_{suffix}",
        )


def make_uniform_linearization(n_regions, outdir):
    """
    Uses your existing linearization.py for uniform segmentation.
    It must save:
        region_edges.npz
        lin_params.csv
        ABb_matrices.csv
    """
    stdout = run_cmd(
        [
            sys.executable,
            "linearization.py",
            "--nT_regions",
            str(n_regions),
        ],
        outdir / "logs" / "linearization_uniform.log",
    )

    return stdout


def make_adaptive_linearization(n_regions, outdir):
    """
    Uses adaptive_linearization.py for adaptive segmentation.
    """
    stdout = run_cmd(
        [
            sys.executable,
            "adaptive_linearization.py",
            "--nT_regions",
            str(n_regions),
            "--split_metric",
            ADAPTIVE_SPLIT_METRIC,
        ],
        outdir / "logs" / "linearization_adaptive.log",
    )

    return stdout


def run_linearization_accuracy(outdir):
    """
    Optional but useful.
    Runs linearization_accuracy.py if it exists.
    """
    if not Path("linearization_accuracy.py").exists():
        print("linearization_accuracy.py not found, skipping.")
        return ""

    stdout = run_cmd(
        [
            sys.executable,
            "linearization_accuracy.py",
        ],
        outdir / "logs" / "linearization_accuracy.log",
    )

    return stdout


def run_projection_check(model_id, outdir):
    stdout = run_cmd(
        [
            sys.executable,
            "main.py",
            "--model",
            "KKThPINN",
            "--model_id",
            model_id,
            "--dataset_type",
            DATASET_TYPE,
            "--dataset_path",
            DATASET_PATH,
            "--job",
            "projection_check",
            "--input_dim",
            str(INPUT_DIM),
            "--z0_dim",
            str(Z0_DIM),
            "--hidden_dim",
            str(HIDDEN_DIM),
            "--hidden_num",
            str(HIDDEN_NUM),
            "--batch_size",
            str(BATCH_SIZE),
            "--dtype",
            str(DTYPE),
        ],
        outdir / "logs" / "projection_check.log",
    )

    scores = extract_last_dict(stdout)
    save_dict(scores, outdir / "projection_check_scores")

    return scores


def run_training(model_id, outdir):
    stdout = run_cmd(
        [
            sys.executable,
            "main.py",
            "--model",
            "KKThPINN",
            "--model_id",
            model_id,
            "--dataset_type",
            DATASET_TYPE,
            "--dataset_path",
            DATASET_PATH,
            "--job",
            "train",
            "--input_dim",
            str(INPUT_DIM),
            "--z0_dim",
            str(Z0_DIM),
            "--hidden_dim",
            str(HIDDEN_DIM),
            "--hidden_num",
            str(HIDDEN_NUM),
            "--batch_size",
            str(BATCH_SIZE),
            "--epochs",
            str(EPOCHS),
            "--lr",
            str(LR),
            "--dtype",
            str(DTYPE),
        ],
        outdir / "logs" / "train.log",
    )

    return stdout


def run_experiment(model_id, outdir):
    stdout = run_cmd(
        [
            sys.executable,
            "main.py",
            "--model",
            "KKThPINN",
            "--model_id",
            model_id,
            "--dataset_type",
            DATASET_TYPE,
            "--dataset_path",
            DATASET_PATH,
            "--job",
            "experiment",
            "--input_dim",
            str(INPUT_DIM),
            "--z0_dim",
            str(Z0_DIM),
            "--hidden_dim",
            str(HIDDEN_DIM),
            "--hidden_num",
            str(HIDDEN_NUM),
            "--batch_size",
            str(BATCH_SIZE),
            "--dtype",
            str(DTYPE),
        ],
        outdir / "logs" / "experiment.log",
    )

    scores = extract_last_dict(stdout)
    save_dict(scores, outdir / "experiment_scores")

    return scores


def make_summary_plots(summary_df, result_root):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available, skipping plots.")
        return

    plot_dir = result_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("projection_mae_scaled", "Projection MAE, scaled"),
        ("projection_rmse_scaled", "Projection RMSE, scaled"),
        ("projection_violation_original_nonlinear", "Projection original nonlinear violation"),
        ("experiment_rmse_total", "Experiment RMSE"),
        ("experiment_violation", "Experiment normalized violation"),
        ("experiment_violation_original_nonlinear", "Experiment original nonlinear violation"),
    ]

    for metric, ylabel in metrics:
        if metric not in summary_df.columns:
            continue

        plt.figure(figsize=(7, 5))

        for segmentation_type, group in summary_df.groupby("segmentation_type"):
            group = group.sort_values("n_regions")
            plt.plot(
                group["n_regions"],
                group[metric],
                marker="o",
                label=segmentation_type,
            )

        plt.xlabel("Number of regions")
        plt.ylabel(ylabel)
        plt.title(ylabel + " vs number of regions")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{metric}.png", dpi=300)
        plt.close()


def make_mean_std_summary(summary_df):
    """
    Create one row per segmentation_type + n_regions.
    Computes mean/std/min/max over repeated NN trainings.
    """

    group_cols = ["segmentation_type", "n_regions"]

    metric_cols = [
        c for c in summary_df.columns
        if c not in ["segmentation_type", "n_regions", "repeat", "model_id"]
    ]

    summary_stats = (
        summary_df
        .groupby(group_cols)[metric_cols]
        .agg(["mean", "std", "min", "max"])
    )

    summary_stats.columns = [
        f"{metric}_{stat}" for metric, stat in summary_stats.columns
    ]

    summary_stats = summary_stats.reset_index()

    return summary_stats


def make_summary_plots_with_errorbars(summary_stats, result_root):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available, skipping plots.")
        return

    plot_dir = result_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("experiment_rmse_total", "Experiment RMSE"),
        ("experiment_violation", "Experiment normalized violation"),
        ("experiment_violation_original_nonlinear", "Experiment original nonlinear violation"),
        ("projection_projection_mae_scaled", "Projection MAE, scaled"),
        ("projection_violation_original_nonlinear", "Projection original nonlinear violation"),
    ]

    for metric, ylabel in metrics:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        if mean_col not in summary_stats.columns:
            continue

        plt.figure(figsize=(7, 5))

        for segmentation_type, group in summary_stats.groupby("segmentation_type"):
            group = group.sort_values("n_regions")

            yerr = group[std_col] if std_col in group.columns else None

            plt.errorbar(
                group["n_regions"],
                group[mean_col],
                yerr=yerr,
                marker="o",
                capsize=4,
                label=segmentation_type,
            )

        plt.xlabel("Number of regions")
        plt.ylabel(ylabel)
        plt.title(ylabel + " vs number of regions")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plt.savefig(plot_dir / f"{metric}_mean_std.png", dpi=300)
        plt.close()
    
# ============================================================
# MAIN WORKFLOW
# ============================================================


def main():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for segmentation_type in SEGMENTATION_TYPES:
        for n_regions in SEGMENT_SCENARIOS:

            scenario_name = f"{segmentation_type}_S{n_regions}"
            scenario_dir = RESULT_ROOT / scenario_name
            scenario_dir.mkdir(parents=True, exist_ok=True)

            print("\n\n" + "#" * 100)
            print(f"PROJECTION CHECK ONLY: {scenario_name}")
            print("#" * 100)

            # ------------------------------------------------------------
            # 1) Build segmentation / linearization
            # ------------------------------------------------------------
            if segmentation_type == "uniform":
                make_uniform_linearization(n_regions, scenario_dir)
            elif segmentation_type == "adaptive":
                make_adaptive_linearization(n_regions, scenario_dir)
            else:
                raise ValueError(f"Unknown segmentation_type: {segmentation_type}")

            # Optional: keep this if you also want linearization accuracy
            run_linearization_accuracy(scenario_dir)

            # Save linearization files
            archive_common_files(scenario_dir)

            # ------------------------------------------------------------
            # 2) Projection check only
            # ------------------------------------------------------------
            projection_model_id = f"{scenario_name}_projection"

            projection_scores = run_projection_check(
                projection_model_id,
                scenario_dir,
            )

            row = {
                "segmentation_type": segmentation_type,
                "n_regions": n_regions,
                "model_id": projection_model_id,
            }

            row.update(projection_scores)
            all_rows.append(row)

            # Save partial results after each scenario
            projection_df = pd.DataFrame(all_rows)
            projection_df.to_csv(
                RESULT_ROOT / "summary_projection_check_only_partial.csv",
                index=False,
            )

    # ------------------------------------------------------------
    # Final projection-only summary
    # ------------------------------------------------------------
    projection_df = pd.DataFrame(all_rows)
    projection_df.to_csv(
        RESULT_ROOT / "summary_projection_check_only.csv",
        index=False,
    )

    make_summary_plots(projection_df, RESULT_ROOT)

    print("\n\nDONE.")
    print(f"Saved projection-only summary to: {RESULT_ROOT / 'summary_projection_check_only.csv'}")
    print(f"Saved plots to: {RESULT_ROOT / 'plots'}")
    

if __name__ == "__main__":
    main()