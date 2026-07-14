import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem, t


SEGMENT_SCENARIOS = [1, 2, 3, 5, 11, 30, 55, 90]

BASE_DIR = Path.cwd()
ARCHIVE_ROOT = BASE_DIR / "models_archive"
WORK_DIR = BASE_DIR / "_retro_eval_work"

CODE_FILES = [
    "main.py",
    "train.py",
    "models.py",
    "utils.py",
    "generate_data.py",
]


def parse_run_info(run_dir):
    info_path = run_dir / "RUN_INFO.txt"
    info = {}

    if not info_path.exists():
        return info

    for line in info_path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()

    return info


def get_run_index(run_dir, info):
    if "run_index" in info:
        return int(info["run_index"])

    m = re.search(r"_run(\d+)_", run_dir.name)
    if m:
        return int(m.group(1))

    raise ValueError(f"Could not determine run index for {run_dir}")


def find_latest_archives_for_model(nseg, model_name):
    scenario_dir = ARCHIVE_ROOT / f"nseg_{nseg}"

    if not scenario_dir.exists():
        print(f"Missing archive folder: {scenario_dir}")
        return {}

    by_run = {}

    for run_dir in scenario_dir.iterdir():
        if not run_dir.is_dir():
            continue

        info = parse_run_info(run_dir)

        if info.get("model") != model_name:
            continue

        model_path = run_dir / "model_state.pth"
        if not model_path.exists():
            continue

        try:
            run_idx = get_run_index(run_dir, info)
        except Exception:
            continue

        if run_idx not in by_run:
            by_run[run_idx] = run_dir
        else:
            if run_dir.stat().st_mtime > by_run[run_idx].stat().st_mtime:
                by_run[run_idx] = run_dir

    return dict(sorted(by_run.items()))


def prepare_work_folder(run_dir, model_name):
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    for file_name in CODE_FILES:
        src = BASE_DIR / file_name
        if not src.exists():
            raise FileNotFoundError(f"Missing {src}")
        shutil.copy2(src, WORK_DIR / file_name)

    shutil.copy2(run_dir / "data.csv", WORK_DIR / "data.csv")

    snap = run_dir / "code_snapshot"
    shutil.copy2(snap / "ABb_matrices.csv", WORK_DIR / "ABb_matrices.csv")
    shutil.copy2(snap / "region_edges.npz", WORK_DIR / "region_edges.npz")

    ckpt_dir = WORK_DIR / "models" / "cstr" / model_name / "0.2"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(
        run_dir / "model_state.pth",
        ckpt_dir / "MODELID_0.2_0.pth"
    )


def parse_scores(stdout):
    for line in reversed(stdout.splitlines()):
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            return ast.literal_eval(s)

    raise RuntimeError("Could not find scores dictionary in stdout.")


def evaluate_archived_run(run_dir, model_name):
    prepare_work_folder(run_dir, model_name)

    cmd = [
        sys.executable,
        "main.py",
        "--model", model_name,
        "--model_id", "MODELID",
        "--dataset_type", "cstr",
        "--dataset_path", "data.csv",
        "--job", "experiment",
        "--dtype", "64",
    ]

    result = subprocess.run(
        cmd,
        cwd=WORK_DIR,
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Experiment failed for {run_dir}")

    return parse_scores(result.stdout)


def mean_ci_halfwidth(values, confidence=0.95):
    values = np.asarray(pd.Series(values).dropna().to_numpy(), dtype=float)
    n = len(values)

    if n == 0:
        return np.nan, np.nan

    mean = float(np.mean(values))

    if n < 2:
        return mean, 0.0

    half = float(sem(values) * t.ppf((1 + confidence) / 2.0, n - 1))
    return mean, half


def reevaluate_all_archives():
    model_name = "KKThPINN"
    prefix = "KKThPINN"

    for nseg in SEGMENT_SCENARIOS:
        print("\n" + "=" * 70)
        print(f"Re-evaluating KKT-hPINN only for nseg_{nseg}")
        print("=" * 70)

        rows_by_iter = {}

        archives = find_latest_archives_for_model(nseg, model_name)

        if len(archives) == 0:
            print(f"No {model_name} archives found for nseg_{nseg}")
            continue

        for run_idx, run_dir in archives.items():
            print(f"Evaluating {model_name}, run {run_idx}: {run_dir.name}")

            scores = evaluate_archived_run(run_dir, model_name)

            row = rows_by_iter.setdefault(run_idx, {"Iteration": run_idx})

            row[f"{prefix}_Experiment_RMSE"] = scores.get("rmse_total", np.nan)
            row[f"{prefix}_Experiment_RMSE_MB_Cc"] = scores.get("rmse_total_mb_Cc", np.nan)
            row[f"{prefix}_Experiment_VIOL"] = scores.get("violation", np.nan)
            row[f"{prefix}_Experiment_VIOL_NL"] = scores.get("violation_original_nonlinear", np.nan)

        df = pd.DataFrame(list(rows_by_iter.values())).sort_values("Iteration")

        out_csv = BASE_DIR / f"experiment_epoch_errors_mb_kkt_only_nseg_{nseg}.csv"
        df.to_csv(out_csv, index=False)

        print(f"Saved {out_csv}")


def make_rmse_mb_plot():
    summary_rows = []

    for nseg in SEGMENT_SCENARIOS:
        csv_path = BASE_DIR / f"experiment_epoch_errors_mb_kkt_only_nseg_{nseg}.csv"

        if not csv_path.exists():
            print(f"Skipping missing {csv_path.name}")
            continue

        df = pd.read_csv(csv_path)

        if "KKThPINN_Experiment_RMSE_MB_Cc" not in df.columns:
            print(f"Skipping {csv_path.name}: missing KKThPINN_Experiment_RMSE_MB_Cc")
            continue

        kkt_vals = df["KKThPINN_Experiment_RMSE_MB_Cc"].dropna().to_numpy()

        kkt_mean, kkt_ci = mean_ci_halfwidth(kkt_vals)

        summary_rows.append({
            "nC_regions": nseg,
            "num_regions": nseg,
            "KKThPINN_RMSE_MB_Cc_mean": kkt_mean,
            "KKThPINN_RMSE_MB_Cc_ci95": kkt_ci,
            "n_runs": len(kkt_vals),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv("metric_summary_mb_Cc_kkt_only_by_segments.csv", index=False)
    print("Saved metric_summary_mb_Cc_kkt_only_by_segments.csv")

    if summary.empty:
        print("No valid summary rows. No plot created.")
        return

    x = summary["num_regions"].to_numpy(dtype=float)

    kkt_mean = summary["KKThPINN_RMSE_MB_Cc_mean"].to_numpy(dtype=float)
    kkt_ci = summary["KKThPINN_RMSE_MB_Cc_ci95"].to_numpy(dtype=float)

    plt.figure(figsize=(9, 6))

    plt.plot(
        x,
        kkt_mean,
        marker="o",
        linewidth=2.5,
        label="KKT-hPINN"
    )

    plt.fill_between(
        x,
        kkt_mean - kkt_ci,
        kkt_mean + kkt_ci,
        alpha=0.2,
        label="KKT-hPINN 95% CI"
    )

    plt.xlabel("Number of Cao regions")
    plt.ylabel("RMSE with Cc from total mass balance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig("rmse_mb_Cc_vs_regions_1d_kkt_only.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("rmse_mb_Cc_vs_regions_1d_kkt_only.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved rmse_mb_Cc_vs_regions_1d_kkt_only.pdf")
    print("Saved rmse_mb_Cc_vs_regions_1d_kkt_only.png")


def main():
    reevaluate_all_archives()
    make_rmse_mb_plot()


if __name__ == "__main__":
    main()