"""Repeated-run driver for the KKT-HardNet 2D PL-style CSTR benchmark.

Each run trains a new KKT-HardNet model from scratch, then reloads/evaluates the
matching final-epoch model on the fixed 34-row PL test set. One row per run is
written to results_2d/kkt_2d_pl_style_benchmark.csv so 50-run mean/95% CI can
be computed later.
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
import math

EVAL_RE = re.compile(r"Evaluation time:\s*([0-9.eE+-]+)\s*s")
RUN_SCRIPT = "run_kkthardnet_cstr_2d_pl_style.py"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--data", default="data_cstr_2d.csv")
    p.add_argument("--output", default="results_2d/kkt_2d_pl_style_benchmark.csv")
    return p.parse_args()


def run_child(cmd, cwd):
    result = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end="")
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {cmd}")
    return result.stdout


def main():
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    base_dir = Path(__file__).resolve().parent
    output_path = base_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for run in range(1, args.runs + 1):
        print(f"\n{'=' * 72}\nKKT-HARDNET 2D RUN {run}/{args.runs}\n{'=' * 72}")

        train_cmd = [sys.executable, RUN_SCRIPT, "--job", "train", "--data", args.data, "--run", str(run)]
        start = time.perf_counter()
        run_child(train_cmd, base_dir)
        training_time = time.perf_counter() - start
        print(f"PL-style external training time: {training_time:.6f} s")

        eval_cmd = [sys.executable, RUN_SCRIPT, "--job", "experiment", "--data", args.data, "--run", str(run)]
        eval_stdout = run_child(eval_cmd, base_dir)
        match = EVAL_RE.search(eval_stdout)
        if not match:
            raise RuntimeError("Could not parse Evaluation time from experiment output")
        evaluation_time = float(match.group(1))

        metrics_path = base_dir / "results_2d" / f"kkt_2d_run_{run}_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        important_metrics = [
            "rmse_pl_style_scaled",
            "original_nonlinear_violation_pl_style",
            "rmse_overall_physical",
            "overall_mean_abs_violation",
            "overall_max_abs_violation",
        ]

        failed_metrics = [
            name for name in important_metrics
            if not math.isfinite(float(metrics[name]))
        ]

        status = "success" if not failed_metrics else "failed_nonfinite"
        row = {
            "run": run,
            "status": status,
            "training_time_sec": training_time,
            "evaluation_time_sec": evaluation_time,
            **{k: v for k, v in metrics.items() if k not in {"run", "evaluation_time_sec"}},
        }
        rows.append(row)

        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nSaved 2D KKT-HardNet benchmark summary: {output_path}")


if __name__ == "__main__":
    main()
