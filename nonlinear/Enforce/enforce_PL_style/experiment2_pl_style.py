"""External PL-KKT-hPINN-style timing wrapper for ENFORCE.

Training time is measured around the complete training subprocess, just like
PL-KKT-hPINN's scripts/experiment2.py. Evaluation time is parsed from the
internal timer printed by run_enforce_cstr.py --job experiment.
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

EVAL_RE = re.compile(r"Evaluation time:\s*([0-9.eE+-]+)\s*s")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--data", default="data.csv")
    p.add_argument("--output", default="results/enforce_pl_style_benchmark.csv")
    return p.parse_args()


def run_child(cmd, cwd):
    result = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
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
        print(f"\n{'=' * 70}\nENFORCE RUN {run}/{args.runs}\n{'=' * 70}")

        train_cmd = [
            sys.executable,
            "run_enforce_cstr.py",
            "--job",
            "train",
            "--data",
            args.data,
            "--run",
            str(run),
        ]

        # Same training-time boundary as PL-KKT-hPINN experiment2.py:
        # wall clock around the entire child process.
        start = time.perf_counter()
        run_child(train_cmd, base_dir)
        training_time = time.perf_counter() - start
        print(f"PL-style external training time: {training_time:.6f} s")

        eval_cmd = [
            sys.executable,
            "run_enforce_cstr.py",
            "--job",
            "experiment",
            "--data",
            args.data,
            "--run",
            str(run),
        ]
        eval_stdout = run_child(eval_cmd, base_dir)
        match = EVAL_RE.search(eval_stdout)
        if not match:
            raise RuntimeError("Could not parse Evaluation time from experiment output")
        evaluation_time = float(match.group(1))

        metrics_path = base_dir / "results" / f"enforce_run_{run}_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        row = {
            "run": run,
            "training_time_sec": training_time,
            "evaluation_time_sec": evaluation_time,
            **{
                k: v
                for k, v in metrics.items()
                if k not in {"run", "evaluation_time_sec", "projection_iterations_by_batch"}
            },
            "projection_iterations_by_batch": json.dumps(
                metrics["projection_iterations_by_batch"]
            ),
        }
        rows.append(row)

        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nSaved benchmark summary: {output_path}")


if __name__ == "__main__":
    main()
