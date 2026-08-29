"""Repeated-run driver for the ENFORCE 2D PL-style CSTR benchmark.

For each repetition this script:
1. launches a completely new training subprocess;
2. measures full training wall time outside that subprocess, like PL experiment2;
3. launches the matching test/evaluation subprocess;
4. reads that run's JSON metrics;
5. appends one row to results_2d/enforce_2d_pl_style_benchmark.csv.

Use --runs 1 for a smoke test and --runs 50 for the paper experiment.
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
RUN_SCRIPT = "run_enforce_cstr_2d_pl_style.py"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--data", default="data_cstr_2d.csv")
    p.add_argument("--output", default="results_2d/enforce_2d_pl_style_benchmark.csv")
    return p.parse_args()


# def run_child(cmd, cwd):
#     result = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#     print(result.stdout, end="")
#     if result.returncode != 0:
#         raise RuntimeError(f"Command failed with code {result.returncode}: {cmd}")
#     return result.stdout
def run_child(cmd, cwd):
    """Run child process and show its output live while also saving it."""

    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    output_lines = []

    for line in process.stdout:
        print(line, end="", flush=True)
        output_lines.append(line)

    returncode = process.wait()

    if returncode != 0:
        raise RuntimeError(
            f"Command failed with code {returncode}: {cmd}"
        )

    return "".join(output_lines)


def main():
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    base_dir = Path(__file__).resolve().parent
    output_path = base_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for run in range(1, args.runs + 1):
        print(f"\n{'=' * 72}\nENFORCE 2D RUN {run}/{args.runs}\n{'=' * 72}")

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

        metrics_path = base_dir / "results_2d" / f"enforce_2d_run_{run}_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {
            "run": run,
            "training_time_sec": training_time,
            "evaluation_time_sec": evaluation_time,
            **{k: v for k, v in metrics.items() if k not in {"run", "evaluation_time_sec", "projection_iterations_by_batch"}},
            "projection_iterations_by_batch": json.dumps(metrics["projection_iterations_by_batch"]),
        }
        rows.append(row)

        # Rewrite after every run so a partial 50-run job still leaves all
        # completed repetitions safely stored on disk.
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nSaved 2D ENFORCE benchmark summary: {output_path}")


if __name__ == "__main__":
    main()
