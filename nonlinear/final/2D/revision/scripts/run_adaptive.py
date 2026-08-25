"""Run one offline-adaptive partition scenario and then the existing experiments."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path.cwd()


def run(command, env=None):
    print(" ".join(map(str, command)))
    subprocess.run(
        list(map(str, command)), cwd=BASE_DIR, env=env, check=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument(
        "--criterion",
        choices=["projection", "remainder", "hybrid"],
        default="projection",
    )
    parser.add_argument("--error_quantile", type=float, default=1.0)
    parser.add_argument("--constraint_scale", type=float, default=1.0)
    parser.add_argument("--max_regions", type=int, default=64)
    parser.add_argument("--max_depth", type=int, default=12)
    parser.add_argument("--min_samples_leaf", type=int, default=8)
    parser.add_argument("--min_relative_improvement", type=float, default=0.01)
    parser.add_argument(
        "--split_mode", choices=["quantile", "midpoint"], default="quantile"
    )
    parser.add_argument("--split_quantiles", type=str, default="0.25,0.5,0.75")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument(
        "--generate_data",
        action="store_true",
        help="Generate the repository's synthetic data.csv before preprocessing.",
    )
    parser.add_argument("--n_total_points", type=int, default=170)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.generate_data:
        run(
            [
                sys.executable,
                "-m",
                "src.generate_data",
                "--n_total_points",
                args.n_total_points,
                "--seed",
                args.seed,
                "--out_csv",
                "data.csv",
            ]
        )

    if not (BASE_DIR / "data.csv").exists():
        raise FileNotFoundError(
            "data.csv was not found. Supply your real data as data.csv or use --generate_data."
        )

    run(
        [
            sys.executable,
            "-m",
            "src.linearization",
            "--data_csv",
            "data.csv",
            "--epsilon",
            args.epsilon,
            "--criterion",
            args.criterion,
            "--error_quantile",
            args.error_quantile,
            "--constraint_scale",
            args.constraint_scale,
            "--max_regions",
            args.max_regions,
            "--max_depth",
            args.max_depth,
            "--min_samples_leaf",
            args.min_samples_leaf,
            "--min_relative_improvement",
            args.min_relative_improvement,
            "--split_mode",
            args.split_mode,
            "--split_quantiles",
            args.split_quantiles,
        ]
    )

    with open(BASE_DIR / "partition_summary.json", encoding="utf-8") as handle:
        summary = json.load(handle)
    print("Adaptive partition summary:")
    print(json.dumps(summary, indent=2))

    env = os.environ.copy()
    env["SCENARIO_ID"] = (
        f"adaptive_{args.criterion}_eps_{args.epsilon:g}_"
        f"regions_{summary['num_regions']}"
    )
    env["NUM_ITERATIONS"] = str(args.repeats)
    env["EPOCHS"] = str(args.epochs)
    env["PYTHON_EXE"] = sys.executable
    run([sys.executable, "-m", "scripts.experiment2"], env=env)


if __name__ == "__main__":
    main()
