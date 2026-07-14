"""Re-evaluate archived NN/KKThPINN runs and report each constraint separately.

Run from the project root, for example:
    python retro_eval_constraints.py --scenario nseg_90 --model KKThPINN
    python retro_eval_constraints.py --scenario nseg_90 --model KKThPINN --set 1
    python retro_eval_constraints.py --scenario nseg_90 --model KKThPINN --set 2

This does not retrain any model. It uses each archived model_state.pth and the
matching code/data snapshot stored by experiment2.py.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


EVALUATOR_CODE = r'''
import argparse
import json
from types import SimpleNamespace

import torch

from utils import LoadData, LoadModel, compute_violation_original_nonlinear


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["NN", "KKThPINN"], required=True)
    args_in = parser.parse_args()

    torch.set_default_dtype(torch.float64)

    args = SimpleNamespace(
        model=args_in.model,
        input_dim=1,
        hidden_dim=32,
        hidden_num=2,
        z0_dim=3,
        dtype=64,
        dataset_type="cstr",
        dataset_path="data.csv",
        val_ratio=0.2,
        batch_size=16,
        job="experiment",
    )

    data = LoadData(args)
    model = LoadModel(args, data)

    checkpoint = torch.load("model_state.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    violations = []
    with torch.no_grad():
        for X, _ in data["test_loader"]:
            pred = model(X)
            v = compute_violation_original_nonlinear(
                X_scaled=X,
                Ypred_scaled=pred,
                scaler=data["scaler"],
                device="cpu",
            )
            violations.append(v.cpu())

    v = torch.cat(violations, dim=0)
    result = {
        "n_test": int(v.shape[0]),
        "reaction_mean": float(v[:, 0].mean().item()),
        "mass_balance_mean": float(v[:, 1].mean().item()),
        "reaction_max": float(v[:, 0].max().item()),
        "mass_balance_max": float(v[:, 1].max().item()),
        "combined_mean": float(v.mean().item()),
    }
    print("JSON_RESULT=" + json.dumps(result))


if __name__ == "__main__":
    main()
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="nseg_90", help="Archive scenario folder, e.g. nseg_90")
    parser.add_argument(
        "--model",
        default="both",
        choices=["NN", "KKThPINN", "both"],
        help="Which archived model runs to evaluate",
    )
    parser.add_argument(
        "--set",
        dest="archive_set",
        type=int,
        default=None,
        help=(
            "Archived experiment set to evaluate. "
            "Use 1 for the earliest archive of each run number, "
            "2 for the second archive, etc. Omit to evaluate all archives."
        ),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root containing models_archive")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use")
    args = parser.parse_args()

    if args.archive_set is not None and args.archive_set < 1:
        parser.error("--set must be 1 or greater")

    return args


def infer_model(run_dir: Path) -> str | None:
    if run_dir.name.endswith("_KKThPINN"):
        return "KKThPINN"
    if run_dir.name.endswith("_NN"):
        return "NN"
    return None


def infer_run_index(name: str) -> int | None:
    match = re.search(r"_run(\d+)_", name)
    return int(match.group(1)) if match else None


def copy_required_files(run_dir: Path, work_dir: Path) -> None:
    code_dir = run_dir / "code_snapshot"
    required = [
        "models.py",
        "utils.py",
        "generate_data.py",
        "ABb_matrices.csv",
        "region_edges.npz",
    ]

    missing = []
    for filename in required:
        src = code_dir / filename
        if not src.exists():
            missing.append(str(src))
        else:
            shutil.copy2(src, work_dir / filename)

    data_src = run_dir / "data.csv"
    model_src = run_dir / "model_state.pth"
    if not data_src.exists():
        missing.append(str(data_src))
    else:
        shutil.copy2(data_src, work_dir / "data.csv")
    if not model_src.exists():
        missing.append(str(model_src))
    else:
        shutil.copy2(model_src, work_dir / "model_state.pth")

    if missing:
        raise FileNotFoundError("Missing archived files:\n  " + "\n  ".join(missing))

    (work_dir / "_evaluate_one.py").write_text(EVALUATOR_CODE, encoding="utf-8")


def evaluate_archive(run_dir: Path, model_name: str, python_exe: str) -> dict[str, float]:
    with tempfile.TemporaryDirectory(prefix="constraint_eval_") as temp_name:
        work_dir = Path(temp_name)
        copy_required_files(run_dir, work_dir)

        result = subprocess.run(
            [python_exe, "_evaluate_one.py", "--model", model_name],
            cwd=work_dir,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Evaluation failed for {run_dir.name}\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

        for line in reversed(result.stdout.splitlines()):
            if line.startswith("JSON_RESULT="):
                return json.loads(line.split("=", 1)[1])

        raise RuntimeError(f"No JSON_RESULT found for {run_dir.name}\n{result.stdout}")


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["reaction_mean", "mass_balance_mean", "reaction_max", "mass_balance_max", "combined_mean"]
    for model_name, group in df.groupby("model", sort=False):
        row = {"model": model_name, "num_runs": len(group)}
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_across_runs_mean"] = float(np.mean(values))
            row[f"{metric}_across_runs_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    scenario_dir = root / "models_archive" / args.scenario
    if not scenario_dir.exists():
        raise FileNotFoundError(f"Scenario archive not found: {scenario_dir}")

    selected_models = {"NN", "KKThPINN"} if args.model == "both" else {args.model}
    run_dirs = []
    for path in scenario_dir.iterdir():
        if not path.is_dir():
            continue
        model_name = infer_model(path)
        if model_name in selected_models:
            run_dirs.append((path, model_name))

    if not run_dirs:
        raise RuntimeError(f"No matching archived runs found in {scenario_dir}")

    # Re-running the same 50-run experiment creates another archive folder
    # with the same model and run number but a later timestamp. Group those
    # duplicate run numbers and select the requested chronological set.
    if args.archive_set is not None:
        grouped_run_dirs = {}

        for path, model_name in run_dirs:
            run_index = infer_run_index(path.name)
            key = (model_name, run_index)
            grouped_run_dirs.setdefault(key, []).append((path, model_name))

        selected_run_dirs = []
        unavailable = []
        set_position = args.archive_set - 1

        for (model_name, run_index), candidates in grouped_run_dirs.items():
            # Folder names begin with YYYYMMDD-HHMMSS, so alphabetical order
            # is chronological from the earliest archive to the latest.
            candidates.sort(key=lambda item: item[0].name)

            if set_position < len(candidates):
                selected_run_dirs.append(candidates[set_position])
            else:
                unavailable.append(
                    f"{model_name} run {run_index}: "
                    f"only {len(candidates)} archived set(s)"
                )

        run_dirs = selected_run_dirs

        if unavailable:
            print(
                f"Warning: archived set {args.archive_set} was unavailable "
                f"for {len(unavailable)} run(s):"
            )
            for message in unavailable:
                print(f"  {message}")

        if not run_dirs:
            raise RuntimeError(
                f"Archived set {args.archive_set} was not available for any "
                f"matching runs in {scenario_dir}"
            )

        print(
            f"Selected archived set {args.archive_set}: "
            f"{len(run_dirs)} run(s)."
        )

    run_dirs.sort(
        key=lambda item: (
            item[1],
            infer_run_index(item[0].name) or 0,
            item[0].name,
        )
    )

    rows = []
    failures = []
    for index, (run_dir, model_name) in enumerate(run_dirs, start=1):
        print(f"[{index}/{len(run_dirs)}] Evaluating {run_dir.name}")
        try:
            metrics = evaluate_archive(run_dir, model_name, args.python)
            rows.append(
                {
                    "scenario": args.scenario,
                    "archive_set": args.archive_set if args.archive_set is not None else "all",
                    "model": model_name,
                    "run_index": infer_run_index(run_dir.name),
                    "archive_folder": run_dir.name,
                    **metrics,
                }
            )
        except Exception as exc:
            failures.append({"archive_folder": run_dir.name, "error": str(exc)})
            print(f"  FAILED: {exc}")

    if not rows:
        raise RuntimeError("All archived evaluations failed.")

    results = pd.DataFrame(rows).sort_values(["model", "run_index"], na_position="last")
    summary = make_summary(results)

    summary.insert(
        1,
        "archive_set",
        args.archive_set if args.archive_set is not None else "all",
    )

    suffix = args.model.lower()
    if args.archive_set is not None:
        suffix += f"_set{args.archive_set}"

    details_path = root / f"constraint_violations_{args.scenario}_{suffix}.csv"
    summary_path = root / f"constraint_violations_{args.scenario}_{suffix}_summary.csv"
    results.to_csv(details_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nPer-run results:")
    print(results[["model", "run_index", "reaction_mean", "mass_balance_mean", "reaction_max", "mass_balance_max"]].to_string(index=False))
    print("\nAcross-run summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved: {details_path}")
    print(f"Saved: {summary_path}")

    if failures:
        failures_path = root / f"constraint_violations_{args.scenario}_{suffix}_failures.csv"
        pd.DataFrame(failures).to_csv(failures_path, index=False)
        print(f"Some runs failed; saved: {failures_path}")


if __name__ == "__main__":
    main()