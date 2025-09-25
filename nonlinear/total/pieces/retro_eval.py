# retro_eval.py
# Re-evaluate archived NN & KKThPINN runs WITHOUT retraining.
# It stages each run's files into new1/ and calls:
#   python main.py --job experiment --model <MODEL> --dataset_type cstr --dataset_path ./data.csv \
#                  --model_id MODELID --val_ratio 0.2 --run 0
# so that train.load_weights() loads: ./models/cstr/<MODEL>/0.2/MODELID_0.2_0.pth

import os
import ast
import csv
import time
import shutil
import subprocess
from pathlib import Path

import torch  # used to ensure checkpoint has a 'state_dict' key

BASE = Path(os.getcwd())
ARCHIVE_ROOT = BASE / "models_archive"                         # e.g., models_archive/30/<timestamp_runXX_*>
TARGET = BASE / "new1"                                         # working dir for main.py
OUT_CSV = BASE / "retro_violation_original_nonlinear.csv"      # results sink

# ---------------- helpers ----------------
def read_model_name(run_dir: Path) -> str:
    """Read model from RUN_INFO.txt; fallback to folder name; default 'NN'."""
    ri = run_dir / "RUN_INFO.txt"
    if ri.exists():
        for line in ri.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip().lower().startswith("model="):
                return line.split("=", 1)[1].strip()
    name = run_dir.name
    if "KKThPINN" in name: return "KKThPINN"
    if "NN" in name: return "NN"
    return "NN"

def find_runs():
    """Yield (num_samples, run_name, model_name, run_dir) for runs that have the required files."""
    runs = []
    if not ARCHIVE_ROOT.exists():
        return runs
    for samples_dir in sorted([p for p in ARCHIVE_ROOT.iterdir() if p.is_dir()]):
        try:
            n = int(samples_dir.name)  # 30, 100, 200, 400, ...
        except ValueError:
            continue
        for run_dir in sorted([p for p in samples_dir.iterdir() if p.is_dir()]):
            if all((run_dir / f).exists() for f in ("model_state.pth", "scaler.pkl", "data.csv")):
                runs.append((n, run_dir.name, read_model_name(run_dir), run_dir))
    return runs

def ensure_state_dict(src_ckpt: Path):
    """Load checkpoint and ensure it has a 'state_dict' key."""
    ckpt = torch.load(src_ckpt, map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        ckpt = {"state_dict": ckpt}
    return ckpt

def stage_into_new1(run_dir: Path, model_name: str):
    """
    Stage this run so main.py can find everything it needs.
    - Copy data.csv and scaler.pkl into new1/
    - Save checkpoint to EXACT path main/train expect:
        new1/models/cstr/<MODEL>/0.2/MODELID_0.2_0.pth
    Returns the full checkpoint path for debugging.
    """
    # 0) ensure clean models dir for clarity
    (TARGET / "models").mkdir(parents=True, exist_ok=True)

    # 1) copy data & scaler the training used
    shutil.copy2(run_dir / "data.csv",   TARGET / "data.csv")
    shutil.copy2(run_dir / "scaler.pkl", TARGET / "scaler.pkl")

    # 2) write checkpoint where load_weights() looks
    nest = TARGET / "models" / "cstr" / model_name / "0.2"
    nest.mkdir(parents=True, exist_ok=True)
    dst_ckpt = nest / "MODELID_0.2_0.pth"   # MUST match args.model_id/val_ratio/run
    # write guaranteed 'state_dict' format
    ckpt = ensure_state_dict(run_dir / "model_state.pth")
    torch.save(ckpt, dst_ckpt)

    return str(dst_ckpt)

def run_experiment(model_name: str):
    """
    Call main.py --job experiment and parse the final printed dict:
    {'rmse_total', 'violation', 'violation_original_nonlinear', ...}
    """
    cmd = [
        "python", "main.py",
        "--job", "experiment",
        "--model", model_name,
        "--model_id", "MODELID",
        "--dataset_type", "cstr",
        "--dataset_path", "./data.csv",
        "--val_ratio", "0.2",
        "--run", "0",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=TARGET)
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()

    scores = {"rmse_total": float("nan"),
              "violation": float("nan"),
              "violation_original_nonlinear": float("nan")}

    # parse the LAST dict-looking line
    for line in reversed(out.splitlines()):
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                d = ast.literal_eval(s)
                for k in scores.keys():
                    if k in d and d[k] is not None:
                        scores[k] = float(d[k])
            except Exception:
                pass
            break

    return scores, out, err

def already_written():
    """Return set of (num_samples, run_name, model) triples already in OUT_CSV."""
    seen = set()
    if OUT_CSV.exists():
        with open(OUT_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    seen.add((int(row["num_samples"]), row["run_name"], row["model"]))
                except Exception:
                    continue
    return seen

def is_nan(x: float) -> bool:
    return x != x

# ---------------- main ----------------
def main():
    runs = find_runs()
    if not runs:
        print(f"No archived runs found under {ARCHIVE_ROOT}")
        return

    seen = already_written()
    rows = []

    for num_samples, run_name, model_name, run_dir in runs:
        key = (num_samples, run_name, model_name)
        if key in seen:
            continue

        print(f"[EVAL] samples={num_samples} | {run_name} | model={model_name}")
        ckpt_path = stage_into_new1(run_dir, model_name)
        scores, stdout_txt, stderr_txt = run_experiment(model_name)

        if any(is_nan(v) for v in scores.values()):
            print("[DEBUG] Expected checkpoint path:", ckpt_path)
            print("[DEBUG] tail(stdout):")
            print("\n".join(stdout_txt.splitlines()[-60:]) or "(empty)")
            if stderr_txt:
                print("[DEBUG] stderr:")
                print(stderr_txt)

        rows.append({
            "num_samples": num_samples,
            "run_name": run_name,
            "model": model_name,
            "rmse_total": scores["rmse_total"],
            "violation": scores["violation"],
            "violation_original_nonlinear": scores["violation_original_nonlinear"],
        })

    if not rows:
        print("[DONE] Nothing new to write.")
        return

    write_header = not OUT_CSV.exists()
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "num_samples", "run_name", "model",
            "rmse_total", "violation", "violation_original_nonlinear"
        ])
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[DONE] Wrote {len(rows)} rows to {OUT_CSV}")

if __name__ == "__main__":
    # make sure working dir exists
    TARGET.mkdir(parents=True, exist_ok=True)
    main()
