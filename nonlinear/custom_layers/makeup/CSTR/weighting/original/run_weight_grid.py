import subprocess
from pathlib import Path

RUNS = 3
STEP = 0.1
DATASET_PATH = "data.csv"

# Optional: based on your previous results, skip large Cb2 cases
MAX_CB2 = 1

def label(x):
    return str(x).replace(".", "p")

def run_cmd(cmd):
    print("\nRunning:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

weights = []
n = int(round(1.0 / STEP))

for i in range(n + 1):
    w_ca = round(i * STEP, 10)
    for j in range(n + 1 - i):
        w_cb1 = round(j * STEP, 10)
        for k in range(n + 1 - i - j):
            w_cb2 = round(k * STEP, 10)
            w_cc = round(1.0 - w_ca - w_cb1 - w_cb2, 10)

            if w_cb1 + w_cb2 <= 0:
                continue

            # optional pruning, because previous results showed Cb2-heavy cases are bad
            if w_cb2 > MAX_CB2:
                continue

            weights.append((w_ca, w_cb1, w_cb2, w_cc))

print(f"Total cases: {len(weights)}")

for w_ca, w_cb1, w_cb2, w_cc in weights:
    model_id = (
        f"grid_"
        f"ca{label(w_ca)}_"
        f"cb1{label(w_cb1)}_"
        f"cb2{label(w_cb2)}_"
        f"cc{label(w_cc)}"
    )

    run_cmd([
        "python", "main.py",
        "--model", "KKThPINN",
        "--model_id", model_id,
        "--dataset_type", "cstr",
        "--dataset_path", DATASET_PATH,
        "--job", "train_experiment",
        "--runs", str(RUNS),
        "--z4_activation", "raw",
        "--ca_weight", str(w_ca),
        "--cb1_weight", str(w_cb1),
        "--cb2_weight", str(w_cb2),
        "--cc_weight", str(w_cc),
    ])

    run_cmd([
        "python", "summarize_runs.py",
        "--model_id", model_id,
        "--model", "KKThPINN",
        "--dataset_type", "cstr",
        "--val_ratio", "0.2",
        "--runs", str(RUNS),
    ])