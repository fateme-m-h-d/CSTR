import subprocess

RUNS = 20   # change to 20 if you want 20 runs
DATASET_PATH = "data.csv"

candidates = [
    ("top_ca0p0_cb10p2_cb20p0_cc0p8", 0.0, 0.2, 0.0, 0.8),
    ("top_ca0p1_cb10p1_cb20p0_cc0p8", 0.1, 0.1, 0.0, 0.8),
    ("top_ca0p0_cb10p3_cb20p0_cc0p7", 0.0, 0.3, 0.0, 0.7),
    ("top_ca0p0_cb10p1_cb20p0_cc0p9", 0.0, 0.1, 0.0, 0.9),
    ("top_ca0p2_cb10p4_cb20p1_cc0p3", 0.2, 0.4, 0.1, 0.3),
    ("top_ca0p2_cb10p5_cb20p0_cc0p3", 0.2, 0.5, 0.0, 0.3),
]

def run_cmd(cmd):
    print("\nRunning:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

for model_id, w_ca, w_cb1, w_cb2, w_cc in candidates:

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