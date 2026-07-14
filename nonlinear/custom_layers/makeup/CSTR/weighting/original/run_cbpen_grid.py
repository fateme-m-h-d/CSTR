import subprocess

RUNS = 3
DATASET_PATH = "data.csv"

# Use your best weight-only case
CA_WEIGHT = 0.0
CB1_WEIGHT = 0.2
CB2_WEIGHT = 0.0
CC_WEIGHT = 0.8

lambda_values = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]


def label(x):
    return str(x).replace(".", "p").replace("-", "m")


def run_cmd(cmd):
    print("\nRunning:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


for lam in lambda_values:
    model_id = f"cbpen_lam{label(lam)}_ca0p0_cb10p2_cb20p0_cc0p8"

    run_cmd([
        "python", "main.py",
        "--model", "KKThPINN",
        "--model_id", model_id,
        "--dataset_type", "cstr",
        "--dataset_path", DATASET_PATH,
        "--job", "train_experiment",
        "--runs", str(RUNS),
        "--z4_activation", "raw",
        "--ca_weight", str(CA_WEIGHT),
        "--cb1_weight", str(CB1_WEIGHT),
        "--cb2_weight", str(CB2_WEIGHT),
        "--cc_weight", str(CC_WEIGHT),
        "--lambda_cb_consistency", str(lam),
    ])

    run_cmd([
        "python", "summarize_runs.py",
        "--model_id", model_id,
        "--model", "KKThPINN",
        "--dataset_type", "cstr",
        "--val_ratio", "0.2",
        "--runs", str(RUNS),
    ])