"""ENFORCE benchmark for the 2D CSTR, matched to PL-KKT-hPINN/2D.

This is the 2D counterpart of the finalized 1D ENFORCE benchmark. It keeps the
same comparison protocol while replacing the fixed-temperature 1D problem by
the true 2D input problem x = [T, Cao].

Common benchmark settings
-------------------------
* Common 170-row 2D dataset: [Temperature (T), Cao, Ca, Cb, Cc].
* Fixed RandomState(42) 60/20/20 split -> 102 train / 34 val / 34 test.
* MaxAbs scaling fitted on all 170 rows, matching PL-KKT-hPINN.
* Network: [2, 32, 32, 3], ReLU, 1000 epochs, batch size 16, lr=1e-4.
* float32 model/training/projection on CPU; final reported metrics are
  recomputed from finished predictions in NumPy float64.
* Final-epoch checkpoint.
* Fixed test order; 34 test samples are evaluated as 16 + 16 + 2.
* Training and inference projection tolerance = 1e-6, max_it = 100.

What changes from the 1D ENFORCE script
---------------------------------------
1. There are two inputs: T and Cao.
2. kf and kr are no longer constants; they are Arrhenius functions of T.
3. All input slices/scales/model dimensions use 2 input columns.
4. File/result names contain "2d" so they cannot be mixed with 1D outputs.

There is NO piecewise linearization here. ENFORCE receives the original
nonlinear CSTR constraints directly.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch

from enforce.core.model import ENFORCE, ENFORCEConfig
from enforce.engines.train import Trainer, TrainingConfig

# ----------------------------- common benchmark -----------------------------
SPLIT_SEED = 42
BASE_MODEL_SEED = 42
VAL_RATIO = 0.20
TEST_RATIO = 0.20
EXPECTED_N = 170

HIDDEN_NEURONS = 32
PL_HIDDEN_LAYERS = 2
# ENFORCE has one input->hidden layer plus `hidden_layers` hidden->hidden
# layers. Therefore passing 1 gives two actual hidden ReLU layers.
ENFORCE_HIDDEN_LAYERS_ARG = PL_HIDDEN_LAYERS - 1
EPOCHS = 1000
BATCH_SIZE = 16
LEARNING_RATE = 1.0e-4
DTYPE = torch.float64
DEVICE = torch.device("cpu")

# PROJECTION_TOLERANCE = 1.0e-6
TRAINING_TOLERANCE = 1.0e-4
INFERENCE_TOLERANCE = 1.0e-6
MAX_PROJECTION_ITERATIONS = 100
DISPLACEMENT_WEIGHT = 0.5

# ------------------------------- CSTR physics -------------------------------
TAU = 10.0
CBO = 2.0
CCO = 0.0
R = 8.314
AFO = 1.0e13
EAF = 90000.0
ARO = 1.0e11
EAR = 80000.0

REQUIRED_COLUMNS = ["Temperature (T)", "Cao", "Ca", "Cb", "Cc"]


def load_and_prepare_data(data_path: Path) -> Dict[str, np.ndarray]:
    """Load the common 2D CSV, MaxAbs-scale it, and reproduce PL's split."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path.resolve()}")

    df = pd.read_csv(data_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Expected {REQUIRED_COLUMNS}")

    # Keep master data in float64. The actual ENFORCE tensors are converted to
    # float32 below; using float64 here only avoids rounding the stored dataset.
    raw = df[REQUIRED_COLUMNS].dropna().to_numpy(dtype=np.float64, copy=True)
    n = raw.shape[0]
    if n != EXPECTED_N:
        print(
            f"WARNING: PL 2D benchmark uses {EXPECTED_N} rows; this file has {n}. "
            "The same 60/20/20 split logic will still be applied."
        )

    # PL's MaxAbsScaler is fitted before the split, so reproduce that exactly.
    scales = np.max(np.abs(raw), axis=0)
    scales[scales == 0.0] = 1.0
    scaled = raw / scales

    permutation = np.random.RandomState(SPLIT_SEED).permutation(n)
    n_val = int(VAL_RATIO * n)
    n_test = int(TEST_RATIO * n)
    n_train = n - n_val - n_test

    train_idx = permutation[:n_train]
    val_idx = permutation[n_train:n_train + n_val]
    test_idx = permutation[n_train + n_val:]

    return {
        "raw": raw,
        "scaled": scaled,
        "scales": scales,
        "permutation": permutation,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        # 2D inputs = columns 0:2; outputs = columns 2:5.
        "x_train_scaled": scaled[train_idx, 0:2],
        "y_train_scaled": scaled[train_idx, 2:5],
        "x_val_scaled": scaled[val_idx, 0:2],
        "y_val_scaled": scaled[val_idx, 2:5],
        "x_test_scaled": scaled[test_idx, 0:2],
        "y_test_scaled": scaled[test_idx, 2:5],
        "x_test_raw": raw[test_idx, 0:2],
        "y_test_raw": raw[test_idx, 2:5],
    }


def save_split_description(prepared: Dict[str, np.ndarray], output_path: Path) -> None:
    rows = []
    for split_name in ("train", "val", "test"):
        for order, source_index in enumerate(prepared[f"{split_name}_idx"]):
            rows.append(
                {
                    "split": split_name,
                    "order_within_split": order,
                    "source_row_index_zero_based": int(source_index),
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


# -------------------------- original nonlinear physics -----------------------
def cstr_constraints(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Original 2D nonlinear CSTR equalities in physical units.

    x[:,0] = T, x[:,1] = Cao. The Arrhenius coefficients are recomputed for
    every sample. This is the key physics difference relative to the 1D case.
    """
    T = x[:, 0]
    cao = x[:, 1]
    ca = y[:, 0]
    cb = y[:, 1]
    cc = y[:, 2]

    kf = AFO * torch.exp(-EAF / (R * T))
    kr = ARO * torch.exp(-EAR / (R * T))
    g1 = cao - ca - kf * ca * cb.pow(2) * TAU + kr * cc * TAU
    g2 = cao - ca + CBO - cb + CCO - cc
    return torch.stack((g1, g2), dim=1)


def model_seed_for_run(run: int) -> int:
    return BASE_MODEL_SEED + (run - 1)


def build_model(scales: np.ndarray, run: int) -> ENFORCE:
    """Build the [2,32,32,3] ENFORCE model with PL-style MaxAbs scaling."""
    seed = model_seed_for_run(run)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    scaling_input = (
        torch.zeros(2, dtype=DTYPE, device=DEVICE),
        torch.as_tensor(scales[0:2], dtype=DTYPE, device=DEVICE),
    )
    scaling_output = (
        torch.zeros(3, dtype=DTYPE, device=DEVICE),
        torch.as_tensor(scales[2:5], dtype=DTYPE, device=DEVICE),
    )

    config = ENFORCEConfig(
        input_neurons=2,
        output_neurons=3,
        hidden_neurons=HIDDEN_NEURONS,
        hidden_layers=ENFORCE_HIDDEN_LAYERS_ARG,
        # training_tolerance=PROJECTION_TOLERANCE,
        # inference_tolerance=PROJECTION_TOLERANCE,
        training_tolerance=TRAINING_TOLERANCE,
        inference_tolerance=INFERENCE_TOLERANCE,
        max_it=MAX_PROJECTION_ITERATIONS,
        supervised=True,
        weight_loss_displacement=DISPLACEMENT_WEIGHT,
        epoch_start_hard_constrained=0,
        ada_np_auto_activation=True,
        random_seed=seed,
    )

    model = ENFORCE(
        scaling_input=scaling_input,
        scaling_output=scaling_output,
        c=cstr_constraints,
        config=config,
        constrained=True,
        weighting_option=1,
    )

    # ENFORCE 1.0.x keeps scaling quantities as ordinary attributes, so force
    # both the network and these attributes to the benchmark float32 dtype.
    model = model.to(device=DEVICE, dtype=DTYPE)
    model.device = "cpu"
    model.mean_input = model.mean_input.to(device=DEVICE, dtype=DTYPE)
    model.std_input = model.std_input.to(device=DEVICE, dtype=DTYPE)
    model.mean_output = model.mean_output.to(device=DEVICE, dtype=DTYPE)
    model.std_output = model.std_output.to(device=DEVICE, dtype=DTYPE)
    return model


def physical_residuals(
    x_physical: np.ndarray, predictions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Recompute the two original nonlinear residuals in NumPy float64."""
    T = x_physical[:, 0]
    cao = x_physical[:, 1]
    ca, cb, cc = predictions[:, 0], predictions[:, 1], predictions[:, 2]
    kf = AFO * np.exp(-EAF / (R * T))
    kr = ARO * np.exp(-EAR / (R * T))
    g1 = cao - ca - kf * ca * cb**2 * TAU + kr * cc * TAU
    g2 = cao - ca + CBO - cb + CCO - cc
    return g1, g2


def batched_slices(n: int, batch_size: int) -> Iterable[slice]:
    for start in range(0, n, batch_size):
        yield slice(start, min(start + batch_size, n))


def compute_metrics(
    y_pred_physical: np.ndarray,
    y_true_physical: np.ndarray,
    x_test_physical: np.ndarray,
    y_output_scales: np.ndarray,
) -> Dict[str, float]:
    """Compute the same PL-style batch aggregation plus physical diagnostics."""
    y_pred_scaled = y_pred_physical / y_output_scales
    y_true_scaled = y_true_physical / y_output_scales

    g1, g2 = physical_residuals(x_test_physical, y_pred_physical)
    abs_residuals = np.abs(np.column_stack((g1, g2)))

    # With 34 test samples and batch size 16, PL-style aggregation uses three
    # equally weighted batch metrics: 16 + 16 + 2.
    batch_mses: List[float] = []
    batch_violations: List[float] = []
    for sl in batched_slices(len(y_pred_physical), BATCH_SIZE):
        batch_mses.append(float(np.mean((y_pred_scaled[sl] - y_true_scaled[sl]) ** 2)))
        batch_violations.append(float(np.mean(abs_residuals[sl])))

    per_output_rmse = np.sqrt(np.mean((y_pred_physical - y_true_physical) ** 2, axis=0))
    return {
        "rmse_pl_style_scaled": float(np.sqrt(np.mean(batch_mses))),
        "original_nonlinear_violation_pl_style": float(np.mean(batch_violations)),
        "rmse_ca_physical": float(per_output_rmse[0]),
        "rmse_cb_physical": float(per_output_rmse[1]),
        "rmse_cc_physical": float(per_output_rmse[2]),
        "rmse_overall_physical": float(np.sqrt(np.mean((y_pred_physical - y_true_physical) ** 2))),
        "g1_mean_abs": float(np.mean(np.abs(g1))),
        "g2_mean_abs": float(np.mean(np.abs(g2))),
        "overall_mean_abs_violation": float(np.mean(abs_residuals)),
        "g1_max_abs": float(np.max(np.abs(g1))),
        "g2_max_abs": float(np.max(np.abs(g2))),
        "overall_max_abs_violation": float(np.max(abs_residuals)),
    }


def write_report(report_path: Path, metrics: Dict[str, float], projection_iterations: List[int]) -> None:
    with report_path.open("w", encoding="utf-8") as f:
        f.write("ENFORCE 2D CSTR - PL-KKT-hPINN-style benchmark\n")
        f.write("=" * 64 + "\n")
        f.write(f"training_tolerance: {TRAINING_TOLERANCE:.3e}\n")
        f.write(f"inference_tolerance: {INFERENCE_TOLERANCE:.3e}\n")
        f.write(f"max_projection_iterations: {MAX_PROJECTION_ITERATIONS}\n")
        f.write(f"test_batch_size: {BATCH_SIZE}\n")
        f.write(f"projection_iterations_by_batch: {projection_iterations}\n")
        if projection_iterations:
            f.write(f"mean_projection_iterations_across_batches: {np.mean(projection_iterations):.6f}\n")
        f.write("\nMetrics\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.12e}\n")


def train_job(args: argparse.Namespace, prepared: Dict[str, np.ndarray]) -> None:
    models_dir = Path(args.models_dir)
    results_dir = Path(args.results_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(prepared["scales"], args.run)
    x_train = torch.as_tensor(prepared["x_train_scaled"], dtype=DTYPE, device=DEVICE)
    y_train = torch.as_tensor(prepared["y_train_scaled"], dtype=DTYPE, device=DEVICE)

    print("\nENFORCE 2D PL-style benchmark configuration")
    print(f"split: train={len(prepared['train_idx'])}, val={len(prepared['val_idx'])}, test={len(prepared['test_idx'])}")
    print(f"network: [2, {HIDDEN_NEURONS}, {HIDDEN_NEURONS}, 3]")
    print(f"epochs={EPOCHS}, batch_size={BATCH_SIZE}, learning_rate={LEARNING_RATE}")
    print(f"dtype=float64, device=cpu, model_seed={model_seed_for_run(args.run)}")
    print(f"training tolerance={TRAINING_TOLERANCE}, inference tolerance={INFERENCE_TOLERANCE}, max_it={MAX_PROJECTION_ITERATIONS}")
    print("validation rows are held out; ENFORCE native Trainer does not use validation")

    trainer = Trainer(
        model=model,
        config=TrainingConfig(
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            random_seed=model_seed_for_run(args.run),
        ),
    )
    model = trainer.fit(x_train, y_train)

    checkpoint_path = models_dir / f"enforce_2d_run_{args.run}.pt"
    torch.save(model.state_dict(), checkpoint_path)

    metadata = {
        "run": args.run,
        "problem_dimension": "2D inputs: T and Cao",
        "model_seed": model_seed_for_run(args.run),
        "split_seed": SPLIT_SEED,
        "columns": REQUIRED_COLUMNS,
        "maxabs_scales": prepared["scales"].tolist(),
        "configuration": {
            "architecture": [2, HIDDEN_NEURONS, HIDDEN_NEURONS, 3],
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "dtype": "float64",
            "device": "cpu",
            "training_tolerance": TRAINING_TOLERANCE,
            "inference_tolerance": INFERENCE_TOLERANCE,
            "max_projection_iterations": MAX_PROJECTION_ITERATIONS,
            "weight_loss_displacement": DISPLACEMENT_WEIGHT,
            "checkpoint_policy": "final_epoch",
            "validation_used_by_native_trainer": False,
            "test_shuffle": False,
            "train_shuffle": True,
            "linearization_used": False,
        },
    }
    with (models_dir / f"enforce_2d_run_{args.run}_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    save_split_description(prepared, results_dir / "pl_2d_split_indices.csv")
    print(f"Saved checkpoint: {checkpoint_path}")


def experiment_job(args: argparse.Namespace, prepared: Dict[str, np.ndarray]) -> None:
    models_dir = Path(args.models_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = models_dir / f"enforce_2d_run_{args.run}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Run --job train first.")

    # Data loading/scaling/split happens before the comparable PL-style timer.
    x_test_scaled = prepared["x_test_scaled"]
    x_test_raw = prepared["x_test_raw"]
    y_test_raw = prepared["y_test_raw"]
    output_scales = prepared["scales"][2:5]

    eval_start = time.perf_counter()

    # Match PL timing boundary: model construction/loading is inside the timer.
    model = build_model(prepared["scales"], args.run)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    predictions_physical: List[np.ndarray] = []
    projection_iterations: List[int] = []
    
    prediction_time = 0.0
    
    for sl in batched_slices(len(x_test_scaled), BATCH_SIZE):
        xb = torch.as_tensor(x_test_scaled[sl], dtype=DTYPE, device=DEVICE)
        # ENFORCE inference needs autograd to form dc/dy, so do not use no_grad.
        
        # Time ONLY NN forward + ENFORCE projection
        prediction_start = time.perf_counter()
    
        with torch.enable_grad():
            ytilde_scaled, _yhat_scaled, proj_iter = model.predict(xb, training=False)
            
        prediction_time += time.perf_counter() - prediction_start
        _, ytilde_physical = model.unscale(xb, ytilde_scaled)
        predictions_physical.append(ytilde_physical.detach().cpu().numpy().astype(np.float64))
        projection_iterations.append(int(proj_iter))

    y_pred_physical = np.concatenate(predictions_physical, axis=0)
    metrics = compute_metrics(y_pred_physical, y_test_raw, x_test_raw, output_scales)
    g1, g2 = physical_residuals(x_test_raw, y_pred_physical)

    report_path = results_dir / f"enforce_2d_run_{args.run}_report.txt"
    write_report(report_path, metrics, projection_iterations)

    # Small report stays inside timing, matching PL. Detailed predictions CSV
    # below is extra diagnostics and intentionally outside the comparison timer.
    evaluation_time = time.perf_counter() - eval_start

    pd.DataFrame(
        {
            "source_row_index_zero_based": prepared["test_idx"],
            "Temperature (T)": x_test_raw[:, 0],
            "Cao": x_test_raw[:, 1],
            "Ca_true": y_test_raw[:, 0],
            "Cb_true": y_test_raw[:, 1],
            "Cc_true": y_test_raw[:, 2],
            "Ca_pred": y_pred_physical[:, 0],
            "Cb_pred": y_pred_physical[:, 1],
            "Cc_pred": y_pred_physical[:, 2],
            "g1": g1,
            "g2": g2,
            "abs_g1": np.abs(g1),
            "abs_g2": np.abs(g2),
        }
    ).to_csv(results_dir / f"enforce_2d_run_{args.run}_test_predictions.csv", index=False)

    payload = {
        **metrics,
        "run": args.run,
        "projection_iterations_by_batch": projection_iterations,
        "mean_projection_iterations_across_batches": float(np.mean(projection_iterations)),
        "evaluation_time_sec": evaluation_time,
        "prediction_time_sec": prediction_time,
    }
    with (results_dir / f"enforce_2d_run_{args.run}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n========== ENFORCE 2D PL-STYLE TEST RESULTS ==========")
    print(f"Scaled RMSE (PL-style): {metrics['rmse_pl_style_scaled']:.10e}")
    print(f"Original nonlinear violation (PL-style): {metrics['original_nonlinear_violation_pl_style']:.10e}")
    print(f"Physical overall RMSE: {metrics['rmse_overall_physical']:.10e}")
    print(f"overall mean |g|: {metrics['overall_mean_abs_violation']:.10e}")
    print(f"overall max |g|:  {metrics['overall_max_abs_violation']:.10e}")
    print(f"Projection iterations by test batch: {projection_iterations}")
    print(f"Evaluation time: {evaluation_time:.6f} s")
    print(f"Pure prediction time: {prediction_time:.6f} s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--job", choices=["train", "experiment"], required=True)
    p.add_argument("--data", default="data_cstr_2d.csv")
    p.add_argument("--run", type=int, default=1)
    p.add_argument("--models-dir", default="models_2d")
    p.add_argument("--results-dir", default="results_2d")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.run < 1:
        raise ValueError("--run must be >= 1")
    torch.set_default_dtype(torch.float64)
    prepared = load_and_prepare_data(Path(args.data))
    if args.job == "train":
        train_job(args, prepared)
    else:
        experiment_job(args, prepared)


if __name__ == "__main__":
    main()
