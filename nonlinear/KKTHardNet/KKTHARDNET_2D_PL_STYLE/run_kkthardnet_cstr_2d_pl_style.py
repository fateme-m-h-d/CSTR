"""KKT-HardNet benchmark for the 2D CSTR, matched to PL-KKT-hPINN/2D.

This is the 2D counterpart of the finalized 1D KKT-HardNet benchmark.
The model receives x = [T, Cao] and predicts [Ca, Cb, Cc] while the KKT
projection enforces the original nonlinear CSTR equations directly.

Common benchmark settings
-------------------------
* Common 170-row 2D dataset -> 102 train / 34 validation / 34 test using the
  exact PL RandomState(42) split.
* Full-data MaxAbs scaling.
* Network [2,32,32,3], 1000 epochs, batch 16, Adam lr=1e-4.
* float32 JAX/model/projection on CPU; finished predictions are converted to
  NumPy float64 only for common external metric calculation.
* gn_tol=1e-6 and gn_max_iters=100.
* KKT-HardNet-specific epoch_mlp=100 and cons_alpha=0.5 are retained.
* Final epoch model, fixed validation/test order, test batches 16+16+2.

Important 2D change
-------------------
The Arrhenius coefficients depend on the input temperature T. The symbolic
KKT constraints therefore contain exp(-E/(R*T)); there is no PL linearization.

The KKT package performs its own train/validation split internally. As in the
1D wrapper, this script rearranges the 136 non-test rows so the package sees
exactly PL's 102 training rows and 34 validation rows.
"""

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"    #changed from false to true

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import jax.numpy as jnp
from kkthn.builder import Expression

from kkthn import KKTHardNet
import jax

# Some KKT-HardNet modules enable x64 during import. Force it off afterwards
# so this benchmark is genuinely float32, consistent with ENFORCE.
jax.config.update("jax_enable_x64", True)     # change false to true

SPLIT_SEED = 42
BASE_MODEL_SEED = 42
VAL_RATIO = 0.20
TEST_RATIO = 0.20
EXPECTED_N = 170

HIDDEN_NEURONS = 32
HIDDEN_LAYERS = 2
EPOCHS = 1000
BATCH_SIZE = 16
LEARNING_RATE = 1.0e-4

PROJECTION_TOLERANCE = 1.0e-6
MAX_PROJECTION_ITERATIONS = 100
EPOCH_MLP = 100
CONS_ALPHA = 0.5

TAU = 10.0
CBO = 2.0
CCO = 0.0
R = 8.314
AFO = 1.0e13
EAF = 90000.0
ARO = 1.0e11
EAR = 80000.0

REQUIRED_COLUMNS = ["Temperature (T)", "Cao", "Ca", "Cb", "Cc"]

def kkt_exp(expr):
    """Exponential that works with KKT-HardNet's custom Expression class."""
    return Expression(
        lambda ctx, e=expr: jnp.exp(e.eval(ctx)),
        f"exp({expr.text})",
    )
    
    
def load_and_prepare_data(data_path: Path) -> Dict[str, np.ndarray]:
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path.resolve()}")
    df = pd.read_csv(data_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Expected {REQUIRED_COLUMNS}")

    raw = df[REQUIRED_COLUMNS].dropna().to_numpy(dtype=np.float64, copy=True)
    n = raw.shape[0]
    if n != EXPECTED_N:
        print(f"WARNING: PL 2D benchmark uses {EXPECTED_N} rows; this file has {n}.")

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
        "x_test_scaled": scaled[test_idx, 0:2],
        "y_test_scaled": scaled[test_idx, 2:5],
        "x_test_raw": raw[test_idx, 0:2],
        "y_test_raw": raw[test_idx, 2:5],
    }


def save_split_description(prepared: Dict[str, np.ndarray], output_path: Path) -> None:
    rows = []
    for split_name in ("train", "val", "test"):
        for order, source_index in enumerate(prepared[f"{split_name}_idx"]):
            rows.append({"split": split_name, "order_within_split": order, "source_row_index_zero_based": int(source_index)})
    pd.DataFrame(rows).to_csv(output_path, index=False)


def model_seed_for_run(run: int) -> int:
    return BASE_MODEL_SEED + (run - 1)


def write_kkt_training_csvs(prepared: Dict[str, np.ndarray], run: int, root: Path) -> Tuple[Path, Path, Path]:
    """Arrange the 136 non-test rows so KKT-HardNet reproduces PL's split.

    For 170 total rows the PL split is 102 train, 34 val, 34 test. KKT-HardNet
    only sees the first 136 (train+val) rows prepared here; the 34 test rows are
    never included in its dataset files.
    """
    train_rows = prepared["scaled"][prepared["train_idx"]]
    val_rows = prepared["scaled"][prepared["val_idx"]]
    n_total = len(train_rows) + len(val_rows)
    expected_train = int(0.75 * n_total)
    if len(train_rows) != expected_train:
        raise ValueError(f"Expected KKT train_frac=.75 to select {expected_train} rows, but PL train has {len(train_rows)}")

    package_idx = np.arange(n_total)
    rng = np.random.default_rng(model_seed_for_run(run))
    rng.shuffle(package_idx)
    package_train_positions = package_idx[:len(train_rows)]
    package_val_positions = package_idx[len(train_rows):]

    arranged = np.empty((n_total, 5), dtype=np.float64)
    source_index = np.empty(n_total, dtype=np.int64)
    intended_split = np.empty(n_total, dtype=object)

    arranged[package_train_positions] = train_rows
    source_index[package_train_positions] = prepared["train_idx"]
    intended_split[package_train_positions] = "train"

    # KKT-HardNet deterministically permutes validation minibatches using the
    # run seed. Pre-invert that permutation so the effective validation order
    # is PL's fixed unshuffled order (16 + 16 + 2 for 34 samples).
    val_batch_perm = np.arange(len(val_rows))
    val_rng = np.random.default_rng(model_seed_for_run(run))
    val_rng.shuffle(val_batch_perm)
    val_rows_preordered = np.empty_like(val_rows)
    val_source_preordered = np.empty(len(val_rows), dtype=np.int64)
    val_rows_preordered[val_batch_perm] = val_rows
    val_source_preordered[val_batch_perm] = prepared["val_idx"]

    arranged[package_val_positions] = val_rows_preordered
    source_index[package_val_positions] = val_source_preordered
    intended_split[package_val_positions] = "val"

    input_dir = root / f"run_{run}"
    input_dir.mkdir(parents=True, exist_ok=True)
    parameters_path = input_dir / "parameters_2d.csv"
    variables_path = input_dir / "variables_2d.csv"
    mapping_path = input_dir / "kkt_2d_internal_row_mapping.csv"

    # KKT-HardNet CSV headers must match model.add_parameter/add_variable names.
    pd.DataFrame(arranged[:, 0:2], columns=["Temperature", "Cao"]).to_csv(parameters_path, index=False)
    pd.DataFrame(arranged[:, 2:5], columns=["Ca", "Cb", "Cc"]).to_csv(variables_path, index=False)
    pd.DataFrame({
        "kkt_csv_row": np.arange(n_total),
        "original_source_row_index_zero_based": source_index,
        "intended_pl_split": intended_split,
    }).to_csv(mapping_path, index=False)
    return parameters_path, variables_path, mapping_path


def build_symbolic_model(scales: np.ndarray, run: int, parameters_path: Path | None = None, variables_path: Path | None = None) -> KKTHardNet:
    train_config = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "train_frac": 0.75,  # 136 rows -> 102 train + 34 val
        "hidden_size": HIDDEN_NEURONS,
        "hidden_layers": HIDDEN_LAYERS,
        "seed": model_seed_for_run(run),
        "dtype": "float64",    # changed from float32 to float64
        "print_every": 100,
        "drop_last": False,
        "epoch_mlp": EPOCH_MLP,
        "cons_alpha": CONS_ALPHA,
    }
    projection_config = {
        "fb_eps": 1.0e-8,
        "gn_max_iters": MAX_PROJECTION_ITERATIONS,
        "gn_tol": PROJECTION_TOLERANCE,
        "gn_reg": 1.0e-3,
        "newton_step_length": 0.5,
        "armijo_alpha": 1.0e-4,
        "armijo_beta": 0.5,
        "max_backtrack_iter": 10,
        "backward_reg": 1.0e-8,
    }

    model = KKTHardNet(name=f"CSTR_2D_KKTHardNet_PLStyle_run{run}", train=train_config, projection=projection_config)
    x = model.add_parameter(["Temperature", "Cao"])
    y = model.add_variable(["Ca", "Cb", "Cc"])

    # KKT-HardNet receives scaled coordinates. Convert symbolic variables back
    # to physical units before building the original nonlinear constraints.
    s_T, s_cao, s_ca, s_cb, s_cc = [float(v) for v in scales]
    T = x.Temperature * s_T
    cao = x.Cao * s_cao
    ca = y.Ca * s_ca
    cb = y.Cb * s_cb
    cc = y.Cc * s_cc

    # Unlike 1D, kf and kr depend on T. SymPy keeps these exponentials symbolic
    # and KKT-HardNet differentiates/projects the resulting nonlinear system.
    kf = AFO * kkt_exp(-EAF / (R * T))
    kr = ARO * kkt_exp(-EAR / (R * T))
    model.constraints.add(
        (cao - ca - kf * ca * (cb**2) * TAU + kr * cc * TAU) == 0,
        (cao - ca + CBO - cb + CCO - cc) == 0,
    )

    if parameters_path is not None and variables_path is not None:
        model.dataset(parameters=parameters_path, variables=variables_path)
    return model


def physical_residuals(x_physical: np.ndarray, predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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


def compute_metrics(y_pred_scaled: np.ndarray, y_true_scaled: np.ndarray, x_test_physical: np.ndarray, y_true_physical: np.ndarray, output_scales: np.ndarray):
    y_pred_physical = y_pred_scaled * output_scales
    g1, g2 = physical_residuals(x_test_physical, y_pred_physical)
    abs_residuals = np.abs(np.column_stack((g1, g2)))

    batch_mses: List[float] = []
    batch_violations: List[float] = []
    for sl in batched_slices(len(y_pred_scaled), BATCH_SIZE):
        batch_mses.append(float(np.mean((y_pred_scaled[sl] - y_true_scaled[sl]) ** 2)))
        batch_violations.append(float(np.mean(abs_residuals[sl])))

    per_output_rmse = np.sqrt(np.mean((y_pred_physical - y_true_physical) ** 2, axis=0))
    metrics = {
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
    return metrics, y_pred_physical, g1, g2


def write_report(report_path: Path, metrics: Dict[str, float]) -> None:
    with report_path.open("w", encoding="utf-8") as f:
        f.write("KKT-HardNet 2D CSTR - PL-KKT-hPINN-style benchmark\n")
        f.write("=" * 68 + "\n")
        f.write("projection_backend_for_test: jax\n")
        f.write(f"kkt_gn_tolerance: {PROJECTION_TOLERANCE:.3e}\n")
        f.write(f"max_projection_iterations: {MAX_PROJECTION_ITERATIONS}\n")
        f.write(f"test_batch_size: {BATCH_SIZE}\n")
        f.write("note: gn_tol is KKT-HardNet's native KKT/Gauss-Newton stopping quantity\n")
        f.write("\nMetrics\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.12e}\n")


def train_job(args: argparse.Namespace, prepared: Dict[str, np.ndarray]) -> None:
    results_dir = Path(args.results_dir)
    models_dir = Path(args.models_dir)
    input_root = Path(args.inputs_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    parameters_path, variables_path, mapping_path = write_kkt_training_csvs(prepared, args.run, input_root)

    print("\nKKT-HardNet 2D PL-style benchmark configuration")
    print(f"split: train={len(prepared['train_idx'])}, val={len(prepared['val_idx'])}, test={len(prepared['test_idx'])}")
    print(f"network: [2, {HIDDEN_NEURONS}, {HIDDEN_NEURONS}, 3]")
    print(f"epochs={EPOCHS}, batch_size={BATCH_SIZE}, learning_rate={LEARNING_RATE}")
    print(f"dtype=float64, backend=cpu, model_seed={model_seed_for_run(args.run)}")
    print(f"JAX x64 enabled: {jax.config.jax_enable_x64}")
    print(f"projection tolerance={PROJECTION_TOLERANCE}, max_it={MAX_PROJECTION_ITERATIONS}")
    print(f"KKT internal row mapping: {mapping_path}")

    model = build_symbolic_model(prepared["scales"], args.run, parameters_path, variables_path)
    result = model.model()

    output_dir = Path(result["output_dir"]).resolve()
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Expected KKT metadata not found: {metadata_path}")

    pointer_path = models_dir / f"kkt_2d_run_{args.run}_metadata_path.txt"
    pointer_path.write_text(str(metadata_path), encoding="utf-8")

    benchmark_metadata = {
        "run": args.run,
        "problem_dimension": "2D inputs: T and Cao",
        "model_seed": model_seed_for_run(args.run),
        "split_seed": SPLIT_SEED,
        "maxabs_scales": prepared["scales"].tolist(),
        "kkt_run_dir": str(output_dir),
        "kkt_metadata": str(metadata_path),
        "configuration": {
            "architecture": [2, HIDDEN_NEURONS, HIDDEN_NEURONS, 3],
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "dtype": "float64",
            "device": "cpu",
            "projection_tolerance": PROJECTION_TOLERANCE,
            "max_projection_iterations": MAX_PROJECTION_ITERATIONS,
            "epoch_mlp": EPOCH_MLP,
            "cons_alpha": CONS_ALPHA,
            "checkpoint_policy": "final_epoch",
            "validation_effective_shuffle": False,
            "test_shuffle": False,
            "train_shuffle": True,
            "linearization_used": False,
        },
    }
    with (models_dir / f"kkt_2d_run_{args.run}_benchmark_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(benchmark_metadata, f, indent=2)

    save_split_description(prepared, results_dir / "pl_2d_split_indices.csv")
    print(f"Saved metadata pointer: {pointer_path}")


def experiment_job(args: argparse.Namespace, prepared: Dict[str, np.ndarray]) -> None:
    results_dir = Path(args.results_dir)
    models_dir = Path(args.models_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    pointer_path = models_dir / f"kkt_2d_run_{args.run}_metadata_path.txt"
    if not pointer_path.exists():
        raise FileNotFoundError(f"Metadata pointer not found: {pointer_path}. Run --job train first.")
    metadata_path = Path(pointer_path.read_text(encoding="utf-8").strip())
    if not metadata_path.exists():
        raise FileNotFoundError(f"KKT-HardNet metadata not found: {metadata_path}")

    x_test_scaled = prepared["x_test_scaled"]
    y_test_scaled = prepared["y_test_scaled"]
    x_test_raw = prepared["x_test_raw"]
    y_test_raw = prepared["y_test_raw"]
    output_scales = prepared["scales"][2:5]

    eval_start = time.perf_counter()
    model = KKTHardNet().load(str(metadata_path))

    predicted_batches: List[np.ndarray] = []
    prediction_time = 0.0
    for sl in batched_slices(len(x_test_scaled), BATCH_SIZE):
        # Time ONLY NN forward + KKT projection
        prediction_start = time.perf_counter()
        # Float32 input keeps JAX/projector computation aligned with ENFORCE.
        pred = model.predict(np.asarray(x_test_scaled[sl], dtype=np.float64), projection_backend="jax")
        
        pred = np.asarray(pred, dtype=np.float64)  # metrics only, after projection
        prediction_time += time.perf_counter() - prediction_start
        
        if pred.ndim == 1:
            pred = pred.reshape(1, -1)
        predicted_batches.append(pred)

    y_pred_scaled = np.concatenate(predicted_batches, axis=0)
    if y_pred_scaled.shape != y_test_scaled.shape:
        raise RuntimeError(f"Unexpected prediction shape {y_pred_scaled.shape}; expected {y_test_scaled.shape}")

    metrics, y_pred_physical, g1, g2 = compute_metrics(
        y_pred_scaled, y_test_scaled, x_test_raw, y_test_raw, output_scales
    )

    report_path = results_dir / f"kkt_2d_run_{args.run}_report.txt"
    write_report(report_path, metrics)
    evaluation_time = time.perf_counter() - eval_start

    # Detailed pointwise diagnostics are outside the PL-style timer.
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
    ).to_csv(results_dir / f"kkt_2d_run_{args.run}_test_predictions.csv", index=False)

    payload = {**metrics, "run": args.run, "evaluation_time_sec": evaluation_time, "projection_backend": "jax", "prediction_time_sec": prediction_time,}
    with (results_dir / f"kkt_2d_run_{args.run}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n========== KKT-HARDNET 2D PL-STYLE TEST RESULTS ==========")
    print(f"Scaled RMSE (PL-style): {metrics['rmse_pl_style_scaled']:.10e}")
    print(f"Original nonlinear violation (PL-style): {metrics['original_nonlinear_violation_pl_style']:.10e}")
    print(f"Physical overall RMSE: {metrics['rmse_overall_physical']:.10e}")
    print(f"overall mean |g|: {metrics['overall_mean_abs_violation']:.10e}")
    print(f"overall max |g|:  {metrics['overall_max_abs_violation']:.10e}")
    print(f"Evaluation time: {evaluation_time:.6f} s")
    print(f"Pure prediction time: {prediction_time:.6f} s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--job", choices=["train", "experiment"], required=True)
    p.add_argument("--data", default="data_cstr_2d.csv")
    p.add_argument("--run", type=int, default=1)
    p.add_argument("--models-dir", default="models_2d")
    p.add_argument("--results-dir", default="results_2d")
    p.add_argument("--inputs-dir", default="kkthn_2d_pl_inputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.run < 1:
        raise ValueError("--run must be >= 1")
    print(f"JAX x64 enabled: {jax.config.jax_enable_x64}")
    prepared = load_and_prepare_data(Path(args.data))
    if args.job == "train":
        train_job(args, prepared)
    else:
        experiment_job(args, prepared)


if __name__ == "__main__":
    main()
