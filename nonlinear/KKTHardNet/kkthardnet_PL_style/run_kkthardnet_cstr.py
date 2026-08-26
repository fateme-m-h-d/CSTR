"""PL-KKT-hPINN-style KKT-HardNet benchmark for the 1D isothermal CSTR.

This script keeps KKT-HardNet's own training/projection implementation, but
aligns the data split, scaling, architecture size, optimizer settings, precision,
and timing boundary with the 1D PL-KKT-hPINN benchmark as closely as practical.

KKT-HardNet's native data API creates only train/validation subsets and uses its
own NumPy default_rng permutation. To guarantee the *same actual 90 training and
30 validation rows* as PL-KKT-hPINN, this script arranges the 120 non-test rows
so KKT-HardNet's internal split recovers exactly those two sets. The 30 PL test
rows are never written to KKT-HardNet's training CSVs and are evaluated only
after the trained model is reloaded.
"""

import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "False"  # KKT-HardNet's PL-style benchmark uses float32 JAX for deterministic comparison

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from kkthn import KKTHardNet

import jax

# KKT-HardNet enables x64 internally, so force it back off
# after importing the package.
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp

print("JAX x64 enabled:", jax.config.jax_enable_x64)
print("JAX default float:", jnp.asarray([1.0]).dtype)

# ---------------------------------------------------------------------------
# Fixed benchmark configuration
# ---------------------------------------------------------------------------

SPLIT_SEED = 42
BASE_MODEL_SEED = 42
VAL_RATIO = 0.20
TEST_RATIO = 0.20

HIDDEN_NEURONS = 32
HIDDEN_LAYERS = 2
EPOCHS = 1000
BATCH_SIZE = 16
LEARNING_RATE = 1.0e-4

PROJECTION_TOLERANCE = 1.0e-6
MAX_PROJECTION_ITERATIONS = 100

# KKT-HardNet-specific training settings retained from the prior working CSTR
# script. They do not have PL-KKT-hPINN equivalents.
EPOCH_MLP = 100
CONS_ALPHA = 0.5

# CSTR constants
TAU = 10.0
CBO = 2.0
CCO = 0.0
T_ISO = 350.0
R = 8.314
AFO = 1.0e13
EAF = 90000.0
ARO = 1.0e11
EAR = 80000.0
KF = AFO * math.exp(-EAF / (R * T_ISO))
KR = ARO * math.exp(-EAR / (R * T_ISO))

REQUIRED_COLUMNS = ["Cao", "Ca", "Cb", "Cc"]


# ---------------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------------


def load_and_prepare_data(data_path: Path) -> Dict[str, np.ndarray]:
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path.resolve()}")

    df = pd.read_csv(data_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing}. Expected columns: {REQUIRED_COLUMNS}"
        )

    raw = (
        df[REQUIRED_COLUMNS]
        .dropna()
        .to_numpy(dtype=np.float64, copy=True)
    )
    n = raw.shape[0]
    if n != 150:
        print(
            f"WARNING: PL case study uses 150 rows; this file contains {n}. "
            "The same 60/20/20 logic will still be applied."
        )

    # Same full-data MaxAbs scaling used in PL-KKT-hPINN.
    scales = np.max(np.abs(raw), axis=0)
    scales[scales == 0.0] = 1.0
    scaled = raw / scales

    # Equivalent to sklearn.utils.shuffle(..., random_state=42).
    permutation = np.random.RandomState(SPLIT_SEED).permutation(n)
    n_val = int(VAL_RATIO * n)
    n_test = int(TEST_RATIO * n)
    n_train = n - n_val - n_test

    train_idx = permutation[:n_train]
    val_idx = permutation[n_train : n_train + n_val]
    test_idx = permutation[n_train + n_val :]

    return {
        "raw": raw,
        "scaled": scaled,
        "scales": scales,
        "permutation": permutation,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "x_test_scaled": scaled[test_idx, 0:1],
        "y_test_scaled": scaled[test_idx, 1:4],
        "x_test_raw": raw[test_idx, 0:1],
        "y_test_raw": raw[test_idx, 1:4],
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


def model_seed_for_run(run: int) -> int:
    return BASE_MODEL_SEED + (run - 1)


def write_kkt_training_csvs(
    prepared: Dict[str, np.ndarray], run: int, root: Path
) -> Tuple[Path, Path, Path]:
    """Arrange 120 rows so KKT-HardNet's internal split exactly matches PL.

    KKT-HardNet internally does roughly:
        idx = arange(N)
        default_rng(seed).shuffle(idx)
        train = idx[:int(train_frac*N)]
        val   = idx[int(train_frac*N):]

    We invert that placement here. For N=120 and train_frac=.75, its 90
    selected train positions contain exactly PL's 90 train rows, and its 30
    validation positions contain exactly PL's 30 validation rows.  We also
    pre-invert KKT-HardNet's fixed validation minibatch permutation so the
    effective validation batches follow PL's deterministic unshuffled order.
    """

    train_rows = prepared["scaled"][prepared["train_idx"]]
    val_rows = prepared["scaled"][prepared["val_idx"]]
    if len(train_rows) != 90 or len(val_rows) != 30:
        raise ValueError(
            "This exact arrangement expects 90 training and 30 validation rows."
        )

    n_total = len(train_rows) + len(val_rows)  # 120; test rows excluded
    package_idx = np.arange(n_total)
    rng = np.random.default_rng(model_seed_for_run(run))
    rng.shuffle(package_idx)
    n_package_train = int(0.75 * n_total)
    package_train_positions = package_idx[:n_package_train]
    package_val_positions = package_idx[n_package_train:]

    arranged = np.empty((n_total, 4), dtype=np.float64)
    source_index = np.empty(n_total, dtype=np.int64)
    intended_split = np.empty(n_total, dtype=object)

    arranged[package_train_positions] = train_rows
    source_index[package_train_positions] = prepared["train_idx"]
    intended_split[package_train_positions] = "train"

    # KKT-HardNet's validation loop internally calls _iterate_minibatches(...,
    # seed=cfg.seed), which applies the same deterministic permutation to the
    # validation subset every epoch.  The public API does not expose a
    # validation-shuffle switch.  Pre-invert that fixed permutation here so
    # the *effective* validation order seen by the package is exactly PL's
    # unshuffled validation order (first 16, then remaining 14).
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
    parameters_path = input_dir / "parameters.csv"
    variables_path = input_dir / "variables.csv"
    mapping_path = input_dir / "kkt_internal_row_mapping.csv"

    pd.DataFrame(arranged[:, 0:1], columns=["Cao"]).to_csv(
        parameters_path, index=False
    )
    pd.DataFrame(arranged[:, 1:4], columns=["Ca", "Cb", "Cc"]).to_csv(
        variables_path, index=False
    )
    pd.DataFrame(
        {
            "kkt_csv_row": np.arange(n_total),
            "original_source_row_index_zero_based": source_index,
            "intended_pl_split": intended_split,
        }
    ).to_csv(mapping_path, index=False)

    return parameters_path, variables_path, mapping_path


# ---------------------------------------------------------------------------
# Symbolic model definition
# ---------------------------------------------------------------------------


def build_symbolic_model(
    scales: np.ndarray,
    run: int,
    parameters_path: Path | None = None,
    variables_path: Path | None = None,
) -> KKTHardNet:
    train_config = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        # 120 package rows = 90 train + 30 val
        "train_frac": 0.75,
        "hidden_size": HIDDEN_NEURONS,
        "hidden_layers": HIDDEN_LAYERS,
        "seed": model_seed_for_run(run),
        "dtype": "float32",
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

    model = KKTHardNet(
        name=f"CSTR_KKTHardNet_PLStyle_run{run}",
        train=train_config,
        projection=projection_config,
    )

    # The package sees scaled coordinates. Convert symbolic x/y back to
    # physical concentrations inside the constraints so the enforced equations
    # remain the original CSTR equations.
    x = model.add_parameter(["Cao"])
    y = model.add_variable(["Ca", "Cb", "Cc"])

    s_cao, s_ca, s_cb, s_cc = [float(v) for v in scales]
    cao = x.Cao * s_cao
    ca = y.Ca * s_ca
    cb = y.Cb * s_cb
    cc = y.Cc * s_cc

    model.constraints.add(
        (cao - ca - KF * ca * (cb**2) * TAU + KR * cc * TAU) == 0,
        (cao - ca + CBO - cb + CCO - cc) == 0,
    )

    if parameters_path is not None and variables_path is not None:
        model.dataset(parameters=parameters_path, variables=variables_path)

    return model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def physical_residuals(
    cao: np.ndarray, predictions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    ca = predictions[:, 0]
    cb = predictions[:, 1]
    cc = predictions[:, 2]
    g1 = cao - ca - KF * ca * cb**2 * TAU + KR * cc * TAU
    g2 = cao - ca + CBO - cb + CCO - cc
    return g1, g2


def batched_slices(n: int, batch_size: int) -> Iterable[slice]:
    for start in range(0, n, batch_size):
        yield slice(start, min(start + batch_size, n))


def compute_metrics(
    y_pred_scaled: np.ndarray,
    y_true_scaled: np.ndarray,
    x_test_physical: np.ndarray,
    y_true_physical: np.ndarray,
    output_scales: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    y_pred_physical = y_pred_scaled * output_scales
    g1, g2 = physical_residuals(x_test_physical[:, 0], y_pred_physical)
    abs_residuals = np.abs(np.column_stack((g1, g2)))

    batch_mses: List[float] = []
    batch_original_violations: List[float] = []
    for sl in batched_slices(len(y_pred_scaled), BATCH_SIZE):
        batch_mses.append(
            float(np.mean((y_pred_scaled[sl] - y_true_scaled[sl]) ** 2))
        )
        batch_original_violations.append(float(np.mean(abs_residuals[sl])))

    per_output_rmse = np.sqrt(
        np.mean((y_pred_physical - y_true_physical) ** 2, axis=0)
    )

    metrics = {
        "rmse_pl_style_scaled": float(np.sqrt(np.mean(batch_mses))),
        "original_nonlinear_violation_pl_style": float(
            np.mean(batch_original_violations)
        ),
        "rmse_ca_physical": float(per_output_rmse[0]),
        "rmse_cb_physical": float(per_output_rmse[1]),
        "rmse_cc_physical": float(per_output_rmse[2]),
        "rmse_overall_physical": float(
            np.sqrt(np.mean((y_pred_physical - y_true_physical) ** 2))
        ),
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
        f.write("KKT-HardNet CSTR - PL-KKT-hPINN-style benchmark\n")
        f.write("=" * 62 + "\n")
        f.write(f"projection_backend_for_test: jax\n")
        f.write(f"projection_tolerance: {PROJECTION_TOLERANCE:.3e}\n")
        f.write(f"max_projection_iterations: {MAX_PROJECTION_ITERATIONS}\n")
        f.write(f"test_batch_size: {BATCH_SIZE}\n")
        f.write("\nMetrics\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.12e}\n")


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


def train_job(args: argparse.Namespace, prepared: Dict[str, np.ndarray]) -> None:
    results_dir = Path(args.results_dir)
    models_dir = Path(args.models_dir)
    input_root = Path(args.inputs_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    parameters_path, variables_path, mapping_path = write_kkt_training_csvs(
        prepared, args.run, input_root
    )

    print("\nPL-style benchmark configuration")
    print(f"split: train={len(prepared['train_idx'])}, val={len(prepared['val_idx'])}, test={len(prepared['test_idx'])}")
    print(f"network: [1, {HIDDEN_NEURONS}, {HIDDEN_NEURONS}, 3]")
    print(f"epochs={EPOCHS}, batch_size={BATCH_SIZE}, learning_rate={LEARNING_RATE}")
    print(f"dtype=float32, backend=cpu, model_seed={model_seed_for_run(args.run)}")
    print(f"projection tolerance={PROJECTION_TOLERANCE}, max_it={MAX_PROJECTION_ITERATIONS}")
    print(f"KKT internal row mapping: {mapping_path}")

    model = build_symbolic_model(
        prepared["scales"],
        args.run,
        parameters_path=parameters_path,
        variables_path=variables_path,
    )
    result = model.model()

    output_dir = Path(result["output_dir"]).resolve()
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Expected KKT metadata not found: {metadata_path}")

    pointer_path = models_dir / f"kkt_run_{args.run}_metadata_path.txt"
    pointer_path.write_text(str(metadata_path), encoding="utf-8")

    benchmark_metadata = {
        "run": args.run,
        "model_seed": model_seed_for_run(args.run),
        "split_seed": SPLIT_SEED,
        "maxabs_scales": prepared["scales"].tolist(),
        "kkt_run_dir": str(output_dir),
        "kkt_metadata": str(metadata_path),
        "configuration": {
            "hidden_neurons": HIDDEN_NEURONS,
            "hidden_layers": HIDDEN_LAYERS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "dtype": "float32",
            "device": "cpu",
            "projection_tolerance": PROJECTION_TOLERANCE,
            "max_projection_iterations": MAX_PROJECTION_ITERATIONS,
            "epoch_mlp": EPOCH_MLP,
            "cons_alpha": CONS_ALPHA,
            "checkpoint_policy": "final_epoch",
            "validation_effective_shuffle": False,
            "test_shuffle": False,
            "train_shuffle": True,
        },
    }
    with (models_dir / f"kkt_run_{args.run}_benchmark_metadata.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(benchmark_metadata, f, indent=2)

    save_split_description(prepared, results_dir / "pl_split_indices.csv")
    print(f"Saved metadata pointer: {pointer_path}")


def experiment_job(args: argparse.Namespace, prepared: Dict[str, np.ndarray]) -> None:
    results_dir = Path(args.results_dir)
    models_dir = Path(args.models_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    pointer_path = models_dir / f"kkt_run_{args.run}_metadata_path.txt"
    if not pointer_path.exists():
        raise FileNotFoundError(
            f"Metadata pointer not found: {pointer_path}. Run --job train first."
        )
    metadata_path = Path(pointer_path.read_text(encoding="utf-8").strip())
    if not metadata_path.exists():
        raise FileNotFoundError(f"KKT-HardNet metadata not found: {metadata_path}")

    # Data loading/scaling/split is intentionally complete before timing begins.
    x_test_scaled = prepared["x_test_scaled"]
    y_test_scaled = prepared["y_test_scaled"]
    x_test_raw = prepared["x_test_raw"]
    y_test_raw = prepared["y_test_raw"]
    output_scales = prepared["scales"][1:4]

    eval_start = time.perf_counter()

    # Model construction/loading is inside the timer, as in PL evaluate_model().
    # load() reconstructs the saved symbolic problem and trained MLP.
    model = KKTHardNet().load(str(metadata_path))

    predicted_batches: List[np.ndarray] = []
    # Use JAX explicitly for deterministic comparison with the package's JAX
    # training/projection implementation and to avoid hardware/compiler-specific
    # native-C backend differences.
    for sl in batched_slices(len(x_test_scaled), BATCH_SIZE):
        pred = model.predict(
            np.asarray(x_test_scaled[sl], dtype=np.float32),
            projection_backend="jax",
        )
        pred = np.asarray(pred, dtype=np.float64)
        if pred.ndim == 1:
            pred = pred.reshape(1, -1)
        predicted_batches.append(pred)

    y_pred_scaled = np.concatenate(predicted_batches, axis=0)
    if y_pred_scaled.shape != y_test_scaled.shape:
        raise RuntimeError(
            f"Unexpected prediction shape {y_pred_scaled.shape}; "
            f"expected {y_test_scaled.shape}"
        )

    metrics, y_pred_physical, g1, g2 = compute_metrics(
        y_pred_scaled=y_pred_scaled,
        y_true_scaled=y_test_scaled,
        x_test_physical=x_test_raw,
        y_true_physical=y_test_raw,
        output_scales=output_scales,
    )

    prediction_path = results_dir / f"kkt_run_{args.run}_test_predictions.csv"
    

    report_path = results_dir / f"kkt_run_{args.run}_report.txt"
    write_report(report_path, metrics)

    evaluation_time = time.perf_counter() - eval_start
    
    pd.DataFrame(
        {
            "source_row_index_zero_based": prepared["test_idx"],
            "Cao": x_test_raw[:, 0],
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
    ).to_csv(prediction_path, index=False)

    metrics_payload = {
        **metrics,
        "run": args.run,
        "evaluation_time_sec": evaluation_time,
        "projection_backend": "jax",
    }
    metrics_path = results_dir / f"kkt_run_{args.run}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print("\n========== KKT-HARDNET PL-STYLE TEST RESULTS ==========")
    print(f"Scaled RMSE (PL-style): {metrics['rmse_pl_style_scaled']:.10e}")
    print(
        "Original nonlinear violation (PL-style): "
        f"{metrics['original_nonlinear_violation_pl_style']:.10e}"
    )
    print(f"Physical overall RMSE: {metrics['rmse_overall_physical']:.10e}")
    print(f"mean |g1|: {metrics['g1_mean_abs']:.10e}")
    print(f"mean |g2|: {metrics['g2_mean_abs']:.10e}")
    print(
        "overall mean |g|: "
        f"{metrics['overall_mean_abs_violation']:.10e}"
    )
    print(
        "overall max |g|:  "
        f"{metrics['overall_max_abs_violation']:.10e}"
    )
    print(f"Evaluation time: {evaluation_time:.6f} s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--job", choices=["train", "experiment"], required=True)
    p.add_argument("--data", default="data.csv")
    p.add_argument("--run", type=int, default=1)
    p.add_argument("--models-dir", default="models")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--inputs-dir", default="kkthn_pl_inputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.run < 1:
        raise ValueError("--run must be >= 1")
    prepared = load_and_prepare_data(Path(args.data))
    if args.job == "train":
        train_job(args, prepared)
    else:
        experiment_job(args, prepared)


if __name__ == "__main__":
    main()
