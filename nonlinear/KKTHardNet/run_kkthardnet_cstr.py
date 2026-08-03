import os

# ============================================================
# Force the CPU JAX backend before importing KKT-HardNet.
# ============================================================

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from kkthn import KKTHardNet


# ============================================================
# 1. Case-study constants
# ============================================================

SEED = 42

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


# ============================================================
# 2. Command-line options
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train KKT-HardNet for the isothermal CSTR case study "
            "with input Cao, outputs Ca/Cb/Cc, and equality "
            "constraints g1 and g2."
        )
    )
    parser.add_argument(
        "--data",
        default="data.csv",
        help="CSV containing Cao, Ca, Cb, and Cc. Default: data.csv",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="Number of training epochs. Default: 2000",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size. Default: 64",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=100,
        help="Epoch logging interval. Default: 100",
    )
    parser.add_argument(
        "--gn-max-iters",
        type=int,
        default=30,
        help="Maximum Gauss-Newton projection iterations. Default: 30",
    )
    parser.add_argument(
        "--gn-tol",
        type=float,
        default=1.0e-6,
        help="KKT projection residual tolerance. Default: 1e-6",
    )
    return parser.parse_args()


# ============================================================
# 3. Data preparation
# ============================================================

def prepare_data(data_path: Path) -> tuple[Path, Path, pd.DataFrame]:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path.resolve()}"
        )

    df = pd.read_csv(data_path)

    required_columns = ["Cao", "Ca", "Cb", "Cc"]
    missing = [name for name in required_columns if name not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}. "
            f"Expected: {required_columns}"
        )

    clean = (
        df[required_columns]
        .dropna()
        .astype(np.float64)
        .reset_index(drop=True)
    )

    if len(clean) < 2:
        raise ValueError("At least two complete data rows are required.")

    input_dir = Path("kkthn_cstr_inputs")
    input_dir.mkdir(parents=True, exist_ok=True)

    parameters_path = input_dir / "parameters.csv"
    variables_path = input_dir / "variables.csv"

    clean[["Cao"]].to_csv(parameters_path, index=False)
    clean[["Ca", "Cb", "Cc"]].to_csv(variables_path, index=False)

    return parameters_path, variables_path, clean


# ============================================================
# 4. Metrics
# ============================================================

def constraint_residuals(
    cao: np.ndarray,
    ca: np.ndarray,
    cb: np.ndarray,
    cc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    g1 = (
        cao
        - ca
        - KF * ca * cb**2 * TAU
        + KR * cc * TAU
    )

    g2 = (
        cao
        - ca
        + CBO
        - cb
        + CCO
        - cc
    )

    return g1, g2


def rmse_by_output(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, float]:
    per_output = np.sqrt(
        np.mean((predictions - targets) ** 2, axis=0)
    )
    overall = float(
        np.sqrt(np.mean((predictions - targets) ** 2))
    )
    return per_output, overall


def print_prediction_metrics(
    label: str,
    cao: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    ca = predictions[:, 0]
    cb = predictions[:, 1]
    cc = predictions[:, 2]

    rmse_outputs, overall_rmse = rmse_by_output(
        predictions,
        targets,
    )

    g1, g2 = constraint_residuals(cao, ca, cb, cc)

    metrics = {
        "rmse_ca": float(rmse_outputs[0]),
        "rmse_cb": float(rmse_outputs[1]),
        "rmse_cc": float(rmse_outputs[2]),
        "rmse_overall": overall_rmse,
        "g1_mean_abs": float(np.mean(np.abs(g1))),
        "g2_mean_abs": float(np.mean(np.abs(g2))),
        "overall_mean_abs_violation": float(
            np.mean(np.abs(np.column_stack((g1, g2))))
        ),
        "g1_max_abs": float(np.max(np.abs(g1))),
        "g2_max_abs": float(np.max(np.abs(g2))),
        "overall_max_abs_violation": float(
            np.max(np.abs(np.column_stack((g1, g2))))
        ),
    }

    print(f"\n========== {label} ==========")

    print("\nRMSE")
    print(f"Ca RMSE:       {metrics['rmse_ca']:.10e}")
    print(f"Cb RMSE:       {metrics['rmse_cb']:.10e}")
    print(f"Cc RMSE:       {metrics['rmse_cc']:.10e}")
    print(f"Overall RMSE:  {metrics['rmse_overall']:.10e}")

    print("\nMean absolute constraint violation")
    print(f"g1 violation:  {metrics['g1_mean_abs']:.10e}")
    print(f"g2 violation:  {metrics['g2_mean_abs']:.10e}")
    print(
        "Overall:       "
        f"{metrics['overall_mean_abs_violation']:.10e}"
    )

    print("\nWorst-case absolute constraint violation")
    print(f"max |g1|:      {metrics['g1_max_abs']:.10e}")
    print(f"max |g2|:      {metrics['g2_max_abs']:.10e}")
    print(
        "overall max:   "
        f"{metrics['overall_max_abs_violation']:.10e}"
    )

    return metrics


# ============================================================
# 5. Main workflow
# ============================================================

def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    parameters_path, variables_path, clean_df = prepare_data(
        data_path
    )

    print("Using JAX backend: CPU")
    print(f"Rows used: {len(clean_df)}")
    print(f"kf = {KF:.10e}")
    print(f"kr = {KR:.10e}")
    print(f"Parameters CSV: {parameters_path}")
    print(f"Variables CSV:  {variables_path}")

    # --------------------------------------------------------
    # Training setup
    #
    # epoch_mlp=100:
    #   train the unconstrained MLP for 100 epochs, then train
    #   through the KKT projection layer.
    #
    # cons_alpha=0.5:
    #   penalize displacement between raw and projected outputs.
    #   This matches the 0.5 displacement weight used in the
    #   earlier ENFORCE test script.
    # --------------------------------------------------------

    train_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": 1.0e-3,
        "train_frac": 0.8,
        "hidden_size": 64,
        "hidden_layers": 2,
        "seed": SEED,
        "dtype": "float64",
        "print_every": args.print_every,
        "drop_last": False,
        "epoch_mlp": 100,
        "cons_alpha": 0.5,
    }

    projection_config = {
        "fb_eps": 1.0e-8,
        "gn_max_iters": args.gn_max_iters,
        "gn_tol": args.gn_tol,
        "gn_reg": 1.0e-3,
        "newton_step_length": 0.5,
        "armijo_alpha": 1.0e-4,
        "armijo_beta": 0.5,
        "max_backtrack_iter": 10,
        "backward_reg": 1.0e-8,
    }

    # --------------------------------------------------------
    # Define the direct physical CSTR model:
    #
    # parameter: Cao
    # variables: Ca, Cb, Cc
    #
    # This is NOT the recovered-KKT representation with Cb1/Cb2.
    # --------------------------------------------------------

    model = KKTHardNet(
        name="CSTR_KKTHardNet",
        train=train_config,
        projection=projection_config,
    )

    x = model.add_parameter(["Cao"])
    y = model.add_variable(["Ca", "Cb", "Cc"])

    model.constraints.add(
        # g1 = 0: nonlinear reaction balance
        (
            x.Cao
            - y.Ca
            - KF * y.Ca * (y.Cb**2) * TAU
            + KR * y.Cc * TAU
        ) == 0,

        # g2 = 0: overall material balance
        (
            x.Cao
            - y.Ca
            + CBO
            - y.Cb
            + CCO
            - y.Cc
        ) == 0,
    )

    # Positivity inequalities are intentionally omitted here so
    # the enforced constraints match the earlier ENFORCE run:
    # only original g1 = 0 and g2 = 0.

    model.dataset(
        parameters=parameters_path,
        variables=variables_path,
    )

    # Supervised surrogate training.
    result = model.model()

    run_dir = Path(result["output_dir"])
    print(f"\nKKT-HardNet run directory: {run_dir}")

    # --------------------------------------------------------
    # Validation predictions returned by KKT-HardNet.
    # Y_hat  = raw MLP prediction
    # Y_proj = KKT-projected prediction
    # --------------------------------------------------------

    validation = result["val_predictions"]

    cao_val = np.asarray(
        validation["X"],
        dtype=np.float64,
    )[:, 0]

    y_true = np.asarray(
        validation["Y"],
        dtype=np.float64,
    )

    y_raw = np.asarray(
        validation["Y_hat"],
        dtype=np.float64,
    )

    y_projected = np.asarray(
        validation["Y_proj"],
        dtype=np.float64,
    )

    sample_indices = np.asarray(
        validation["sample_indices"],
        dtype=int,
    )

    raw_metrics = print_prediction_metrics(
        "RAW MLP VALIDATION RESULTS",
        cao_val,
        y_raw,
        y_true,
    )

    projected_metrics = print_prediction_metrics(
        "KKT-HARDNET PROJECTED VALIDATION RESULTS",
        cao_val,
        y_projected,
        y_true,
    )

    # --------------------------------------------------------
    # Save detailed validation predictions and residuals.
    # --------------------------------------------------------

    raw_g1, raw_g2 = constraint_residuals(
        cao_val,
        y_raw[:, 0],
        y_raw[:, 1],
        y_raw[:, 2],
    )

    proj_g1, proj_g2 = constraint_residuals(
        cao_val,
        y_projected[:, 0],
        y_projected[:, 1],
        y_projected[:, 2],
    )

    output_df = pd.DataFrame(
        {
            "sample_index": sample_indices,
            "Cao": cao_val,
            "Ca_true": y_true[:, 0],
            "Cb_true": y_true[:, 1],
            "Cc_true": y_true[:, 2],
            "Ca_raw": y_raw[:, 0],
            "Cb_raw": y_raw[:, 1],
            "Cc_raw": y_raw[:, 2],
            "Ca_projected": y_projected[:, 0],
            "Cb_projected": y_projected[:, 1],
            "Cc_projected": y_projected[:, 2],
            "g1_raw": raw_g1,
            "g2_raw": raw_g2,
            "abs_g1_raw": np.abs(raw_g1),
            "abs_g2_raw": np.abs(raw_g2),
            "g1_projected": proj_g1,
            "g2_projected": proj_g2,
            "abs_g1_projected": np.abs(proj_g1),
            "abs_g2_projected": np.abs(proj_g2),
        }
    )

    results_path = run_dir / "cstr_validation_metrics.csv"
    output_df.to_csv(results_path, index=False)

    summary_df = pd.DataFrame(
        [
            {"prediction_type": "raw_mlp", **raw_metrics},
            {
                "prediction_type": "kkt_projected",
                **projected_metrics,
            },
        ]
    )

    summary_path = run_dir / "cstr_metric_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved detailed CSTR metrics: {results_path}")
    print(f"Saved CSTR metric summary:   {summary_path}")

    # Package-generated summary, including dimensions, sample
    # counts, maximum violation, and timing estimates.
    print()
    model.summary()


if __name__ == "__main__":
    main()