"""Offline adaptive piecewise-linearization for the 2D CSTR example.

This replaces the fixed Cartesian T x Cao grid with a data-driven binary
hyper-rectangle partition built before neural-network training.

The partition is based only on the available data and the known analytical
constraints. No simulator calls and no trained-network predictions are used.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import sympy as sym

# -----------------------------------------------------------------------------
# CSTR constants and column definitions
# -----------------------------------------------------------------------------

V = 10.0
Q = 1.0
tau = V / Q

Afo = 10e12
Eaf = 90000.0
Aro = 10e10
Ear = 80000.0
R = 8.314

Cbo = 2.0
Cco = 0.0

INPUT_COLUMNS = ["Temperature (T)", "Cao"]
OUTPUT_COLUMNS = ["Ca", "Cb", "Cc"]
ALL_COLUMNS = INPUT_COLUMNS + OUTPUT_COLUMNS

INPUT_SHORT_NAMES = ["T", "Cao"]
OUTPUT_SHORT_NAMES = ["Ca", "Cb", "Cc"]


@dataclass
class ConstraintFunctions:
    """Callable nonlinear constraint and its first derivatives."""

    value: callable
    gradient: tuple[callable, ...]


@dataclass
class RegionModel:
    """A leaf region, its local Taylor model, and its offline error estimates."""

    low: np.ndarray
    high: np.ndarray
    indices: np.ndarray
    depth: int
    anchor_index: int
    anchor_z: np.ndarray
    gradient: np.ndarray
    b_reaction: float
    remainder_values: np.ndarray
    projected_residual_values: np.ndarray
    remainder_score: float
    projected_score: float
    criterion_score: float
    stop_reason: str = ""


# -----------------------------------------------------------------------------
# Constraint definition and local linearization
# -----------------------------------------------------------------------------


def build_constraint_functions() -> ConstraintFunctions:
    """Build the known nonlinear reaction constraint and its gradient.

    Variable order is [T, Cao, Ca, Cb, Cc].
    """

    T_sym, Cao_sym, Ca_sym, Cb_sym, Cc_sym = sym.symbols(
        "T Cao Ca Cb Cc", real=True
    )
    variables = (T_sym, Cao_sym, Ca_sym, Cb_sym, Cc_sym)

    kf_sym = sym.Float(Afo) * sym.exp(-sym.Float(Eaf) / (sym.Float(R) * T_sym))
    kr_sym = sym.Float(Aro) * sym.exp(-sym.Float(Ear) / (sym.Float(R) * T_sym))

    reaction_sym = (
        Cao_sym
        - Ca_sym
        - kf_sym * Ca_sym * Cb_sym**2 * sym.Float(tau)
        + kr_sym * Cc_sym * sym.Float(tau)
    )

    value_fun = sym.lambdify(variables, reaction_sym, "numpy")
    gradient_funs = tuple(
        sym.lambdify(variables, sym.diff(reaction_sym, variable), "numpy")
        for variable in variables
    )
    return ConstraintFunctions(value=value_fun, gradient=gradient_funs)


def evaluate_scalar_function(fun: callable, z: np.ndarray) -> np.ndarray:
    """Evaluate a scalar SymPy-lambdified function on one or many points."""

    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        return np.asarray(fun(*z.tolist()), dtype=float).reshape(())
    values = fun(*[z[:, column] for column in range(z.shape[1])])
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        values = np.full(z.shape[0], float(values), dtype=float)
    return values.reshape(-1)


def linearize_reaction(
    anchor_z: np.ndarray, functions: ConstraintFunctions
) -> tuple[np.ndarray, float]:
    """Return a and b for the affine equation a^T z = b.

    The Taylor model is
        ell(z) = c(z0) + grad c(z0)^T (z-z0)
               = a^T z - b.
    Therefore b = a^T z0 - c(z0).
    """

    anchor_z = np.asarray(anchor_z, dtype=float)
    gradient = np.array(
        [float(evaluate_scalar_function(fun, anchor_z)) for fun in functions.gradient],
        dtype=float,
    )
    c0 = float(evaluate_scalar_function(functions.value, anchor_z))
    b_reaction = float(gradient @ anchor_z - c0)
    return gradient, b_reaction


def reaction_constraint(z: np.ndarray, functions: ConstraintFunctions) -> np.ndarray:
    return evaluate_scalar_function(functions.value, z)


def mass_balance_constraint(z: np.ndarray) -> np.ndarray:
    """Known exact affine constraint used by the repository.

    -Cao + Ca + Cb + Cc - (Cbo + Cco) = 0.
    """

    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        return np.asarray(-z[1] + z[2] + z[3] + z[4] - (Cbo + Cco))
    return -z[:, 1] + z[:, 2] + z[:, 3] + z[:, 4] - (Cbo + Cco)


# -----------------------------------------------------------------------------
# Region scoring
# -----------------------------------------------------------------------------


def select_anchor_index(
    X: np.ndarray,
    indices: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> int:
    """Choose the observed point nearest the geometric box center.

    Distances are normalized by the box width so variables with different units
    are comparable. If the exact center exists in the data, it is selected.
    """

    center = 0.5 * (low + high)
    width = np.maximum(high - low, np.finfo(float).eps)
    local_X = X[indices]
    distances = np.sum(((local_X - center) / width) ** 2, axis=1)
    return int(indices[int(np.argmin(distances))])


def affine_reaction_value(z: np.ndarray, gradient: np.ndarray, b: float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        return np.asarray(float(gradient @ z - b))
    return z @ gradient - b


def project_outputs_to_local_affine_constraints(
    X: np.ndarray,
    Y: np.ndarray,
    reaction_gradient: np.ndarray,
    b_reaction: float,
) -> np.ndarray:
    """Apply the same closed-form output projection used in NNOPT.

    Constraints are written as A x + B y = b. The first row is the local
    reaction Taylor model and the second row is the exact mass balance.
    """

    input_dim = X.shape[1]
    mass_balance_A = np.zeros(input_dim, dtype=float)
    # In this repository Cao is the second input. For another application,
    # replace this row by the input coefficients of the known affine constraint.
    mass_balance_A[1] = -1.0
    A = np.vstack([reaction_gradient[:input_dim], mass_balance_A])
    B = np.vstack(
        [
            reaction_gradient[input_dim:],
            np.array([1.0, 1.0, 1.0], dtype=float),
        ]
    )
    b = np.array([b_reaction, Cbo + Cco], dtype=float)

    # y_projected = y - B^T (B B^T)^(-1) (A x + B y - b)
    residual = X @ A.T + Y @ B.T - b[None, :]
    normal_map = B.T @ np.linalg.pinv(B @ B.T)
    return Y - residual @ normal_map.T


def statistic(values: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("inf")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("error_quantile must be in (0, 1].")
    if quantile == 1.0:
        return float(np.max(values))
    return float(np.quantile(values, quantile))


def build_region_model(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    indices: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    depth: int,
    functions: ConstraintFunctions,
    error_quantile: float,
    criterion: str,
    constraint_scale: float,
) -> RegionModel:
    """Construct one local model and evaluate its offline error indicators."""

    if indices.size == 0:
        raise ValueError("Cannot build a region with zero data points.")
    if constraint_scale <= 0.0:
        raise ValueError("constraint_scale must be positive.")

    anchor_index = select_anchor_index(X, indices, low, high)
    anchor_z = Z[anchor_index].copy()
    gradient, b_reaction = linearize_reaction(anchor_z, functions)

    local_Z = Z[indices]
    nonlinear_values = reaction_constraint(local_Z, functions)
    affine_values = affine_reaction_value(local_Z, gradient, b_reaction)

    # This isolates nonlinear Taylor error even when measured data do not
    # satisfy the known constraint exactly.
    remainder_values = np.abs(nonlinear_values - affine_values) / constraint_scale

    # Projection-aware offline check: use each available measured output as a
    # pre-projection point, project it with the local affine constraints, and
    # evaluate the original known nonlinear constraint at the projected point.
    projected_Y = project_outputs_to_local_affine_constraints(
        X[indices], Y[indices], gradient, b_reaction
    )
    projected_Z = np.column_stack([X[indices], projected_Y])
    projected_reaction = np.abs(reaction_constraint(projected_Z, functions))
    projected_mass = np.abs(mass_balance_constraint(projected_Z))
    projected_residual_values = (
        np.maximum(projected_reaction, projected_mass) / constraint_scale
    )

    remainder_score = statistic(remainder_values, error_quantile)
    projected_score = statistic(projected_residual_values, error_quantile)

    if criterion == "remainder":
        criterion_score = remainder_score
    elif criterion == "projection":
        criterion_score = projected_score
    elif criterion == "hybrid":
        criterion_score = max(remainder_score, projected_score)
    else:
        raise ValueError(f"Unsupported criterion: {criterion}")

    return RegionModel(
        low=low.copy(),
        high=high.copy(),
        indices=indices.copy(),
        depth=depth,
        anchor_index=anchor_index,
        anchor_z=anchor_z,
        gradient=gradient,
        b_reaction=b_reaction,
        remainder_values=remainder_values,
        projected_residual_values=projected_residual_values,
        remainder_score=remainder_score,
        projected_score=projected_score,
        criterion_score=criterion_score,
    )


# -----------------------------------------------------------------------------
# Adaptive binary hyper-rectangle partition
# -----------------------------------------------------------------------------


def candidate_thresholds(
    values: np.ndarray,
    low: float,
    high: float,
    split_mode: str,
    split_quantiles: Sequence[float],
) -> list[float]:
    """Generate candidate cut locations for one input dimension."""

    if split_mode == "midpoint":
        candidates = [0.5 * (low + high)]
    elif split_mode == "quantile":
        candidates = [float(np.quantile(values, q)) for q in split_quantiles]
    else:
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    tolerance = 1e-12 * max(1.0, abs(low), abs(high))
    candidates = sorted(
        {
            value
            for value in candidates
            if low + tolerance < value < high - tolerance
        }
    )
    return candidates


def best_split_for_region(
    region: RegionModel,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    functions: ConstraintFunctions,
    error_quantile: float,
    criterion: str,
    constraint_scale: float,
    min_samples_leaf: int,
    split_mode: str,
    split_quantiles: Sequence[float],
) -> tuple[RegionModel, RegionModel, int, float, float] | None:
    """Return the minimax candidate split for a region.

    Candidate score = max(left error, right error). The dimension and threshold
    giving the smallest candidate score are selected.
    """

    best = None
    local_indices = region.indices

    for dimension in range(X.shape[1]):
        local_values = X[local_indices, dimension]
        thresholds = candidate_thresholds(
            local_values,
            float(region.low[dimension]),
            float(region.high[dimension]),
            split_mode,
            split_quantiles,
        )

        for threshold in thresholds:
            left_indices = local_indices[local_values < threshold]
            right_indices = local_indices[local_values >= threshold]
            if (
                left_indices.size < min_samples_leaf
                or right_indices.size < min_samples_leaf
            ):
                continue

            left_low = region.low.copy()
            left_high = region.high.copy()
            left_high[dimension] = threshold

            right_low = region.low.copy()
            right_low[dimension] = threshold
            right_high = region.high.copy()

            left_region = build_region_model(
                X,
                Y,
                Z,
                left_indices,
                left_low,
                left_high,
                region.depth + 1,
                functions,
                error_quantile,
                criterion,
                constraint_scale,
            )
            right_region = build_region_model(
                X,
                Y,
                Z,
                right_indices,
                right_low,
                right_high,
                region.depth + 1,
                functions,
                error_quantile,
                criterion,
                constraint_scale,
            )

            after_score = max(
                left_region.criterion_score, right_region.criterion_score
            )
            balance = abs(left_indices.size - right_indices.size)

            candidate = (
                after_score,
                balance,
                dimension,
                threshold,
                left_region,
                right_region,
            )
            if best is None or candidate[:2] < best[:2]:
                best = candidate

    if best is None:
        return None

    after_score, _, dimension, threshold, left_region, right_region = best
    return left_region, right_region, dimension, threshold, after_score


def adaptive_partition(
    X: np.ndarray,
    Y: np.ndarray,
    functions: ConstraintFunctions,
    epsilon: float,
    max_regions: int,
    max_depth: int,
    min_samples_leaf: int,
    min_relative_improvement: float,
    error_quantile: float,
    criterion: str,
    constraint_scale: float,
    split_mode: str,
    split_quantiles: Sequence[float],
) -> list[RegionModel]:
    """Build the offline adaptive hyper-rectangle partition."""

    if epsilon < 0.0:
        raise ValueError("epsilon must be nonnegative.")
    if max_regions < 1:
        raise ValueError("max_regions must be at least 1.")
    if min_samples_leaf < 2:
        raise ValueError("min_samples_leaf must be at least 2.")

    Z = np.column_stack([X, Y])
    root_low = np.min(X, axis=0)
    root_high = np.max(X, axis=0)
    root = build_region_model(
        X,
        Y,
        Z,
        np.arange(X.shape[0], dtype=int),
        root_low,
        root_high,
        depth=0,
        functions=functions,
        error_quantile=error_quantile,
        criterion=criterion,
        constraint_scale=constraint_scale,
    )

    leaves: list[RegionModel] = [root]

    while len(leaves) < max_regions:
        candidate_positions = [
            index
            for index, region in enumerate(leaves)
            if not region.stop_reason
            and region.criterion_score > epsilon
            and region.depth < max_depth
            and region.indices.size >= 2 * min_samples_leaf
        ]
        if not candidate_positions:
            break

        # Refine the currently worst region first.
        parent_position = max(
            candidate_positions,
            key=lambda index: leaves[index].criterion_score,
        )
        parent = leaves[parent_position]
        split = best_split_for_region(
            parent,
            X,
            Y,
            Z,
            functions,
            error_quantile,
            criterion,
            constraint_scale,
            min_samples_leaf,
            split_mode,
            split_quantiles,
        )

        if split is None:
            parent.stop_reason = "no_valid_split"
            continue

        left, right, dimension, threshold, after_score = split
        parent_score = parent.criterion_score
        absolute_improvement = parent_score - after_score
        relative_improvement = absolute_improvement / max(parent_score, 1e-15)

        if relative_improvement < min_relative_improvement:
            parent.stop_reason = "insufficient_improvement"
            continue

        print(
            "Split depth={} samples={} error={:.6e} along {} at {:.8g}; "
            "child max error={:.6e}".format(
                parent.depth,
                parent.indices.size,
                parent_score,
                INPUT_SHORT_NAMES[dimension],
                threshold,
                after_score,
            )
        )

        leaves.pop(parent_position)
        leaves.extend([left, right])

    for region in leaves:
        if region.criterion_score <= epsilon:
            region.stop_reason = "tolerance_met"
        elif region.depth >= max_depth and not region.stop_reason:
            region.stop_reason = "max_depth"
        elif region.indices.size < 2 * min_samples_leaf and not region.stop_reason:
            region.stop_reason = "insufficient_data"
        elif not region.stop_reason:
            region.stop_reason = "max_regions"

    # Stable ordering must match ABb_matrices.csv and region_bounds.npz.
    leaves.sort(key=lambda item: tuple(item.low.tolist()) + tuple(item.high.tolist()))
    return leaves


# -----------------------------------------------------------------------------
# Saving artifacts used by the repository
# -----------------------------------------------------------------------------


def save_partition_artifacts(
    regions: Sequence[RegionModel],
    data_df: pd.DataFrame,
    functions: ConstraintFunctions,
    epsilon: float,
    criterion: str,
    error_quantile: float,
    constraint_scale: float,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    lows = np.vstack([region.low for region in regions])
    highs = np.vstack([region.high for region in regions])
    np.savez(
        output_dir / "region_bounds.npz",
        lows=lows,
        highs=highs,
        global_low=np.min(lows, axis=0),
        global_high=np.max(highs, axis=0),
        input_names=np.asarray(INPUT_COLUMNS, dtype=str),
    )

    region_rows: list[dict] = []
    lin_rows: list[dict] = []
    abb_rows: list[dict] = []

    for region_id, region in enumerate(regions):
        anchor_row = data_df.iloc[region.anchor_index]
        local_df = data_df.iloc[region.indices]

        region_record = {
            "region_id": region_id,
            "depth": region.depth,
            "n_samples": int(region.indices.size),
            "criterion": criterion,
            "criterion_score": region.criterion_score,
            "remainder_score": region.remainder_score,
            "remainder_max": float(np.max(region.remainder_values)),
            "remainder_mean": float(np.mean(region.remainder_values)),
            "projected_score": region.projected_score,
            "projected_max": float(np.max(region.projected_residual_values)),
            "projected_mean": float(np.mean(region.projected_residual_values)),
            "stop_reason": region.stop_reason,
            "anchor_row_index": region.anchor_index,
        }

        for dimension, name in enumerate(INPUT_SHORT_NAMES):
            region_record[f"{name}_low"] = float(region.low[dimension])
            region_record[f"{name}_high"] = float(region.high[dimension])
            region_record[f"{name}_center"] = float(
                0.5 * (region.low[dimension] + region.high[dimension])
            )
            region_record[f"anchor_{name}"] = float(
                anchor_row[INPUT_COLUMNS[dimension]]
            )

        for output_name, column_name in zip(OUTPUT_SHORT_NAMES, OUTPUT_COLUMNS):
            region_record[f"anchor_{output_name}"] = float(anchor_row[column_name])
            region_record[f"data_{output_name}_min"] = float(local_df[column_name].min())
            region_record[f"data_{output_name}_max"] = float(local_df[column_name].max())

        region_rows.append(region_record)

        gradient = region.gradient
        lin_rows.append(
            {
                **region_record,
                "f_anchor": float(
                    reaction_constraint(region.anchor_z, functions)
                ),
                "aT": float(gradient[0]),
                "aCao": float(gradient[1]),
                "aCa": float(gradient[2]),
                "aCb": float(gradient[3]),
                "aCc": float(gradient[4]),
                "b": float(region.b_reaction),
            }
        )

        abb_rows.append(
            {
                "region_id": region_id,
                "constraint_order": 0,
                "constraint_name": "reaction_linearized_adaptive",
                "A_T": float(gradient[0]),
                "A_Cao": float(gradient[1]),
                "B_Ca": float(gradient[2]),
                "B_Cb": float(gradient[3]),
                "B_Cc": float(gradient[4]),
                "b": float(region.b_reaction),
            }
        )
        abb_rows.append(
            {
                "region_id": region_id,
                "constraint_order": 1,
                "constraint_name": "mass_balance_exact",
                "A_T": 0.0,
                "A_Cao": -1.0,
                "B_Ca": 1.0,
                "B_Cb": 1.0,
                "B_Cc": 1.0,
                "b": float(Cbo + Cco),
            }
        )

    region_frame = pd.DataFrame(region_rows).sort_values("region_id")
    lin_frame = pd.DataFrame(lin_rows).sort_values("region_id")
    abb_frame = pd.DataFrame(abb_rows).sort_values(
        ["region_id", "constraint_order"]
    )

    region_frame.to_csv(output_dir / "adaptive_regions.csv", index=False)
    lin_frame.to_csv(output_dir / "lin_params.csv", index=False)
    abb_frame.to_csv(output_dir / "ABb_matrices.csv", index=False)

    summary = {
        "num_regions": len(regions),
        "epsilon": epsilon,
        "criterion": criterion,
        "error_quantile": error_quantile,
        "constraint_scale": constraint_scale,
        "maximum_final_criterion_score": float(
            max(region.criterion_score for region in regions)
        ),
        "maximum_final_remainder_score": float(
            max(region.remainder_score for region in regions)
        ),
        "maximum_final_projected_score": float(
            max(region.projected_score for region in regions)
        ),
        "all_regions_meet_tolerance": bool(
            all(region.criterion_score <= epsilon for region in regions)
        ),
        "stop_reasons": pd.Series(
            [region.stop_reason for region in regions]
        ).value_counts().to_dict(),
    }
    with open(output_dir / "partition_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved {output_dir / 'region_bounds.npz'}")
    print(f"Saved {output_dir / 'adaptive_regions.csv'}")
    print(f"Saved {output_dir / 'lin_params.csv'}")
    print(f"Saved {output_dir / 'ABb_matrices.csv'}")
    print(f"Saved {output_dir / 'partition_summary.json'}")
    print(json.dumps(summary, indent=2))


def parse_quantiles(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one split quantile is required.")
    if any(value <= 0.0 or value >= 1.0 for value in values):
        raise argparse.ArgumentTypeError("Split quantiles must be strictly between 0 and 1.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline adaptive PL construction from finite data and known constraints."
    )
    parser.add_argument("--data_csv", type=str, default="data.csv")
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument(
        "--criterion",
        choices=["projection", "remainder", "hybrid"],
        default="projection",
        help=(
            "projection: nonlinear residual after applying the local affine projection "
            "to available data; remainder: |c-ell| on available data; hybrid: max of both."
        ),
    )
    parser.add_argument(
        "--error_quantile",
        type=float,
        default=1.0,
        help="1.0 uses the maximum; 0.95 gives a noise-robust 95th percentile.",
    )
    parser.add_argument("--constraint_scale", type=float, default=1.0)
    parser.add_argument("--max_regions", type=int, default=64)
    parser.add_argument("--max_depth", type=int, default=12)
    parser.add_argument("--min_samples_leaf", type=int, default=8)
    parser.add_argument("--min_relative_improvement", type=float, default=0.01)
    parser.add_argument(
        "--split_mode",
        choices=["quantile", "midpoint"],
        default="quantile",
    )
    parser.add_argument(
        "--split_quantiles",
        type=parse_quantiles,
        default=(0.25, 0.5, 0.75),
        help="Comma-separated candidate data quantiles, e.g. 0.25,0.5,0.75.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("."))
    args = parser.parse_args()

    data_df = pd.read_csv(args.data_csv)
    missing = [column for column in ALL_COLUMNS if column not in data_df.columns]
    if missing:
        raise ValueError(f"Missing required data columns: {missing}")
    if data_df[ALL_COLUMNS].isnull().any().any():
        raise ValueError("The adaptive linearization data contain NaN values.")

    X = data_df[INPUT_COLUMNS].to_numpy(dtype=float)
    Y = data_df[OUTPUT_COLUMNS].to_numpy(dtype=float)
    functions = build_constraint_functions()

    regions = adaptive_partition(
        X=X,
        Y=Y,
        functions=functions,
        epsilon=args.epsilon,
        max_regions=args.max_regions,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        min_relative_improvement=args.min_relative_improvement,
        error_quantile=args.error_quantile,
        criterion=args.criterion,
        constraint_scale=args.constraint_scale,
        split_mode=args.split_mode,
        split_quantiles=args.split_quantiles,
    )

    save_partition_artifacts(
        regions=regions,
        data_df=data_df,
        functions=functions,
        epsilon=args.epsilon,
        criterion=args.criterion,
        error_quantile=args.error_quantile,
        constraint_scale=args.constraint_scale,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
