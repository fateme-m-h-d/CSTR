# adaptive_linearization.py

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler

from generate_data import (
    Cao, Cbo, Cco, tau,
    Afo, Eaf, Aro, Ear, R,
    solve_equilibrium,
    Tmin, Tmax
)


def k_values(T):
    kf = Afo * np.exp(-Eaf / (R * T))
    kr = Aro * np.exp(-Ear / (R * T))
    return kf, kr


def nonlinear_residual(T, Ca, Cb):
    kf, kr = k_values(T)

    f = (
        Cao - Ca
        - kf * Ca * (Cb ** 2) * tau
        + kr * (Cao - Ca + Cbo - Cb + Cco) * tau
    )

    return f


def nonlinear_gradients(T, Ca, Cb):
    """
    Gradients of:
    f = Cao - Ca - kf*Ca*Cb^2*tau
        + kr*(Cao - Ca + Cbo - Cb + Cco)*tau

    with respect to Ca, Cb, and T.
    """

    kf, kr = k_values(T)

    dkf_dT = kf * Eaf / (R * T ** 2)
    dkr_dT = kr * Ear / (R * T ** 2)

    mass_term = Cao - Ca + Cbo - Cb + Cco

    df_dCa = -1.0 - kf * (Cb ** 2) * tau - kr * tau
    df_dCb = -2.0 * kf * Ca * Cb * tau - kr * tau
    df_dT = (
        -dkf_dT * Ca * (Cb ** 2) * tau
        + dkr_dT * mass_term * tau
    )

    return df_dT, df_dCa, df_dCb


def solve_center(Tss, df):
    """
    Solve steady-state concentrations at the region center.
    We use the nearest data row only as the initial guess.
    """

    idx = np.argmin(np.abs(df["Temperature (T)"].to_numpy() - Tss))
    nearest = df.iloc[idx]

    guess = np.array(
        [
            nearest["Cc"],
            nearest["Cb"],
            nearest["Ca"],
        ],
        dtype=float,
    )

    sol, ok, mesg = solve_equilibrium(float(Tss), guess)

    if not ok:
        raise RuntimeError(f"fsolve failed at Tss={Tss}: {mesg}")

    Cc_sol, Cb_sol, Ca_sol = sol

    return float(Ca_sol), float(Cb_sol), float(Cc_sol)


def build_linearization_for_interval(lo, hi, df):
    """
    Build original-unit A, B, b for one interval [lo, hi].
    Constraint form:
        A_T*T + B_Ca*Ca + B_Cb*Cb + B_Cc*Cc = b
    """

    Tss = 0.5 * (lo + hi)

    Cass, Cbss, Ccss = solve_center(Tss, df)

    fss = nonlinear_residual(Tss, Cass, Cbss)
    aT, aCa, aCb = nonlinear_gradients(Tss, Cass, Cbss)

    b_val = (
        -fss
        + aCa * Cass
        + aCb * Cbss
        + aT * Tss
    )

    params = {
        "T_low": float(lo),
        "T_high": float(hi),
        "Tss": float(Tss),
        "Cass": float(Cass),
        "Cbss": float(Cbss),
        "Ccss": float(Ccss),
        "fss": float(fss),
        "aT": float(aT),
        "aCa": float(aCa),
        "aCb": float(aCb),
        "b": float(b_val),
    }

    return params


def project_true_y_with_PL(df_region, params, scaler):
    """
    Direct PL projection of true Y.

    This computes the same mathematical operation as the KKT projection layer,
    but directly in NumPy.

    In scaled variables:
        y_PL = y - B^T (B B^T)^(-1) (A x + B y - b)
    """

    if len(df_region) == 0:
        return None

    scale = scaler.scale_
    x_scale = scale[0]
    y_scale = scale[1:4]

    T = df_region["Temperature (T)"].to_numpy(dtype=float)
    Ca = df_region["Ca"].to_numpy(dtype=float)
    Cb = df_region["Cb"].to_numpy(dtype=float)
    Cc = df_region["Cc"].to_numpy(dtype=float)

    A = np.array([[params["aT"]]], dtype=float)
    B = np.array([[params["aCa"], params["aCb"], 0.0]], dtype=float)
    b = np.array([params["b"]], dtype=float)

    # Same scaling convention as utils.py:
    # A_scaled = A * x_scale
    # B_scaled = B * y_scale
    A_scaled = A * x_scale
    B_scaled = B * y_scale.reshape(1, -1)

    X_scaled = (T / x_scale).reshape(-1, 1)
    Y_scaled = np.column_stack(
        [
            Ca / y_scale[0],
            Cb / y_scale[1],
            Cc / y_scale[2],
        ]
    )

    BBt = B_scaled @ B_scaled.T
    chunk = B_scaled.T @ np.linalg.pinv(BBt)

    residual_PL_scaled = (
        X_scaled @ A_scaled.T
        + Y_scaled @ B_scaled.T
        - b.reshape(1, 1)
    )

    Y_PL_scaled = Y_scaled - residual_PL_scaled @ chunk.T
    Y_PL_original = Y_PL_scaled * y_scale.reshape(1, -1)

    Ca_PL = Y_PL_original[:, 0]
    Cb_PL = Y_PL_original[:, 1]
    Cc_PL = Y_PL_original[:, 2]

    f_true = nonlinear_residual(T, Ca, Cb)
    f_PL = nonlinear_residual(T, Ca_PL, Cb_PL)

    y_error_scaled = Y_PL_scaled - Y_scaled
    y_error_original = Y_PL_original - np.column_stack([Ca, Cb, Cc])

    result = {
        "T": T,
        "Ca": Ca,
        "Cb": Cb,
        "Cc": Cc,
        "Ca_PL": Ca_PL,
        "Cb_PL": Cb_PL,
        "Cc_PL": Cc_PL,
        "f_true": f_true,
        "f_PL": f_PL,
        "abs_f_PL": np.abs(f_PL),
        "y_error_scaled": y_error_scaled,
        "y_error_original": y_error_original,
        "abs_y_error_scaled": np.abs(y_error_scaled),
        "abs_y_error_original": np.abs(y_error_original),
        "residual_PL_scaled": residual_PL_scaled[:, 0],
    }

    return result


def get_region_data(df, lo, hi, is_last):
    if is_last:
        mask = (df["Temperature (T)"] >= lo) & (df["Temperature (T)"] <= hi)
    else:
        mask = (df["Temperature (T)"] >= lo) & (df["Temperature (T)"] < hi)

    return df.loc[mask].copy()


def score_interval(df, lo, hi, scaler, split_metric, is_last=False):
    """
    Score one interval. Larger score means this interval should be split first.
    """

    df_region = get_region_data(df, lo, hi, is_last=is_last)

    if len(df_region) == 0:
        return -np.inf, None

    params = build_linearization_for_interval(lo, hi, df)
    result = project_true_y_with_PL(df_region, params, scaler)

    abs_f_PL = result["abs_f_PL"]
    abs_y_scaled = result["abs_y_error_scaled"]

    if split_metric == "mean_abs_f_pl":
        score = float(np.mean(abs_f_PL))

    elif split_metric == "max_abs_f_pl":
        score = float(np.max(abs_f_PL))

    elif split_metric == "rmse_f_pl":
        score = float(np.sqrt(np.mean(result["f_PL"] ** 2)))

    elif split_metric == "mean_abs_y_pl_scaled":
        score = float(np.mean(abs_y_scaled))

    elif split_metric == "max_abs_y_pl_scaled":
        score = float(np.max(abs_y_scaled))

    else:
        raise ValueError(f"Unknown split_metric: {split_metric}")

    info = {
        "score": score,
        "n_points": len(df_region),
        "mean_abs_f_PL": float(np.mean(abs_f_PL)),
        "max_abs_f_PL": float(np.max(abs_f_PL)),
        "rmse_f_PL": float(np.sqrt(np.mean(result["f_PL"] ** 2))),
        "mean_abs_y_PL_scaled": float(np.mean(abs_y_scaled)),
        "max_abs_y_PL_scaled": float(np.max(abs_y_scaled)),
    }

    return score, info


def build_adaptive_edges(df, scaler, n_regions, split_metric, min_width):
    """
    Greedy adaptive segmentation.

    Start with [Tmin, Tmax].
    Repeatedly split the interval with the largest PL error.
    """

    edges = [float(Tmin), float(Tmax)]

    while len(edges) - 1 < n_regions:
        interval_scores = []

        for i in range(len(edges) - 1):
            lo = edges[i]
            hi = edges[i + 1]
            width = hi - lo

            if width <= 2.0 * min_width:
                score = -np.inf
                info = None
            else:
                is_last = i == len(edges) - 2
                score, info = score_interval(
                    df=df,
                    lo=lo,
                    hi=hi,
                    scaler=scaler,
                    split_metric=split_metric,
                    is_last=is_last,
                )

            interval_scores.append((score, i, info))

        interval_scores.sort(reverse=True, key=lambda x: x[0])

        best_score, best_i, best_info = interval_scores[0]

        if not np.isfinite(best_score):
            raise RuntimeError(
                "No splittable interval found. Try reducing n_regions or min_width."
            )

        lo = edges[best_i]
        hi = edges[best_i + 1]
        new_edge = 0.5 * (lo + hi)

        edges.insert(best_i + 1, float(new_edge))

        print(
            f"Added edge {new_edge:.8f} K | "
            f"split [{lo:.4f}, {hi:.4f}] | "
            f"score={best_score:.6e} | "
            f"regions={len(edges)-1}/{n_regions}"
        )

    return np.array(edges, dtype=float)


def save_linearization_files(df, edges, scaler):
    """
    Save:
        region_edges.npz
        lin_params.csv
        ABb_matrices.csv
        adaptive_segmentation_summary.csv
    """

    np.savez("region_edges.npz", T_edges=edges)

    lin_rows = []
    ab_rows = []
    summary_rows = []

    for rid in range(len(edges) - 1):
        lo = edges[rid]
        hi = edges[rid + 1]
        is_last = rid == len(edges) - 2

        params = build_linearization_for_interval(lo, hi, df)

        df_region = get_region_data(df, lo, hi, is_last=is_last)
        result = project_true_y_with_PL(df_region, params, scaler)

        lin_rows.append({
            "region_id": rid,
            "T_low": params["T_low"],
            "T_high": params["T_high"],
            "Tss": params["Tss"],
            "Cass": params["Cass"],
            "Cbss": params["Cbss"],
            "Ccss": params["Ccss"],
            "fss": params["fss"],
            "aT": params["aT"],
            "aCa": params["aCa"],
            "aCb": params["aCb"],
        })

        ab_rows.append({
            "region_id": rid,
            "A_T": params["aT"],
            "B_Ca": params["aCa"],
            "B_Cb": params["aCb"],
            "B_Cc": 0.0,
            "b": params["b"],
        })

        if result is None:
            summary_rows.append({
                "region_id": rid,
                "T_low": lo,
                "T_high": hi,
                "width": hi - lo,
                "n_points": 0,
                "mean_abs_f_PL": np.nan,
                "max_abs_f_PL": np.nan,
                "rmse_f_PL": np.nan,
                "mean_abs_y_PL_scaled": np.nan,
                "max_abs_y_PL_scaled": np.nan,
            })
        else:
            summary_rows.append({
                "region_id": rid,
                "T_low": lo,
                "T_high": hi,
                "width": hi - lo,
                "n_points": len(df_region),
                "mean_abs_f_PL": float(np.mean(result["abs_f_PL"])),
                "max_abs_f_PL": float(np.max(result["abs_f_PL"])),
                "rmse_f_PL": float(np.sqrt(np.mean(result["f_PL"] ** 2))),
                "mean_abs_y_PL_scaled": float(np.mean(result["abs_y_error_scaled"])),
                "max_abs_y_PL_scaled": float(np.max(result["abs_y_error_scaled"])),
            })

    lin_df = pd.DataFrame(lin_rows)
    ab_df = pd.DataFrame(ab_rows)
    summary_df = pd.DataFrame(summary_rows)

    lin_df.to_csv("lin_params.csv", index=False)
    ab_df.to_csv("ABb_matrices.csv", index=False)
    summary_df.to_csv("adaptive_segmentation_summary.csv", index=False)

    print("\nSaved:")
    print("region_edges.npz")
    print("lin_params.csv")
    print("ABb_matrices.csv")
    print("adaptive_segmentation_summary.csv")

    print("\n=== Adaptive segmentation summary ===")
    print(summary_df.to_string(index=False))

    print("\n=== Overall adaptive PL diagnostics ===")
    print(f"Mean |f_PL|:              {summary_df['mean_abs_f_PL'].mean():.6e}")
    print(f"Max  |f_PL|:              {summary_df['max_abs_f_PL'].max():.6e}")
    print(f"Mean PL-y error, scaled:  {summary_df['mean_abs_y_PL_scaled'].mean():.6e}")
    print(f"Max  PL-y error, scaled:  {summary_df['max_abs_y_PL_scaled'].max():.6e}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--nT_regions", type=int, required=True)
    parser.add_argument("--data_csv", type=str, default="data.csv")

    parser.add_argument(
        "--split_metric",
        type=str,
        default="mean_abs_f_pl",
        choices=[
            "mean_abs_f_pl",
            "max_abs_f_pl",
            "rmse_f_pl",
            "mean_abs_y_pl_scaled",
            "max_abs_y_pl_scaled",
        ],
    )

    parser.add_argument(
        "--min_width",
        type=float,
        default=1e-6,
        help="Minimum interval width in Kelvin."
    )

    args = parser.parse_args()

    df = pd.read_csv(args.data_csv).sort_values("Temperature (T)").reset_index(drop=True)

    XY_raw = df[["Temperature (T)", "Ca", "Cb", "Cc"]].to_numpy()

    scaler = MaxAbsScaler()
    scaler.fit(XY_raw)
    scaler.scale_[0] = max(scaler.scale_[0], 800)

    print("\n=== Adaptive segmentation ===")
    print(f"Target number of regions: {args.nT_regions}")
    print(f"Split metric:             {args.split_metric}")
    print(f"Tmin, Tmax:               {Tmin}, {Tmax}")
    print(f"Scaler factors:           {scaler.scale_}")

    edges = build_adaptive_edges(
        df=df,
        scaler=scaler,
        n_regions=args.nT_regions,
        split_metric=args.split_metric,
        min_width=args.min_width,
    )

    save_linearization_files(df, edges, scaler)


if __name__ == "__main__":
    main()