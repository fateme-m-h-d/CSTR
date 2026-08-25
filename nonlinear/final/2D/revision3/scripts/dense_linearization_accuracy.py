import argparse
import numpy as np
import pandas as pd

from src.adaptive_partition import build_reference_surface
from src.generate_data import Afo, Aro, Eaf, Ear, R, tau


def nonlinear_residual(T, Cao, Ca, Cb, Cc):
    kf = Afo * np.exp(-Eaf / (R * T))
    kr = Aro * np.exp(-Ear / (R * T))

    return (
        Cao
        - Ca
        - kf * Ca * Cb**2 * tau
        + kr * Cc * tau
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nT", type=int, default=181)
    parser.add_argument("--nC", type=int, default=81)
    args = parser.parse_args()

    print(
        f"Building dense physical grid: "
        f"{args.nT} x {args.nC} = {args.nT * args.nC} points"
    )

    # Works for both revision2 and revision3.
    # state columns are:
    # [T, Cao, Ca, Cb, Cc]
    surface = build_reference_surface(
        n_T=args.nT,
        n_C=args.nC,
    )

    state = surface.state.reshape(-1, 5)

    data = pd.DataFrame(
        state,
        columns=[
            "Temperature (T)",
            "Cao",
            "Ca",
            "Cb",
            "Cc",
        ],
    )

    lin = (
        pd.read_csv("lin_params.csv")
        .sort_values("region_id")
        .reset_index(drop=True)
    )

    global_T_high = float(lin["T_high"].max())
    global_C_high = float(lin["C_high"].max())

    all_detailed = []
    summary = []

    for _, row in lin.iterrows():

        is_global_T_max = np.isclose(
            float(row["T_high"]),
            global_T_high,
        )

        is_global_C_max = np.isclose(
            float(row["C_high"]),
            global_C_high,
        )

        T_mask = (
            (data["Temperature (T)"] >= row["T_low"])
            &
            (
                (data["Temperature (T)"] <= row["T_high"])
                if is_global_T_max
                else
                (data["Temperature (T)"] < row["T_high"])
            )
        )

        C_mask = (
            (data["Cao"] >= row["C_low"])
            &
            (
                (data["Cao"] <= row["C_high"])
                if is_global_C_max
                else
                (data["Cao"] < row["C_high"])
            )
        )

        region = data.loc[T_mask & C_mask].copy()

        if region.empty:
            summary.append({
                "region_id": int(row["region_id"]),
                "T_low": float(row["T_low"]),
                "T_high": float(row["T_high"]),
                "C_low": float(row["C_low"]),
                "C_high": float(row["C_high"]),
                "n_points": 0,
                "mean_abs_linearization_error": np.nan,
                "rmse_linearization_error": np.nan,
                "max_abs_linearization_error": np.nan,
            })
            continue

        T = region["Temperature (T)"].to_numpy()
        Cao = region["Cao"].to_numpy()
        Ca = region["Ca"].to_numpy()
        Cb = region["Cb"].to_numpy()
        Cc = region["Cc"].to_numpy()

        # Original nonlinear constraint
        f_nl = nonlinear_residual(
            T, Cao, Ca, Cb, Cc
        )

        # Piecewise-linear approximation in this region
        f_lin = (
            row["fss"]
            + row["aT"] * (T - row["Tss"])
            + row["aCao"] * (Cao - row["Caoss"])
            + row["aCa"] * (Ca - row["Cass"])
            + row["aCb"] * (Cb - row["Cbss"])
            + row["aCc"] * (Cc - row["Ccss"])
        )

        error = f_lin - f_nl

        region["region_id"] = int(row["region_id"])
        region["f_nonlinear"] = f_nl
        region["f_linear"] = f_lin
        region["linearization_error"] = error
        region["abs_linearization_error"] = np.abs(error)

        all_detailed.append(region)

        summary.append({
            "region_id": int(row["region_id"]),
            "T_low": float(row["T_low"]),
            "T_high": float(row["T_high"]),
            "C_low": float(row["C_low"]),
            "C_high": float(row["C_high"]),
            "n_points": len(region),
            "mean_abs_linearization_error":
                float(np.mean(np.abs(error))),
            "rmse_linearization_error":
                float(np.sqrt(np.mean(error**2))),
            "max_abs_linearization_error":
                float(np.max(np.abs(error))),
        })

    detailed = pd.concat(
        all_detailed,
        ignore_index=True,
    )

    detailed.to_csv(
        "dense_linearization_accuracy_detailed.csv",
        index=False,
    )

    pd.DataFrame(summary).to_csv(
        "dense_linearization_accuracy_summary.csv",
        index=False,
    )

    error = detailed["linearization_error"].to_numpy()

    overall = pd.DataFrame([{
        "n_grid_points": len(detailed),
        "n_regions": len(lin),
        "overall_MAE":
            np.mean(np.abs(error)),
        "overall_RMSE":
            np.sqrt(np.mean(error**2)),
        "overall_MAX":
            np.max(np.abs(error)),
    }])

    overall.to_csv(
        "dense_linearization_accuracy_overall.csv",
        index=False,
    )

    print("\nDense-grid results")
    print(overall.to_string(index=False))

    print(
        "\nSaved:"
        "\n  dense_linearization_accuracy_detailed.csv"
        "\n  dense_linearization_accuracy_summary.csv"
        "\n  dense_linearization_accuracy_overall.csv"
    )


if __name__ == "__main__":
    main()