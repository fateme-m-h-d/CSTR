"""Measure actual sampled linearization error on data.csv for arbitrary cells."""

import numpy as np
import pandas as pd

from src.generate_data import Afo, Aro, Eaf, Ear, R, tau


def nonlinear_residual(T, Cao, Ca, Cb, Cc):
    kf = Afo * np.exp(-Eaf / (R * T))
    kr = Aro * np.exp(-Ear / (R * T))
    return Cao - Ca - kf * Ca * Cb**2 * tau + kr * Cc * tau


def main():
    data = pd.read_csv("data.csv")
    lin = pd.read_csv("lin_params.csv")

    global_T_high = float(lin["T_high"].max())
    global_C_high = float(lin["C_high"].max())

    detailed = []
    summary = []
    for _, row in lin.iterrows():
        # Same half-open convention used by models.get_rectangle_masks.
        is_global_T_max = np.isclose(float(row["T_high"]), global_T_high)
        is_global_C_max = np.isclose(float(row["C_high"]), global_C_high)

        T_mask = (
            (data["Temperature (T)"] >= row["T_low"])
            & (
                data["Temperature (T)"] <= row["T_high"]
                if is_global_T_max
                else data["Temperature (T)"] < row["T_high"]
            )
        )
        C_mask = (
            (data["Cao"] >= row["C_low"])
            & (
                data["Cao"] <= row["C_high"]
                if is_global_C_max
                else data["Cao"] < row["C_high"]
            )
        )
        region = data.loc[T_mask & C_mask].copy()

        base_summary = {
            "region_id": int(row["region_id"]),
            "T_low": float(row["T_low"]),
            "T_high": float(row["T_high"]),
            "C_low": float(row["C_low"]),
            "C_high": float(row["C_high"]),
            "estimated_cell_taylor_bound": float(
                row.get("estimated_cell_taylor_bound", np.nan)
            ),
        }

        if region.empty:
            summary.append({
                **base_summary,
                "n_points": 0,
                "mean_abs_linearization_error": np.nan,
                "max_abs_linearization_error": np.nan,
                "rmse_linearization_error": np.nan,
            })
            continue

        T = region["Temperature (T)"].to_numpy()
        Cao = region["Cao"].to_numpy()
        Ca = region["Ca"].to_numpy()
        Cb = region["Cb"].to_numpy()
        Cc = region["Cc"].to_numpy()
        f_nl = nonlinear_residual(T, Cao, Ca, Cb, Cc)
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
        region["linearization_error"] = error
        region["abs_linearization_error"] = np.abs(error)
        detailed.append(region)
        summary.append({
            **base_summary,
            "n_points": len(region),
            "mean_abs_linearization_error": float(np.mean(np.abs(error))),
            "max_abs_linearization_error": float(np.max(np.abs(error))),
            "rmse_linearization_error": float(np.sqrt(np.mean(error**2))),
        })

    if detailed:
        pd.concat(detailed, ignore_index=True).to_csv(
            "linearization_accuracy_detailed.csv", index=False
        )
    pd.DataFrame(summary).to_csv("linearization_accuracy_summary.csv", index=False)
    print("Saved linearization_accuracy_summary.csv")


if __name__ == "__main__":
    main()
