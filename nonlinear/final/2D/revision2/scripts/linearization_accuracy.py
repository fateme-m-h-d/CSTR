"""Measure actual sampled linearization error on data.csv for every 2D cell."""

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

    detailed = []
    summary = []
    for _, row in lin.iterrows():
        is_last_T = int(row["iT"]) == int(lin["iT"].max())
        is_last_C = int(row["iC"]) == int(lin["iC"].max())
        T_mask = (
            (data["Temperature (T)"] >= row["T_low"])
            & (
                data["Temperature (T)"] <= row["T_high"]
                if is_last_T
                else data["Temperature (T)"] < row["T_high"]
            )
        )
        C_mask = (
            (data["Cao"] >= row["C_low"])
            & (
                data["Cao"] <= row["C_high"]
                if is_last_C
                else data["Cao"] < row["C_high"]
            )
        )
        region = data.loc[T_mask & C_mask].copy()
        if region.empty:
            summary.append({
                "region_id": int(row["region_id"]),
                "iT": int(row["iT"]),
                "iC": int(row["iC"]),
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
            "region_id": int(row["region_id"]),
            "iT": int(row["iT"]),
            "iC": int(row["iC"]),
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
