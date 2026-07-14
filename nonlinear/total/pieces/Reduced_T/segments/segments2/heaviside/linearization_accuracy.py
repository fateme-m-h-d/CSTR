# linearization_accuracy.py

import numpy as np
import pandas as pd

from generate_data import (
    Cao, Cbo, Cco, tau,
    Afo, Eaf, Aro, Ear, R
)


def nonlinear_residual(T, Ca, Cb):
    kf = Afo * np.exp(-Eaf / (R * T))
    kr = Aro * np.exp(-Ear / (R * T))

    f_nl = (
        Cao - Ca
        - kf * Ca * (Cb ** 2) * tau
        + kr * (Cao - Ca + Cbo - Cb + Cco) * tau
    )

    return f_nl


def linearized_residual(T, Ca, Cb, row):
    """
    f_lin = fss
          + aCa * (Ca - Cass)
          + aCb * (Cb - Cbss)
          + aT  * (T  - Tss)
    """

    f_lin = (
        row["fss"]
        + row["aCa"] * (Ca - row["Cass"])
        + row["aCb"] * (Cb - row["Cbss"])
        + row["aT"]  * (T  - row["Tss"])
    )

    return f_lin


def main():
    df = pd.read_csv("data.csv")
    lin_df = pd.read_csv("lin_params.csv")

    detailed_rows = []
    summary_rows = []

    for _, row in lin_df.iterrows():
        region_id = int(row["region_id"])
        T_low = row["T_low"]
        T_high = row["T_high"]

        # last segment includes the right boundary
        if region_id < len(lin_df) - 1:
            mask = (df["Temperature (T)"] >= T_low) & (df["Temperature (T)"] < T_high)
        else:
            mask = (df["Temperature (T)"] >= T_low) & (df["Temperature (T)"] <= T_high)

        df_region = df.loc[mask].copy()

        if len(df_region) == 0:
            summary_rows.append({
                "region_id": region_id,
                "T_low": T_low,
                "T_high": T_high,
                "n_points": 0,
                "mean_abs_linearization_error": np.nan,
                "rmse_linearization_error": np.nan,
                "max_abs_linearization_error": np.nan,
                "mean_abs_nonlinear_residual": np.nan,
                "mean_abs_linearized_residual": np.nan,
            })
            continue

        T = df_region["Temperature (T)"].to_numpy()
        Ca = df_region["Ca"].to_numpy()
        Cb = df_region["Cb"].to_numpy()

        f_nl = nonlinear_residual(T, Ca, Cb)
        f_lin = linearized_residual(T, Ca, Cb, row)

        error = f_lin - f_nl
        abs_error = np.abs(error)

        df_region["region_id"] = region_id
        df_region["residual_nonlinear"] = f_nl
        df_region["residual_linearized"] = f_lin
        df_region["linearization_error"] = error
        df_region["abs_linearization_error"] = abs_error

        detailed_rows.append(df_region)

        summary_rows.append({
            "region_id": region_id,
            "T_low": T_low,
            "T_high": T_high,
            "n_points": len(df_region),
            "mean_abs_linearization_error": np.mean(abs_error),
            "rmse_linearization_error": np.sqrt(np.mean(error ** 2)),
            "max_abs_linearization_error": np.max(abs_error),
            "mean_abs_nonlinear_residual": np.mean(np.abs(f_nl)),
            "mean_abs_linearized_residual": np.mean(np.abs(f_lin)),
        })

    detailed_df = pd.concat(detailed_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    overall_mean_abs = detailed_df["abs_linearization_error"].mean()
    overall_rmse = np.sqrt(np.mean(detailed_df["linearization_error"] ** 2))
    overall_max_abs = detailed_df["abs_linearization_error"].max()

    print("\n=== Linearization accuracy by region ===")
    print(summary_df.to_string(index=False))

    print("\n=== Overall linearization accuracy ===")
    print(f"Overall mean absolute linearization error: {overall_mean_abs:.6e}")
    print(f"Overall RMSE linearization error:          {overall_rmse:.6e}")
    print(f"Overall max absolute linearization error: {overall_max_abs:.6e}")

    detailed_df.to_csv("linearization_accuracy_detailed.csv", index=False)
    summary_df.to_csv("linearization_accuracy_summary.csv", index=False)

    print("\nSaved:")
    print("linearization_accuracy_detailed.csv")
    print("linearization_accuracy_summary.csv")


if __name__ == "__main__":
    main()