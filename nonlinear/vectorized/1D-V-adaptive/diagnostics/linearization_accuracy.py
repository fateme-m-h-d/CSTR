import numpy as np
import pandas as pd

from src.generate_data import tau, kf_const, kr_const


def nonlinear_residual(Cao, Ca, Cb, Cc):
    return (
        Cao - Ca
        - kf_const * Ca * (Cb ** 2) * tau
        + kr_const * Cc * tau
    )


def linearized_residual(Cao, Ca, Cb, Cc, row):
    return (
        row["fss"]
        + row["aCao"] * (Cao - row["Caoss"])
        + row["aCa"] * (Ca - row["Cass"])
        + row["aCb"] * (Cb - row["Cbss"])
        + row["aCc"] * (Cc - row["Ccss"])
    )


def main():
    df = pd.read_csv("data.csv")
    lin_df = pd.read_csv("lin_params.csv")

    detailed_rows = []
    summary_rows = []

    for _, row in lin_df.iterrows():
        region_id = int(row["region_id"])
        C_low = row["C_low"]
        C_high = row["C_high"]

        if region_id < len(lin_df) - 1:
            mask = (df["Cao"] >= C_low) & (df["Cao"] < C_high)
        else:
            mask = (df["Cao"] >= C_low) & (df["Cao"] <= C_high)

        df_region = df.loc[mask].copy()
        if len(df_region) == 0:
            summary_rows.append({
                "region_id": region_id,
                "C_low": C_low,
                "C_high": C_high,
                "n_points": 0,
                "mean_abs_linearization_error": np.nan,
                "rmse_linearization_error": np.nan,
                "max_abs_linearization_error": np.nan,
                "mean_abs_nonlinear_residual": np.nan,
                "mean_abs_linearized_residual": np.nan,
            })
            continue

        Cao = df_region["Cao"].to_numpy()
        Ca = df_region["Ca"].to_numpy()
        Cb = df_region["Cb"].to_numpy()
        Cc = df_region["Cc"].to_numpy()

        f_nl = nonlinear_residual(Cao, Ca, Cb, Cc)
        f_lin = linearized_residual(Cao, Ca, Cb, Cc, row)
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
            "C_low": C_low,
            "C_high": C_high,
            "n_points": len(df_region),
            "mean_abs_linearization_error": np.mean(abs_error),
            "rmse_linearization_error": np.sqrt(np.mean(error ** 2)),
            "max_abs_linearization_error": np.max(abs_error),
            "mean_abs_nonlinear_residual": np.mean(np.abs(f_nl)),
            "mean_abs_linearized_residual": np.mean(np.abs(f_lin)),
        })

    detailed_df = pd.concat(detailed_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    detailed_df.to_csv("linearization_accuracy_detailed.csv", index=False)
    summary_df.to_csv("linearization_accuracy_summary.csv", index=False)


if __name__ == "__main__":
    main()
