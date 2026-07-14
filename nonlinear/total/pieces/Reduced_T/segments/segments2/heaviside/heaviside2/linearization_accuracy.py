# linearization_accuracy.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler

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


def nonlinear_gradients(T, Ca, Cb):
    kf = Afo * np.exp(-Eaf / (R * T))
    kr = Aro * np.exp(-Ear / (R * T))

    df_dCa = -1.0 - kf * (Cb ** 2) * tau - kr * tau
    df_dCb = -2.0 * kf * Ca * Cb * tau - kr * tau

    return df_dCa, df_dCb


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
    ABb_df = pd.read_csv("ABb_matrices.csv").sort_values("region_id").reset_index(drop=True)

    # Same scaling convention as utils.py / projection layer
    XY_raw = df[["Temperature (T)", "Ca", "Cb", "Cc"]].to_numpy()

    scaler = MaxAbsScaler()
    scaler.fit(XY_raw)
    scaler.scale_[0] = max(scaler.scale_[0], 800)

    scale = scaler.scale_
    x_scale = scale[0]
    y_scale = scale[1:4]

    detailed_rows = []
    summary_rows = []

    for _, row in lin_df.iterrows():
        region_id = int(row["region_id"])
        T_low = row["T_low"]
        T_high = row["T_high"]

        # Last segment includes the right boundary
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
                "mean_abs_y_PL_error_scaled": np.nan,
                "rmse_y_PL_error_scaled": np.nan,
                "max_abs_y_PL_error_scaled": np.nan,
                "mean_abs_y_PL_error_original": np.nan,
                "rmse_y_PL_error_original": np.nan,
                "max_abs_y_PL_error_original": np.nan,
                "mean_abs_f_PL": np.nan,
                "max_abs_f_PL": np.nan,
                "mean_amplification": np.nan,
                "max_amplification": np.nan,
            })
            continue

        T = df_region["Temperature (T)"].to_numpy()
        Ca = df_region["Ca"].to_numpy()
        Cb = df_region["Cb"].to_numpy()
        Cc = df_region["Cc"].to_numpy()

        # ------------------------------------------------------------
        # Residual-based linearization accuracy
        # ------------------------------------------------------------
        f_nl = nonlinear_residual(T, Ca, Cb)
        f_lin = linearized_residual(T, Ca, Cb, row)

        linearization_error = f_lin - f_nl
        abs_linearization_error = np.abs(linearization_error)

        # ------------------------------------------------------------
        # Manual PL-projected y calculation
        #
        # In scaled variables:
        # y_PL = y - B^T (B B^T)^(-1) (A x + B y - b)
        #
        # This should match the projection layer.
        # ------------------------------------------------------------
        ABb_row = ABb_df.loc[ABb_df["region_id"] == region_id].iloc[0]

        A = np.array([[ABb_row["A_T"]]], dtype=float)
        B = np.array([[ABb_row["B_Ca"], ABb_row["B_Cb"], ABb_row["B_Cc"]]], dtype=float)
        b = np.array([ABb_row["b"]], dtype=float)

        # Scale A and B exactly like get_scaledABb_list/get_scaledABb in utils.py
        A_scaled = A * x_scale
        B_scaled = B * y_scale

        X_scaled = (T / x_scale).reshape(-1, 1)
        Y_scaled = np.column_stack([
            Ca / y_scale[0],
            Cb / y_scale[1],
            Cc / y_scale[2],
        ])

        chunk = B_scaled.T @ np.linalg.inv(B_scaled @ B_scaled.T)

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

        y_error_scaled = Y_PL_scaled - Y_scaled
        abs_y_error_scaled = np.abs(y_error_scaled)

        y_error_original = Y_PL_original - np.column_stack([Ca, Cb, Cc])
        abs_y_error_original = np.abs(y_error_original)

        mean_abs_y_error_scaled = np.mean(abs_y_error_scaled)
        rmse_y_error_scaled = np.sqrt(np.mean(y_error_scaled ** 2))
        max_abs_y_error_scaled = np.max(abs_y_error_scaled)

        mean_abs_y_error_original = np.mean(abs_y_error_original)
        rmse_y_error_original = np.sqrt(np.mean(y_error_original ** 2))
        max_abs_y_error_original = np.max(abs_y_error_original)

        # ------------------------------------------------------------
        # Nonlinear sensitivity diagnostic
        # ------------------------------------------------------------
        f_true = nonlinear_residual(T, Ca, Cb)
        f_PL = nonlinear_residual(T, Ca_PL, Cb_PL)

        dCa = Ca_PL - Ca
        dCb = Cb_PL - Cb

        df_dCa, df_dCb = nonlinear_gradients(T, Ca, Cb)

        # First-order Taylor estimate of the nonlinear residual change
        taylor_residual_change = df_dCa * dCa + df_dCb * dCb

        dy_norm = np.sqrt(dCa ** 2 + dCb ** 2)
        amplification = np.abs(f_PL - f_true) / (dy_norm + 1e-30)

        # ------------------------------------------------------------
        # Store detailed pointwise results
        # ------------------------------------------------------------
        df_region["region_id"] = region_id

        df_region["residual_nonlinear"] = f_nl
        df_region["residual_linearized"] = f_lin
        df_region["linearization_error"] = linearization_error
        df_region["abs_linearization_error"] = abs_linearization_error

        df_region["residual_PL_scaled_before_projection"] = residual_PL_scaled[:, 0]

        df_region["Ca_PL"] = Ca_PL
        df_region["Cb_PL"] = Cb_PL
        df_region["Cc_PL"] = Cc_PL

        df_region["dCa_PL"] = dCa
        df_region["dCb_PL"] = dCb
        df_region["dCc_PL"] = Cc_PL - Cc

        df_region["abs_Ca_PL_error"] = abs_y_error_original[:, 0]
        df_region["abs_Cb_PL_error"] = abs_y_error_original[:, 1]
        df_region["abs_Cc_PL_error"] = abs_y_error_original[:, 2]

        df_region["abs_Ca_PL_error_scaled"] = abs_y_error_scaled[:, 0]
        df_region["abs_Cb_PL_error_scaled"] = abs_y_error_scaled[:, 1]
        df_region["abs_Cc_PL_error_scaled"] = abs_y_error_scaled[:, 2]

        df_region["f_true"] = f_true
        df_region["f_PL"] = f_PL
        df_region["abs_f_true"] = np.abs(f_true)
        df_region["abs_f_PL"] = np.abs(f_PL)

        df_region["df_dCa"] = df_dCa
        df_region["df_dCb"] = df_dCb
        df_region["abs_df_dCa"] = np.abs(df_dCa)
        df_region["abs_df_dCb"] = np.abs(df_dCb)

        df_region["taylor_residual_change"] = taylor_residual_change
        df_region["abs_taylor_residual_change"] = np.abs(taylor_residual_change)

        df_region["true_residual_change"] = f_PL - f_true
        df_region["abs_true_residual_change"] = np.abs(f_PL - f_true)

        df_region["dy_norm_Ca_Cb"] = dy_norm
        df_region["amplification"] = amplification

        detailed_rows.append(df_region)

        summary_rows.append({
            "region_id": region_id,
            "T_low": T_low,
            "T_high": T_high,
            "n_points": len(df_region),

            "mean_abs_linearization_error": np.mean(abs_linearization_error),
            "rmse_linearization_error": np.sqrt(np.mean(linearization_error ** 2)),
            "max_abs_linearization_error": np.max(abs_linearization_error),

            "mean_abs_nonlinear_residual": np.mean(np.abs(f_nl)),
            "mean_abs_linearized_residual": np.mean(np.abs(f_lin)),

            "mean_abs_y_PL_error_scaled": mean_abs_y_error_scaled,
            "rmse_y_PL_error_scaled": rmse_y_error_scaled,
            "max_abs_y_PL_error_scaled": max_abs_y_error_scaled,

            "mean_abs_y_PL_error_original": mean_abs_y_error_original,
            "rmse_y_PL_error_original": rmse_y_error_original,
            "max_abs_y_PL_error_original": max_abs_y_error_original,

            "mean_abs_f_PL": np.mean(np.abs(f_PL)),
            "max_abs_f_PL": np.max(np.abs(f_PL)),

            "mean_abs_dCa": np.mean(np.abs(dCa)),
            "mean_abs_dCb": np.mean(np.abs(dCb)),
            "max_abs_dCa": np.max(np.abs(dCa)),
            "max_abs_dCb": np.max(np.abs(dCb)),

            "mean_abs_df_dCa": np.mean(np.abs(df_dCa)),
            "mean_abs_df_dCb": np.mean(np.abs(df_dCb)),
            "max_abs_df_dCa": np.max(np.abs(df_dCa)),
            "max_abs_df_dCb": np.max(np.abs(df_dCb)),

            "mean_amplification": np.mean(amplification),
            "max_amplification": np.max(amplification),
        })

    detailed_df = pd.concat(detailed_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    # ------------------------------------------------------------
    # Overall residual-based linearization accuracy
    # ------------------------------------------------------------
    overall_mean_abs = detailed_df["abs_linearization_error"].mean()
    overall_rmse = np.sqrt(np.mean(detailed_df["linearization_error"] ** 2))
    overall_max_abs = detailed_df["abs_linearization_error"].max()

    # ------------------------------------------------------------
    # Overall PL-y accuracy, scaled
    # ------------------------------------------------------------
    scaled_error_array = detailed_df[
        [
            "abs_Ca_PL_error_scaled",
            "abs_Cb_PL_error_scaled",
            "abs_Cc_PL_error_scaled",
        ]
    ].to_numpy()

    overall_mean_abs_y_PL_scaled = np.mean(scaled_error_array)
    overall_rmse_y_PL_scaled = np.sqrt(np.mean(scaled_error_array ** 2))
    overall_max_abs_y_PL_scaled = np.max(scaled_error_array)

    # ------------------------------------------------------------
    # Overall PL-y accuracy, original units
    # ------------------------------------------------------------
    original_error_array = detailed_df[
        [
            "abs_Ca_PL_error",
            "abs_Cb_PL_error",
            "abs_Cc_PL_error",
        ]
    ].to_numpy()

    overall_mean_abs_y_PL_original = np.mean(original_error_array)
    overall_rmse_y_PL_original = np.sqrt(np.mean(original_error_array ** 2))
    overall_max_abs_y_PL_original = np.max(original_error_array)

    print("\n=== Linearization accuracy by region ===")
    print(summary_df.to_string(index=False))

    print("\n=== Overall linearization accuracy ===")
    print(f"Overall mean absolute linearization error: {overall_mean_abs:.6e}")
    print(f"Overall RMSE linearization error:          {overall_rmse:.6e}")
    print(f"Overall max absolute linearization error: {overall_max_abs:.6e}")

    print("\n=== Manual PL y accuracy ===")
    print(f"Overall mean absolute PL-y error, scaled:   {overall_mean_abs_y_PL_scaled:.6e}")
    print(f"Overall RMSE PL-y error, scaled:            {overall_rmse_y_PL_scaled:.6e}")
    print(f"Overall max absolute PL-y error, scaled:    {overall_max_abs_y_PL_scaled:.6e}")

    print(f"Overall mean absolute PL-y error, original: {overall_mean_abs_y_PL_original:.6e}")
    print(f"Overall RMSE PL-y error, original:          {overall_rmse_y_PL_original:.6e}")
    print(f"Overall max absolute PL-y error, original:  {overall_max_abs_y_PL_original:.6e}")

    print("\n=== Nonlinear sensitivity diagnostic ===")
    print(f"Mean |f(true)|:                 {detailed_df['abs_f_true'].mean():.6e}")
    print(f"Mean |f(PL)|:                   {detailed_df['abs_f_PL'].mean():.6e}")
    print(f"Max  |f(PL)|:                   {detailed_df['abs_f_PL'].max():.6e}")

    print(f"Mean |dCa|:                     {detailed_df['dCa_PL'].abs().mean():.6e}")
    print(f"Mean |dCb|:                     {detailed_df['dCb_PL'].abs().mean():.6e}")
    print(f"Max  |dCa|:                     {detailed_df['dCa_PL'].abs().max():.6e}")
    print(f"Max  |dCb|:                     {detailed_df['dCb_PL'].abs().max():.6e}")

    print(f"Mean |df/dCa|:                  {detailed_df['abs_df_dCa'].mean():.6e}")
    print(f"Mean |df/dCb|:                  {detailed_df['abs_df_dCb'].mean():.6e}")
    print(f"Max  |df/dCa|:                  {detailed_df['abs_df_dCa'].max():.6e}")
    print(f"Max  |df/dCb|:                  {detailed_df['abs_df_dCb'].max():.6e}")

    print(f"Mean amplification |df|/|dy|:   {detailed_df['amplification'].mean():.6e}")
    print(f"Max  amplification |df|/|dy|:   {detailed_df['amplification'].max():.6e}")

    print("\nTop 10 largest nonlinear residuals after PL projection:")
    cols = [
        "Temperature (T)",
        "Ca",
        "Cb",
        "Ca_PL",
        "Cb_PL",
        "abs_Ca_PL_error",
        "abs_Cb_PL_error",
        "abs_df_dCa",
        "abs_df_dCb",
        "abs_f_PL",
        "amplification",
    ]

    print(
        detailed_df.sort_values("abs_f_PL", ascending=False)[cols]
        .head(10)
        .to_string(index=False)
    )

    detailed_df.to_csv("linearization_accuracy_detailed.csv", index=False)
    summary_df.to_csv("linearization_accuracy_summary.csv", index=False)

    print("\nSaved:")
    print("linearization_accuracy_detailed.csv")
    print("linearization_accuracy_summary.csv")


if __name__ == "__main__":
    main()