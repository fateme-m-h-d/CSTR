# plot_constraint_residual_vs_input.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from generate_data import Cbo, Cco, tau, kf_const, kr_const


def nonlinear_residual(Cao, Ca, Cb):
    return (
        Cao - Ca
        - kf_const * Ca * (Cb ** 2) * tau
        + kr_const * (Cao - Ca + Cbo - Cb + Cco) * tau
    )


def linearized_residual(Cao, Ca, Cb, row):
    return (
        row["fss"]
        + row["aCao"] * (Cao - row["Caoss"])
        + row["aCa"] * (Ca - row["Cass"])
        + row["aCb"] * (Cb - row["Cbss"])
    )


def main():
    df = pd.read_csv("data.csv").sort_values("Cao").reset_index(drop=True)
    lin_df = pd.read_csv("lin_params.csv").sort_values("region_id").reset_index(drop=True)

    Cao_all = df["Cao"].to_numpy()
    Ca_all = df["Ca"].to_numpy()
    Cb_all = df["Cb"].to_numpy()

    f_nl_all = nonlinear_residual(Cao_all, Ca_all, Cb_all)

    f_lin_all = np.zeros_like(f_nl_all)
    region_id_all = np.zeros_like(f_nl_all, dtype=int)

    for _, row in lin_df.iterrows():
        region_id = int(row["region_id"])
        C_low = row["C_low"]
        C_high = row["C_high"]

        if region_id < len(lin_df) - 1:
            mask = (Cao_all >= C_low) & (Cao_all < C_high)
        else:
            mask = (Cao_all >= C_low) & (Cao_all <= C_high)

        f_lin_all[mask] = linearized_residual(
            Cao_all[mask],
            Ca_all[mask],
            Cb_all[mask],
            row,
        )

        region_id_all[mask] = region_id

    lin_error = f_lin_all - f_nl_all
    abs_lin_error = np.abs(lin_error)

    # ------------------------------------------------------------
    # Plot 1: nonlinear residual on true data
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(Cao_all, f_nl_all, marker="o")
    plt.xlabel("Input feed concentration Cao")
    plt.ylabel("Original nonlinear residual f_nl")
    plt.title("Original nonlinear residual evaluated on true data")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("residual_true_data_vs_Cao.png", dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # Plot 2: linearized residual on true data
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(Cao_all, f_lin_all, marker="o")
    for edge in lin_df["C_low"].to_numpy():
        plt.axvline(edge, linestyle="--", linewidth=0.7, alpha=0.3)
    plt.axvline(lin_df["C_high"].iloc[-1], linestyle="--", linewidth=0.7, alpha=0.3)

    plt.xlabel("Input feed concentration Cao")
    plt.ylabel("Linearized residual f_lin")
    plt.title("Linearized residual evaluated on true data")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("linearized_residual_vs_Cao.png", dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # Plot 3: absolute linearization error
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(Cao_all, abs_lin_error, marker="o")
    plt.yscale("log")
    for edge in lin_df["C_low"].to_numpy():
        plt.axvline(edge, linestyle="--", linewidth=0.7, alpha=0.3)
    plt.axvline(lin_df["C_high"].iloc[-1], linestyle="--", linewidth=0.7, alpha=0.3)

    plt.xlabel("Input feed concentration Cao")
    plt.ylabel("|f_lin - f_nl|")
    plt.title("Linearization error versus input concentration")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("linearization_error_vs_Cao.png", dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # Plot 4: nonlinear reaction term
    # ------------------------------------------------------------
    nonlinear_term = Ca_all * (Cb_all ** 2)

    plt.figure(figsize=(8, 5))
    plt.plot(Cao_all, nonlinear_term, marker="o")
    plt.xlabel("Input feed concentration Cao")
    plt.ylabel("Ca * Cb^2")
    plt.title("Nonlinear reaction term versus input concentration")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("nonlinear_reaction_term_vs_Cao.png", dpi=300)
    plt.close()

    print("Saved plots:")
    print("residual_true_data_vs_Cao.png")
    print("linearized_residual_vs_Cao.png")
    print("linearization_error_vs_Cao.png")
    print("nonlinear_reaction_term_vs_Cao.png")

    print("\nSummary:")
    print(f"Mean |f_nl true data|:        {np.mean(np.abs(f_nl_all)):.6e}")
    print(f"Mean |f_lin - f_nl|:          {np.mean(abs_lin_error):.6e}")
    print(f"Max  |f_lin - f_nl|:          {np.max(abs_lin_error):.6e}")
    print(f"Mean |Ca * Cb^2|:             {np.mean(np.abs(nonlinear_term)):.6e}")


if __name__ == "__main__":
    main()