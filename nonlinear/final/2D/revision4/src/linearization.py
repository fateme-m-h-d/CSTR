import argparse

import numpy as np
import pandas as pd
import sympy as sym

from .generate_data import solve_equilibrium
from .partitioning import geometric_edges_nd


np_dtype = np.float64

# ============================================================
# Physical constants
# ============================================================

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

# Input-domain bounds
Tmin, Tmax = 280.0, 460.0
Caomin, Caomax = 0.8, 1.2


# ============================================================
# Solve the physical system at an arbitrary region center
# ============================================================

def solve_region_center(
    Tss,
    Caoss,
    guess,
):
    """
    Solve the original nonlinear CSTR equations at a new
    linearization center.

    This removes the old requirement that the center must
    already exist in data.csv.
    """

    sol, ok, message = solve_equilibrium(
        float(Tss),
        float(Caoss),
        guess,
    )

    # If continuation from the previous center fails,
    # retry using a physically reasonable generic guess.
    if not ok:

        fallback_guess = np.array(
            [
                Cco,       # Cc
                Cbo,       # Cb
                Caoss,     # Ca
            ],
            dtype=np_dtype,
        )

        sol, ok, message = solve_equilibrium(
            float(Tss),
            float(Caoss),
            fallback_guess,
        )

    if not ok:
        raise RuntimeError(
            "Equilibrium solve failed at linearization center "
            f"T={Tss}, Cao={Caoss}.\n"
            f"Solver message: {message}"
        )

    # solve_equilibrium returns:
    # [Cc, Cb, Ca]
    Ccss, Cbss, Cass = sol

    return (
        float(Cass),
        float(Cbss),
        float(Ccss),
        sol.copy(),
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    # Number of regions in each dimension
    parser.add_argument(
        "--nT_regions",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--nC_regions",
        type=int,
        default=3,
    )

    # --------------------------------------------------------
    # NEW:
    # one hyperparameter per input dimension
    # --------------------------------------------------------

    parser.add_argument(
        "--ratio_T",
        type=float,
        default=1.0,
        help=(
            "Geometric segment ratio for temperature. "
            "1.0 gives the old uniform partition."
        ),
    )

    parser.add_argument(
        "--ratio_C",
        type=float,
        default=1.0,
        help=(
            "Geometric segment ratio for Cao. "
            "1.0 gives the old uniform partition."
        ),
    )

    # Optional direction switches.
    #
    # Default:
    # segments get smaller as the input increases.
    #
    # reverse=True:
    # segments get smaller toward the LOWER end instead.
    parser.add_argument(
        "--reverse_T",
        action="store_true",
    )

    parser.add_argument(
        "--reverse_C",
        action="store_true",
    )

    args = parser.parse_args()

    # ========================================================
    # 1. Build geometric partitions
    # ========================================================

    bounds = [
        (Tmin, Tmax),
        (Caomin, Caomax),
    ]

    region_counts = [
        args.nT_regions,
        args.nC_regions,
    ]

    ratios = [
        args.ratio_T,
        args.ratio_C,
    ]

    shrink_toward = [
    "lower" if args.reverse_T else "upper",
    "lower" if args.reverse_C else "upper",
    ]

    T_edges, C_edges = geometric_edges_nd(
        bounds=bounds,
        n_regions=region_counts,
        ratios=ratios,
        shrink_toward=shrink_toward,
    )

    nT_regions = len(T_edges) - 1
    nC_regions = len(C_edges) - 1

    print("\n=== Partition ===")

    print(
        f"T: {nT_regions} regions, "
        f"ratio_T={args.ratio_T}"
    )

    print(
        f"Cao: {nC_regions} regions, "
        f"ratio_C={args.ratio_C}"
    )

    print("\nT_edges:")
    print(T_edges)

    print("\nC_edges:")
    print(C_edges)

    # Save both the edges and the ratios for reproducibility.
    np.savez(
        "region_edges.npz",
        T_edges=T_edges,
        C_edges=C_edges,
        ratio_T=float(args.ratio_T),
        ratio_C=float(args.ratio_C),
        reverse_T=bool(args.reverse_T),
        reverse_C=bool(args.reverse_C),
    )

    print("\nSaved region_edges.npz")

    # ========================================================
    # 2. Build symbolic nonlinear constraint
    # ========================================================

    T_sym, Cao_sym, Ca_sym, Cb_sym, Cc_sym = sym.symbols(
        "T Cao Ca Cb Cc",
        real=True,
    )

    kf_sym = (
        sym.Float(Afo)
        * sym.exp(
            -sym.Float(Eaf)
            / (sym.Float(R) * T_sym)
        )
    )

    kr_sym = (
        sym.Float(Aro)
        * sym.exp(
            -sym.Float(Ear)
            / (sym.Float(R) * T_sym)
        )
    )

    # Original nonlinear reaction balance
    f_sym = (
        Cao_sym
        - Ca_sym
        - kf_sym
        * Ca_sym
        * (Cb_sym ** 2)
        * sym.Float(tau)
        + kr_sym
        * Cc_sym
        * sym.Float(tau)
    )

    # First derivatives
    df_Ca_sym = sym.diff(f_sym, Ca_sym)
    df_Cb_sym = sym.diff(f_sym, Cb_sym)
    df_Cc_sym = sym.diff(f_sym, Cc_sym)

    df_T_sym = sym.diff(f_sym, T_sym)
    df_Cao_sym = sym.diff(f_sym, Cao_sym)

    # Convert symbolic expressions to numerical functions
    f_fun = sym.lambdify(
        (
            T_sym,
            Cao_sym,
            Ca_sym,
            Cb_sym,
            Cc_sym,
        ),
        f_sym,
        "numpy",
    )

    df_Ca_fun = sym.lambdify(
        (
            T_sym,
            Cao_sym,
            Ca_sym,
            Cb_sym,
            Cc_sym,
        ),
        df_Ca_sym,
        "numpy",
    )

    df_Cb_fun = sym.lambdify(
        (
            T_sym,
            Cao_sym,
            Ca_sym,
            Cb_sym,
            Cc_sym,
        ),
        df_Cb_sym,
        "numpy",
    )

    df_Cc_fun = sym.lambdify(
        (
            T_sym,
            Cao_sym,
            Ca_sym,
            Cb_sym,
            Cc_sym,
        ),
        df_Cc_sym,
        "numpy",
    )

    df_T_fun = sym.lambdify(
        (
            T_sym,
            Cao_sym,
            Ca_sym,
            Cb_sym,
            Cc_sym,
        ),
        df_T_sym,
        "numpy",
    )

    df_Cao_fun = sym.lambdify(
        (
            T_sym,
            Cao_sym,
            Ca_sym,
            Cb_sym,
            Cc_sym,
        ),
        df_Cao_sym,
        "numpy",
    )

    # ========================================================
    # 3. Initial equilibrium solution
    # ========================================================

    # Start the continuation from the center of the full domain.
    Tc = 0.5 * (Tmin + Tmax)
    Cc0 = 0.5 * (Caomin + Caomax)

    initial_guess = np.array(
        [
            Cco,
            Cbo,
            Cc0,
        ],
        dtype=np_dtype,
    )

    center_solution, ok, message = solve_equilibrium(
        Tc,
        Cc0,
        initial_guess,
    )

    if not ok:
        raise RuntimeError(
            "Initial center equilibrium solve failed: "
            f"{message}"
        )

    # Use the previous successful solution as the next
    # initial guess.
    guess = center_solution.copy()

    # ========================================================
    # 4. Linearize at every rectangle center
    # ========================================================

    rows = []

    for iT in range(nT_regions):

        for iC in range(nC_regions):

            rid = (
                iT * nC_regions
                + iC
            )

            T0 = float(T_edges[iT])
            T1 = float(T_edges[iT + 1])

            C0 = float(C_edges[iC])
            C1 = float(C_edges[iC + 1])

            # Center of this rectangle
            Tss = 0.5 * (T0 + T1)
            Caoss = 0.5 * (C0 + C1)

            # ------------------------------------------------
            # NEW:
            # solve the physical system directly here.
            #
            # We no longer search for the center in data.csv.
            # ------------------------------------------------

            Cass, Cbss, Ccss, guess = solve_region_center(
                Tss=Tss,
                Caoss=Caoss,
                guess=guess,
            )

            # Nonlinear residual at the center
            fss = float(
                f_fun(
                    Tss,
                    Caoss,
                    Cass,
                    Cbss,
                    Ccss,
                )
            )

            # Taylor coefficients
            aCa = float(
                df_Ca_fun(
                    Tss,
                    Caoss,
                    Cass,
                    Cbss,
                    Ccss,
                )
            )

            aCb = float(
                df_Cb_fun(
                    Tss,
                    Caoss,
                    Cass,
                    Cbss,
                    Ccss,
                )
            )

            aCc = float(
                df_Cc_fun(
                    Tss,
                    Caoss,
                    Cass,
                    Cbss,
                    Ccss,
                )
            )

            aT = float(
                df_T_fun(
                    Tss,
                    Caoss,
                    Cass,
                    Cbss,
                    Ccss,
                )
            )

            aCao = float(
                df_Cao_fun(
                    Tss,
                    Caoss,
                    Cass,
                    Cbss,
                    Ccss,
                )
            )

            # Taylor equation:
            #
            # 0 =
            # fss
            # + aT   (T   - Tss)
            # + aCao (Cao - Caoss)
            # + aCa  (Ca  - Cass)
            # + aCb  (Cb  - Cbss)
            # + aCc  (Cc  - Ccss)
            #
            # rearranged to:
            #
            # aT*T + aCao*Cao
            # + aCa*Ca + aCb*Cb + aCc*Cc
            # = b

            b = (
                -fss
                + aCa * Cass
                + aCb * Cbss
                + aCc * Ccss
                + aT * Tss
                + aCao * Caoss
            )

            rows.append(
                {
                    "region_id": rid,

                    "iT": iT,
                    "iC": iC,

                    "T_low": T0,
                    "T_high": T1,

                    "C_low": C0,
                    "C_high": C1,

                    "Tss": Tss,
                    "Caoss": Caoss,

                    "Cass": Cass,
                    "Cbss": Cbss,
                    "Ccss": Ccss,

                    "fss": fss,

                    "aCa": aCa,
                    "aCb": aCb,
                    "aCc": aCc,

                    "aT": aT,
                    "aCao": aCao,

                    "b": b,
                }
            )

    lin_df = (
        pd.DataFrame(rows)
        .sort_values("region_id")
        .reset_index(drop=True)
    )

    lin_df.to_csv(
        "lin_params.csv",
        index=False,
    )

    print("Saved lin_params.csv")

    # ========================================================
    # 5. Construct A, B, b matrices
    # ========================================================

    AB_rows = []

    for _, r in lin_df.iterrows():

        # ----------------------------------------------------
        # Constraint 1:
        # linearized nonlinear reaction balance
        # ----------------------------------------------------

        AB_rows.append(
            {
                "region_id": int(r["region_id"]),

                "constraint_order": 0,
                "constraint_name": "reaction_linearized",

                "A_T": float(r["aT"]),
                "A_Cao": float(r["aCao"]),

                "B_Ca": float(r["aCa"]),
                "B_Cb": float(r["aCb"]),
                "B_Cc": float(r["aCc"]),

                "b": float(r["b"]),
            }
        )

        # ----------------------------------------------------
        # Constraint 2:
        # exact total mass balance
        # ----------------------------------------------------

        AB_rows.append(
            {
                "region_id": int(r["region_id"]),

                "constraint_order": 1,
                "constraint_name": "mass_balance",

                "A_T": 0.0,
                "A_Cao": -1.0,

                "B_Ca": 1.0,
                "B_Cb": 1.0,
                "B_Cc": 1.0,

                "b": float(Cbo + Cco),
            }
        )

    AB_df = (
        pd.DataFrame(AB_rows)
        .sort_values(
            [
                "region_id",
                "constraint_order",
            ]
        )
        .reset_index(drop=True)
    )

    AB_df.to_csv(
        "ABb_matrices.csv",
        index=False,
    )

    print("Saved ABb_matrices.csv")

    print(
        f"Regions: {len(lin_df)} "
        f"= {args.nT_regions} x {args.nC_regions}"
    )

    print("Constraints per region: 2")


if __name__ == "__main__":
    main()