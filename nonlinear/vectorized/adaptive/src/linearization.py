import argparse

import numpy as np
import pandas as pd
import sympy as sym

from .adaptive_partition import build_rectangle_partition, rectangles_to_array
from .generate_data import (
    Afo,
    Aro,
    Cbo,
    Cco,
    Eaf,
    Ear,
    R,
    solve_equilibrium,
    tau,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_regions",
        type=int,
        required=True,
        help="Fixed total number of adaptive rectangles.",
    )
    parser.add_argument(
        "--partition",
        choices=["taylor_rectangles"],
        default="taylor_rectangles",
        help="Full-cell 2D Taylor-error rectangle refinement.",
    )
    parser.add_argument("--reference_T_points", type=int, default=181)
    parser.add_argument("--reference_C_points", type=int, default=81)
    parser.add_argument("--safety_factor", type=float, default=1.10)
    return parser.parse_args()


def solve_rectangle_centers(rectangles, surface):
    """Solve the physical steady state at each arbitrary rectangle center."""

    solutions = []  # solve_equilibrium order: [Cc, Cb, Ca]
    for rid, rect in enumerate(rectangles):
        Tss = rect.T_center
        Caoss = rect.C_center

        # Use the closest already-solved reference-surface point as the first guess.
        i = int(np.argmin(np.abs(surface.T - Tss)))
        j = int(np.argmin(np.abs(surface.Cao - Caoss)))
        ref = surface.state[i, j]  # [T, Cao, Ca, Cb, Cc]
        guesses = [
            np.array([ref[4], ref[3], ref[2]], dtype=float),
            np.array([Cco, Cbo, Caoss], dtype=float),
        ]
        messages = []
        for guess in guesses:
            sol, ok, message = solve_equilibrium(Tss, Caoss, guess)
            if ok and np.all(np.isfinite(sol)):
                solutions.append(np.asarray(sol, dtype=float))
                break
            messages.append(str(message))
        else:
            raise RuntimeError(
                f"Center solve failed for region {rid} at "
                f"T={Tss}, Cao={Caoss}: {messages[-2:]}"
            )
    return np.asarray(solutions, dtype=float)


def main():
    args = parse_args()
    if args.n_regions < 1:
        raise ValueError("n_regions must be positive")

    rectangles, surface = build_rectangle_partition(
        n_regions=args.n_regions,
        reference_T_points=args.reference_T_points,
        reference_C_points=args.reference_C_points,
        safety_factor=args.safety_factor,
    )
    region_bounds = rectangles_to_array(rectangles)

    # Keep the existing artifact filename so experiment2.py does not need to change.
    np.savez(
        "region_edges.npz",
        region_bounds=region_bounds,
        partition=np.asarray(args.partition),
        n_regions=np.asarray(len(rectangles), dtype=int),
        safety_factor=np.asarray(args.safety_factor, dtype=float),
    )
    print("Saved region_edges.npz (now stores arbitrary region_bounds)")
    print(f"Partition: {args.partition}")
    print(f"Regions: {len(rectangles)}")

    partition_rows = []
    for rid, rect in enumerate(rectangles):
        partition_rows.append({
            "region_id": rid,
            "T_low": rect.T_low,
            "T_high": rect.T_high,
            "C_low": rect.C_low,
            "C_high": rect.C_high,
            "T_center": rect.T_center,
            "C_center": rect.C_center,
            "h_T": rect.h_T,
            "h_C": rect.h_C,
            "depth": rect.depth,
            "M_TT": rect.M_TT,
            "M_TC": rect.M_TC,
            "M_CC": rect.M_CC,
            "estimated_cell_taylor_bound": rect.estimated_bound,
        })
    pd.DataFrame(partition_rows).to_csv("region_partition_summary.csv", index=False)
    print("Saved region_partition_summary.csv")

    center_solutions = solve_rectangle_centers(rectangles, surface)

    T_sym, Cao_sym, Ca_sym, Cb_sym, Cc_sym = sym.symbols(
        "T Cao Ca Cb Cc", real=True
    )
    kf_sym = sym.Float(Afo) * sym.exp(-sym.Float(Eaf) / (sym.Float(R) * T_sym))
    kr_sym = sym.Float(Aro) * sym.exp(-sym.Float(Ear) / (sym.Float(R) * T_sym))
    f_sym = (
        Cao_sym
        - Ca_sym
        - kf_sym * Ca_sym * Cb_sym**2 * sym.Float(tau)
        + kr_sym * Cc_sym * sym.Float(tau)
    )

    variables = (T_sym, Cao_sym, Ca_sym, Cb_sym, Cc_sym)
    f_fun = sym.lambdify(variables, f_sym, "numpy")
    derivative_funs = {
        "aT": sym.lambdify(variables, sym.diff(f_sym, T_sym), "numpy"),
        "aCao": sym.lambdify(variables, sym.diff(f_sym, Cao_sym), "numpy"),
        "aCa": sym.lambdify(variables, sym.diff(f_sym, Ca_sym), "numpy"),
        "aCb": sym.lambdify(variables, sym.diff(f_sym, Cb_sym), "numpy"),
        "aCc": sym.lambdify(variables, sym.diff(f_sym, Cc_sym), "numpy"),
    }

    rows = []
    for rid, (rect, sol) in enumerate(zip(rectangles, center_solutions)):
        Tss = rect.T_center
        Caoss = rect.C_center
        Ccss, Cbss, Cass = sol
        values = (
            float(Tss), float(Caoss), float(Cass), float(Cbss), float(Ccss)
        )
        fss = float(f_fun(*values))
        coeff = {
            name: float(fun(*values)) for name, fun in derivative_funs.items()
        }
        b = (
            -fss
            + coeff["aT"] * Tss
            + coeff["aCao"] * Caoss
            + coeff["aCa"] * Cass
            + coeff["aCb"] * Cbss
            + coeff["aCc"] * Ccss
        )
        rows.append({
            "region_id": rid,
            "T_low": rect.T_low,
            "T_high": rect.T_high,
            "C_low": rect.C_low,
            "C_high": rect.C_high,
            "Tss": Tss,
            "Caoss": Caoss,
            "Cass": Cass,
            "Cbss": Cbss,
            "Ccss": Ccss,
            "fss": fss,
            **coeff,
            "b": b,
            "h_T": rect.h_T,
            "h_C": rect.h_C,
            "M_TT": rect.M_TT,
            "M_TC": rect.M_TC,
            "M_CC": rect.M_CC,
            "estimated_cell_taylor_bound": rect.estimated_bound,
        })

    lin_df = pd.DataFrame(rows).sort_values("region_id").reset_index(drop=True)
    lin_df.to_csv("lin_params.csv", index=False)
    print("Saved lin_params.csv")

    AB_rows = []
    for _, r in lin_df.iterrows():
        AB_rows.append({
            "region_id": int(r["region_id"]),
            "constraint_order": 0,
            "constraint_name": "reaction_linearized",
            "A_T": float(r["aT"]),
            "A_Cao": float(r["aCao"]),
            "B_Ca": float(r["aCa"]),
            "B_Cb": float(r["aCb"]),
            "B_Cc": float(r["aCc"]),
            "b": float(r["b"]),
        })
        AB_rows.append({
            "region_id": int(r["region_id"]),
            "constraint_order": 1,
            "constraint_name": "mass_balance",
            "A_T": 0.0,
            "A_Cao": -1.0,
            "B_Ca": 1.0,
            "B_Cb": 1.0,
            "B_Cc": 1.0,
            "b": float(Cbo + Cco),
        })

    AB_df = (
        pd.DataFrame(AB_rows)
        .sort_values(["region_id", "constraint_order"])
        .reset_index(drop=True)
    )
    AB_df.to_csv("ABb_matrices.csv", index=False)
    print("Saved ABb_matrices.csv")
    print(f"Regions: {len(lin_df)} (arbitrary adaptive rectangles)")


if __name__ == "__main__":
    main()
