import argparse

import numpy as np
import pandas as pd
import sympy as sym

from .adaptive_partition import build_axiswise_partition
from .generate_data import (
    Afo,
    Aro,
    Cbo,
    Cco,
    Eaf,
    Ear,
    R,
    Caomin,
    Caomax,
    Tmin,
    Tmax,
    solve_equilibrium,
    tau,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nT_regions", type=int, required=True)
    parser.add_argument("--nC_regions", type=int, default=3)
    parser.add_argument(
        "--partition",
        choices=["uniform", "taylor_axiswise"],
        default="uniform",
        help=(
            "uniform: original np.linspace grid; "
            "taylor_axiswise: offline nonuniform tensor grid using sampled "
            "full-Hessian/sensitivity Taylor indicators"
        ),
    )
    parser.add_argument("--reference_T_points", type=int, default=181)
    parser.add_argument("--reference_C_points", type=int, default=81)
    parser.add_argument("--safety_factor", type=float, default=1.10)
    return parser.parse_args()


def solve_centers(T_centers, C_centers):
    """Solve physical steady states at arbitrary tensor-grid midpoints."""

    solved = np.empty((len(T_centers), len(C_centers), 3), dtype=float)
    i0 = int(np.argmin(np.abs(T_centers - 0.5 * (Tmin + Tmax))))
    j0 = int(np.argmin(np.abs(C_centers - 0.5 * (Caomin + Caomax))))

    fallback = np.array([Cco, Cbo, C_centers[j0]], dtype=float)
    sol, ok, message = solve_equilibrium(T_centers[i0], C_centers[j0], fallback)
    if not ok:
        raise RuntimeError(
            f"Center solve failed at T={T_centers[i0]}, Cao={C_centers[j0]}: {message}"
        )
    solved[i0, j0] = sol

    def solve_one(i, j, guesses):
        messages = []
        for guess in guesses:
            sol_, ok_, message_ = solve_equilibrium(
                float(T_centers[i]), float(C_centers[j]), np.asarray(guess, dtype=float)
            )
            if ok_ and np.all(np.isfinite(sol_)):
                solved[i, j] = sol_
                return
            messages.append(str(message_))
        raise RuntimeError(
            f"Center solve failed at T={T_centers[i]}, Cao={C_centers[j]}: "
            f"{messages[-3:]}"
        )

    def fill_row(i):
        for j in range(j0 + 1, len(C_centers)):
            fallback_ = np.array([Cco, Cbo, C_centers[j]], dtype=float)
            solve_one(i, j, [solved[i, j - 1], fallback_])
        for j in range(j0 - 1, -1, -1):
            fallback_ = np.array([Cco, Cbo, C_centers[j]], dtype=float)
            solve_one(i, j, [solved[i, j + 1], fallback_])

    fill_row(i0)
    for i in range(i0 + 1, len(T_centers)):
        fallback_ = np.array([Cco, Cbo, C_centers[j0]], dtype=float)
        solve_one(i, j0, [solved[i - 1, j0], fallback_])
        fill_row(i)
    for i in range(i0 - 1, -1, -1):
        fallback_ = np.array([Cco, Cbo, C_centers[j0]], dtype=float)
        solve_one(i, j0, [solved[i + 1, j0], fallback_])
        fill_row(i)

    return solved


def main():
    args = parse_args()
    if args.nT_regions < 1 or args.nC_regions < 1:
        raise ValueError("numbers of regions must be positive")

    if args.partition == "uniform":
        T_edges = np.linspace(Tmin, Tmax, args.nT_regions + 1, dtype=float)
        C_edges = np.linspace(Caomin, Caomax, args.nC_regions + 1, dtype=float)
        T_bounds = np.full(args.nT_regions, np.nan)
        C_bounds = np.full(args.nC_regions, np.nan)
    else:
        T_edges, C_edges, T_bounds, C_bounds, _surface = build_axiswise_partition(
            n_T_regions=args.nT_regions,
            n_C_regions=args.nC_regions,
            reference_T_points=args.reference_T_points,
            reference_C_points=args.reference_C_points,
            safety_factor=args.safety_factor,
        )

    np.savez(
        "region_edges.npz",
        T_edges=T_edges,
        C_edges=C_edges,
        partition=np.asarray(args.partition),
        estimated_T_axis_bounds=T_bounds,
        estimated_C_axis_bounds=C_bounds,
    )
    print("Saved region_edges.npz")
    print(f"Partition: {args.partition}")
    print(f"T edges: {T_edges}")
    print(f"C edges: {C_edges}")

    axis_rows = []
    for i in range(args.nT_regions):
        axis_rows.append({
            "axis": "T",
            "segment": i,
            "low": T_edges[i],
            "high": T_edges[i + 1],
            "center": 0.5 * (T_edges[i] + T_edges[i + 1]),
            "length": T_edges[i + 1] - T_edges[i],
            "estimated_axis_taylor_bound": T_bounds[i],
        })
    for i in range(args.nC_regions):
        axis_rows.append({
            "axis": "Cao",
            "segment": i,
            "low": C_edges[i],
            "high": C_edges[i + 1],
            "center": 0.5 * (C_edges[i] + C_edges[i + 1]),
            "length": C_edges[i + 1] - C_edges[i],
            "estimated_axis_taylor_bound": C_bounds[i],
        })
    pd.DataFrame(axis_rows).to_csv("axis_partition_summary.csv", index=False)
    print("Saved axis_partition_summary.csv")

    T_centers = 0.5 * (T_edges[:-1] + T_edges[1:])
    C_centers = 0.5 * (C_edges[:-1] + C_edges[1:])
    center_solutions = solve_centers(T_centers, C_centers)

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
    for iT, Tss in enumerate(T_centers):
        for iC, Caoss in enumerate(C_centers):
            rid = iT * args.nC_regions + iC
            Ccss, Cbss, Cass = center_solutions[iT, iC]
            values = (float(Tss), float(Caoss), float(Cass), float(Cbss), float(Ccss))
            fss = float(f_fun(*values))
            coeff = {name: float(fun(*values)) for name, fun in derivative_funs.items()}
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
                "iT": iT,
                "iC": iC,
                "T_low": T_edges[iT],
                "T_high": T_edges[iT + 1],
                "C_low": C_edges[iC],
                "C_high": C_edges[iC + 1],
                "Tss": Tss,
                "Caoss": Caoss,
                "Cass": Cass,
                "Cbss": Cbss,
                "Ccss": Ccss,
                "fss": fss,
                **coeff,
                "b": b,
                "estimated_T_axis_bound": T_bounds[iT],
                "estimated_C_axis_bound": C_bounds[iC],
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
    print(f"Regions: {len(lin_df)} = {args.nT_regions} x {args.nC_regions}")


if __name__ == "__main__":
    main()
