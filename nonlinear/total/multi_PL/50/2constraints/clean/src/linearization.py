import numpy as np
import pandas as pd
import sympy as sym

from .config import C_MAX, C_MIN, N_C_REGIONS, T_EDGES
from .generate_data import Afo, Aro, Cbo, Cco, Eaf, Ear, R, solve_equilibrium, tau


def main():
    T_edges = np.asarray(T_EDGES, dtype=float)
    C_edges = np.linspace(C_MIN, C_MAX, N_C_REGIONS + 1)
    np.savez("region_edges.npz", T_edges=T_edges, C_edges=C_edges)

    T, Cao, Ca, Cb = sym.symbols("T Cao Ca Cb", real=True)
    kf = sym.Float(Afo) * sym.exp(-sym.Float(Eaf) / (sym.Float(R) * T))
    kr = sym.Float(Aro) * sym.exp(-sym.Float(Ear) / (sym.Float(R) * T))
    residual = (
        Cao - Ca - kf * Ca * Cb**2 * sym.Float(tau)
        + kr * (Cao - Ca + sym.Float(Cbo) - Cb + sym.Float(Cco)) * sym.Float(tau)
    )
    functions = [
        sym.lambdify((T, Cao, Ca, Cb), expression, "numpy")
        for expression in [
            residual,
            sym.diff(residual, Ca),
            sym.diff(residual, Cb),
            sym.diff(residual, T),
            sym.diff(residual, Cao),
        ]
    ]

    rows = []
    nC = len(C_edges) - 1
    for iT, (T0, T1) in enumerate(zip(T_edges[:-1], T_edges[1:])):
        for iC, (C0, C1) in enumerate(zip(C_edges[:-1], C_edges[1:])):
            region_id = iT * nC + iC
            Tss, Caoss = 0.5 * (T0 + T1), 0.5 * (C0 + C1)
            guess = np.array([Cco, Cbo, Caoss], dtype=np.float32)
            center_solution, ok, message = solve_equilibrium(Tss, Caoss, guess)
            if not ok:
                raise RuntimeError(f"Center solve failed for region {region_id}: {message}")
            _, Cbss, Cass = map(float, center_solution)
            fss, aCa, aCb, aT, aCao = [
                float(function(Tss, Caoss, Cass, Cbss)) for function in functions
            ]
            rows.append({
                "region_id": region_id,
                "Tss": Tss,
                "Caoss": Caoss,
                "Cass": Cass,
                "Cbss": Cbss,
                "fss": fss,
                "aCa": aCa,
                "aCb": aCb,
                "aT": aT,
                "aCao": aCao,
            })

    linearization = pd.DataFrame(rows)
    linearization.to_csv("lin_params.csv", index=False)
    constraints = []
    for _, row in linearization.iterrows():
        rhs = (
            -row["fss"] + row["aCa"] * row["Cass"] + row["aCb"] * row["Cbss"]
            + row["aT"] * row["Tss"] + row["aCao"] * row["Caoss"]
        )
        constraints.extend([
            {
                "region_id": int(row["region_id"]),
                "constraint_order": 0,
                "constraint_name": "reaction_linearized",
                "A_T": row["aT"],
                "A_Cao": row["aCao"],
                "B_Ca": row["aCa"],
                "B_Cb": row["aCb"],
                "B_Cc": 0.0,
                "b": rhs,
            },
            {
                "region_id": int(row["region_id"]),
                "constraint_order": 1,
                "constraint_name": "mass_balance",
                "A_T": 0.0,
                "A_Cao": -1.0,
                "B_Ca": 1.0,
                "B_Cb": 1.0,
                "B_Cc": 1.0,
                "b": float(Cbo + Cco),
            },
        ])
    pd.DataFrame(constraints).sort_values(
        ["region_id", "constraint_order"]
    ).reset_index(drop=True).to_csv("ABb_matrices.csv", index=False)


if __name__ == "__main__":
    main()
