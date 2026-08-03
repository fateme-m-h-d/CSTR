import argparse

import numpy as np
import pandas as pd
import sympy as sym

from .generate_data import Caomin, Caomax, Cbo, Cco, kf_const, kr_const, tau


def get_center_row(frame, center, atol=1e-10):
    mask = np.isclose(frame["Cao"].to_numpy(), center, atol=atol, rtol=0.0)
    rows = frame.loc[mask]
    if len(rows) != 1:
        raise RuntimeError(
            f"Center point not found uniquely for Cao={center}. "
            f"Found {len(rows)} rows."
        )
    return rows.iloc[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nC_regions", type=int, required=True)
    args = parser.parse_args()

    C_edges = np.linspace(Caomin, Caomax, args.nC_regions + 1, dtype=float)
    C_centers = 0.5 * (C_edges[:-1] + C_edges[1:])
    np.savez("region_edges.npz", C_edges=C_edges)

    frame = pd.read_csv("data.csv")
    Cao, Ca, Cb, Cc = sym.symbols("Cao Ca Cb Cc")
    residual = (
        Cao - Ca
        - sym.Float(kf_const) * Ca * Cb**2 * sym.Float(tau)
        + sym.Float(kr_const) * Cc * sym.Float(tau)
    )

    rows = []
    for region_id, center in enumerate(C_centers):
        point = get_center_row(frame, center)
        values = {
            Cao: center,
            Ca: float(point["Ca"]),
            Cb: float(point["Cb"]),
            Cc: float(point["Cc"]),
        }
        rows.append({
            "region_id": region_id,
            "C_low": float(C_edges[region_id]),
            "C_high": float(C_edges[region_id + 1]),
            "Caoss": float(center),
            "Cass": values[Ca],
            "Cbss": values[Cb],
            "Ccss": values[Cc],
            "fss": float(residual.subs(values)),
            "aCao": float(sym.diff(residual, Cao).subs(values)),
            "aCa": float(sym.diff(residual, Ca).subs(values)),
            "aCb": float(sym.diff(residual, Cb).subs(values)),
            "aCc": float(sym.diff(residual, Cc).subs(values)),
        })

    linearization = pd.DataFrame(rows).sort_values("region_id").reset_index(drop=True)
    linearization.to_csv("lin_params.csv", index=False)

    constraints = []
    for _, row in linearization.iterrows():
        reaction_rhs = (
            -row["fss"]
            + row["aCao"] * row["Caoss"]
            + row["aCa"] * row["Cass"]
            + row["aCb"] * row["Cbss"]
            + row["aCc"] * row["Ccss"]
        )
        constraints.extend([
            {
                "region_id": int(row["region_id"]),
                "constraint_order": 0,
                "constraint_name": "reaction_linearized",
                "A_Cao": float(row["aCao"]),
                "B_Ca": float(row["aCa"]),
                "B_Cb": float(row["aCb"]),
                "B_Cc": float(row["aCc"]),
                "b": float(reaction_rhs),
            },
            {
                "region_id": int(row["region_id"]),
                "constraint_order": 1,
                "constraint_name": "mass_balance",
                "A_Cao": -1.0,
                "B_Ca": 1.0,
                "B_Cb": 1.0,
                "B_Cc": 1.0,
                "b": float(Cbo + Cco),
            },
        ])

    constraint_frame = (
        pd.DataFrame(constraints)
        .sort_values(["region_id", "constraint_order"])
        .reset_index(drop=True)
    )
    constraint_frame.to_csv("ABb_matrices.csv", index=False)


if __name__ == "__main__":
    main()
