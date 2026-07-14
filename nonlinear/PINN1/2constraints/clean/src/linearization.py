import argparse
import numpy as np
import pandas as pd
import sympy as sym
from sympy import symbols

from .generate_data import Caomin, Caomax, Cbo, Cco, tau, kf_const, kr_const

np_dtype = np.float32


def get_center_row(df, Caoss, atol=1e-10):
    mask = np.isclose(df["Cao"].to_numpy(), Caoss, atol=atol, rtol=0.0)
    rows = df.loc[mask]
    if len(rows) != 1:
        raise RuntimeError(
            f"Center point not found uniquely in data.csv for Caoss={Caoss}. "
            f"Found {len(rows)} rows. Make sure generate_data.py included all scenario centers."
        )
    return rows.iloc[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nC_regions", type=int, required=True)
    args = parser.parse_args()

    C_edges = np.linspace(Caomin, Caomax, args.nC_regions + 1, dtype=float)
    C_centers = 0.5 * (C_edges[:-1] + C_edges[1:])
    np.savez("region_edges.npz", C_edges=C_edges)

    df = pd.read_csv("data.csv")

    Cao, Ca, Cb, Cc = symbols("Cao Ca Cb Cc")

    # Isothermal nonlinear residual: kf and kr are constants now.
    f = (
        Cao - Ca
        - sym.Float(kf_const) * Ca * Cb**2 * sym.Float(tau)
        + sym.Float(kr_const) * Cc * sym.Float(tau)
    )

    lin_params = []
    for rid, Caoss in enumerate(C_centers):
        row = get_center_row(df, Caoss)
        Cass = float(row["Ca"])
        Cbss = float(row["Cb"])
        Ccss = float(row["Cc"])

        subs = {Cao: Caoss, Ca: Cass, Cb: Cbss, Cc: Ccss}
        fss = float(f.subs(subs))
        aCao = float(sym.diff(f, Cao).subs(subs))
        aCa = float(sym.diff(f, Ca).subs(subs))
        aCb = float(sym.diff(f, Cb).subs(subs))
        aCc = float(sym.diff(f, Cc).subs(subs))

        lin_params.append({
            "region_id": rid,
            "C_low": float(C_edges[rid]),
            "C_high": float(C_edges[rid + 1]),
            "Caoss": float(Caoss),
            "Cass": Cass,
            "Cbss": Cbss,
            "Ccss": Ccss,
            "fss": fss,
            "aCao": aCao,
            "aCa": aCa,
            "aCb": aCb,
            "aCc": aCc,
        })

    lin_df = pd.DataFrame(lin_params).sort_values("region_id").reset_index(drop=True)
    lin_df.to_csv("lin_params.csv", index=False)

    # Linear form: aCao*Cao + aCa*Ca + aCb*Cb = b
    # Two constraints per region:
    # 1) linearized reaction constraint
    # 2) exact mass-balance constraint
    AB_rows = []

    for _, r in lin_df.iterrows():
        rid = int(r["region_id"])

        # Constraint 1: reaction linearization
        b_rxn = (
            -float(r["fss"])
            + float(r["aCao"]) * float(r["Caoss"])
            + float(r["aCa"]) * float(r["Cass"])
            + float(r["aCb"]) * float(r["Cbss"])
            + float(r["aCc"]) * float(r["Ccss"])
        )

        AB_rows.append({
            "region_id": rid,
            "constraint": "rxn",
            "A_Cao": float(r["aCao"]),
            "B_Ca": float(r["aCa"]),
            "B_Cb": float(r["aCb"]),
            "B_Cc": float(r["aCc"]),
            "b": float(b_rxn),
        })

        # Constraint 2: exact mass balance
        # Cc - Cao + Ca - Cbo + Cb - Cco = 0
        # -Cao + Ca + Cb + Cc = Cbo + Cco
        AB_rows.append({
            "region_id": rid,
            "constraint": "mb",
            "A_Cao": -1.0,
            "B_Ca": 1.0,
            "B_Cb": 1.0,
            "B_Cc": 1.0,
            "b": float(Cbo + Cco),
        })

    AB_df = pd.DataFrame(AB_rows).sort_values("region_id").reset_index(drop=True)
    AB_df.to_csv("ABb_matrices.csv", index=False)



if __name__ == "__main__":
    main()
