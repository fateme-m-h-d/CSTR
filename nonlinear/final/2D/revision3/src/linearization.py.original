import numpy as np
import pandas as pd
import sympy as sym
import argparse

np_dtype = np.float32

V = 10.0
Q = 1.0
tau = V / Q

Afo = 10e12
Eaf = 90000.0
Aro = 10e10
Ear = 80000.0
R   = 8.314

Cbo = 2.0
Cco = 0.0

Tmin, Tmax     = 280.0, 460.0
Caomin, Caomax = 0.8, 1.2

def get_center_row(df, Tss, Caoss, atol=1e-10):
    mask = (
        np.isclose(df["Temperature (T)"].to_numpy(), Tss, atol=atol) &
        np.isclose(df["Cao"].to_numpy(), Caoss, atol=atol)
    )
    rows = df.loc[mask]
    if len(rows) != 1:
        raise RuntimeError(
            f"Center point not found uniquely in data.csv for "
            f"Tss={Tss}, Caoss={Caoss}. Found {len(rows)} rows."
        )
    return rows.iloc[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nT_regions", type=int, required=True)
    parser.add_argument("--nC_regions", type=int, default=3)
    args = parser.parse_args()

    T_edges = np.linspace(Tmin, Tmax, args.nT_regions + 1, dtype=float)
    C_edges = np.linspace(Caomin, Caomax, args.nC_regions + 1, dtype=float)

    np.savez("region_edges.npz", T_edges=T_edges, C_edges=C_edges)
    print("Saved region_edges.npz")

    nT_regions = len(T_edges) - 1
    nC_regions = len(C_edges) - 1

    rows = []

    T_sym, Cao_sym, Ca_sym, Cb_sym, Cc_sym = sym.symbols("T Cao Ca Cb Cc", real=True)

    kf_sym = sym.Float(Afo) * sym.exp(-sym.Float(Eaf) / (sym.Float(R) * T_sym))
    kr_sym = sym.Float(Aro) * sym.exp(-sym.Float(Ear) / (sym.Float(R) * T_sym))

    f_sym = (
        Cao_sym - Ca_sym
        - kf_sym * Ca_sym * (Cb_sym**2) * sym.Float(tau)
        + kr_sym * (Cc_sym) * sym.Float(tau)
    )

    df_Ca_sym  = sym.diff(f_sym, Ca_sym)
    df_Cb_sym  = sym.diff(f_sym, Cb_sym)
    df_T_sym   = sym.diff(f_sym, T_sym)
    df_Cao_sym = sym.diff(f_sym, Cao_sym)
    df_Cc_sym = sym.diff(f_sym, Cc_sym)

    f_fun      = sym.lambdify((T_sym, Cao_sym, Ca_sym, Cb_sym, Cc_sym), f_sym, "numpy")
    df_Ca_fun  = sym.lambdify((T_sym, Cao_sym, Ca_sym, Cb_sym, Cc_sym), df_Ca_sym, "numpy")
    df_Cb_fun  = sym.lambdify((T_sym, Cao_sym, Ca_sym, Cb_sym, Cc_sym), df_Cb_sym, "numpy")
    df_T_fun   = sym.lambdify((T_sym, Cao_sym, Ca_sym, Cb_sym, Cc_sym), df_T_sym, "numpy")
    df_Cao_fun = sym.lambdify((T_sym, Cao_sym, Ca_sym, Cb_sym, Cc_sym), df_Cao_sym, "numpy")
    df_Cc_fun = sym.lambdify((T_sym, Cao_sym, Ca_sym, Cb_sym, Cc_sym), df_Cc_sym, "numpy")

    data_df = pd.read_csv("data.csv")

    for iT in range(nT_regions):
        for iC in range(nC_regions):
            rid = iT * nC_regions + iC

            T0, T1 = float(T_edges[iT]), float(T_edges[iT + 1])
            C0, C1 = float(C_edges[iC]), float(C_edges[iC + 1])

            Tss   = 0.5 * (T0 + T1)
            Caoss = 0.5 * (C0 + C1)

            row = get_center_row(data_df, Tss, Caoss)
            Cass = float(row["Ca"])
            Cbss = float(row["Cb"])
            Ccss = float(row["Cc"])

            fss  = float(f_fun(Tss, Caoss, Cass, Cbss, Ccss))
            aCa  = float(df_Ca_fun(Tss, Caoss, Cass, Cbss, Ccss))
            aCb  = float(df_Cb_fun(Tss, Caoss, Cass, Cbss, Ccss))
            aT   = float(df_T_fun(Tss, Caoss, Cass, Cbss, Ccss))
            aCao = float(df_Cao_fun(Tss, Caoss, Cass, Cbss, Ccss))
            aCc  = float(df_Cc_fun(Tss, Caoss, Cass, Cbss, Ccss))
            b = (-fss + aCa*Cass + aCb*Cbss + aT*Tss + aCao*Caoss + aCc*Ccss)

            rows.append({
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
    print(f"Constraints per region: 2")

if __name__ == "__main__":
    main()