import numpy as np
import pandas as pd
import sympy as sym
# from scipy.optimize import fsolve
import argparse

np_dtype = np.float32

# ============================================================
# CONSTANTS
# ============================================================
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

# XTOL = 1e-11

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

# ============================================================
# MAIN
# ============================================================
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

    # symbolic derivatives
    T_sym, Cao_sym, Ca_sym, Cb_sym = sym.symbols("T Cao Ca Cb", real=True)

    kf_sym = sym.Float(Afo) * sym.exp(-sym.Float(Eaf) / (sym.Float(R) * T_sym))
    kr_sym = sym.Float(Aro) * sym.exp(-sym.Float(Ear) / (sym.Float(R) * T_sym))

    f_sym = (
        Cao_sym - Ca_sym
        - kf_sym * Ca_sym * (Cb_sym**2) * sym.Float(tau)
        + kr_sym * (Cao_sym - Ca_sym + sym.Float(Cbo) - Cb_sym + sym.Float(Cco)) * sym.Float(tau)
    )

    df_Ca_sym  = sym.diff(f_sym, Ca_sym)
    df_Cb_sym  = sym.diff(f_sym, Cb_sym)
    df_T_sym   = sym.diff(f_sym, T_sym)
    df_Cao_sym = sym.diff(f_sym, Cao_sym)

    f_fun      = sym.lambdify((T_sym, Cao_sym, Ca_sym, Cb_sym), f_sym, "numpy")
    df_Ca_fun  = sym.lambdify((T_sym, Cao_sym, Ca_sym, Cb_sym), df_Ca_sym, "numpy")
    df_Cb_fun  = sym.lambdify((T_sym, Cao_sym, Ca_sym, Cb_sym), df_Cb_sym, "numpy")
    df_T_fun   = sym.lambdify((T_sym, Cao_sym, Ca_sym, Cb_sym), df_T_sym, "numpy")
    df_Cao_fun = sym.lambdify((T_sym, Cao_sym, Ca_sym, Cb_sym), df_Cao_sym, "numpy")
    
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

            fss  = float(f_fun(Tss, Caoss, Cass, Cbss))
            aCa  = float(df_Ca_fun(Tss, Caoss, Cass, Cbss))
            aCb  = float(df_Cb_fun(Tss, Caoss, Cass, Cbss))
            aT   = float(df_T_fun(Tss, Caoss, Cass, Cbss))
            aCao = float(df_Cao_fun(Tss, Caoss, Cass, Cbss))

            b = (-fss + aCa*Cass + aCb*Cbss + aT*Tss + aCao*Caoss)

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
                "fss": fss,
                "aCa": aCa,
                "aCb": aCb,
                "aT": aT,
                "aCao": aCao,
                "b": b,
            })

    lin_df = pd.DataFrame(rows).sort_values("region_id").reset_index(drop=True)
    lin_df.to_csv("lin_params.csv", index=False)
    print("Saved lin_params.csv")

    AB_df = lin_df[["region_id"]].copy()
    AB_df["A_T"]   = lin_df["aT"]
    AB_df["A_Cao"] = lin_df["aCao"]
    AB_df["B_Ca"]  = lin_df["aCa"]
    AB_df["B_Cb"]  = lin_df["aCb"]
    AB_df["B_Cc"]  = 0.0
    AB_df["b"]     = lin_df["b"]

    AB_df.to_csv("ABb_matrices.csv", index=False)
    print("Saved ABb_matrices.csv")
    print(f"Regions: {len(AB_df)} = {args.nT_regions} x {args.nC_regions}")

if __name__ == "__main__":
    main()