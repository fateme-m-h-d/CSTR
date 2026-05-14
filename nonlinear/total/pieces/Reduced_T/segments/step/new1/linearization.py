import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sym
from sympy import symbols
import argparse

from generate_data import (
    Cao, Cbo, Cco, tau,
    Afo, Eaf, Aro, Ear, R
)

np_dtype = np.float32

Tmin, Tmax = 280.0, 460.0

def get_center_row(df, Tss, atol=1e-10):
    mask = np.isclose(df["Temperature (T)"].to_numpy(), Tss, atol=atol,  rtol=0.0)
    rows = df.loc[mask]
    if len(rows) != 1:
        raise RuntimeError(
            f"Center point not found uniquely in data.csv for Tss={Tss}. "
            f"Found {len(rows)} rows."
        )
    return rows.iloc[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nT_regions", type=int, required=True)
    args = parser.parse_args()
    
    T_edges = np.linspace(Tmin, Tmax, args.nT_regions + 1, dtype=float)
    T_centers = 0.5 * (T_edges[:-1] + T_edges[1:])
    
    np.savez("region_edges.npz", T_edges=T_edges)
    print("Saved region_edges.npz")
    
    df = pd.read_csv("data.csv")

    # === 2) Build  symbolic nonlinear constraint f(T, Ca, Cb) = 0 and its linearization ===
    Ca, Cb, T = symbols('Ca Cb T')
    kf = Afo * sym.exp(-Eaf/(R*T))
    kr = Aro * sym.exp(-Ear/(R*T))

    f       = (Cao - Ca
            - kf*Ca*Cb**2 * tau
            + kr*(Cao - Ca + Cbo - Cb + Cco) * tau)
    
    # 3) center rows: guaranteed to exist in data.csv
    lin_params = []
    for rid, Tss in enumerate(T_centers):
        row = get_center_row(df, Tss)

        Cass = row["Ca"]
        Cbss = row["Cb"]

        fss     = f.subs({Ca: Cass, Cb: Cbss, T: Tss})
        df_Ca   = sym.diff(f, Ca).subs({Ca: Cass, Cb: Cbss, T: Tss})
        df_Cb   = sym.diff(f, Cb).subs({Ca: Cass, Cb: Cbss, T: Tss})
        df_T    = sym.diff(f, T).subs({Ca: Cass, Cb: Cbss, T: Tss})

        f_lin   = fss \
          + df_Ca*(Ca - Cass) \
          + df_Cb*(Cb - Cbss) \
          + df_T *(T  - Tss)
          
        print("Linearized MB on A is", f_lin)
        
        lin_params.append({
            "region_id": rid,
            "T_low": float(T_edges[rid]),
            "T_high": float(T_edges[rid + 1]),
            "Tss": float(Tss),
            "Cass": Cass,
            "Cbss": Cbss,
            "fss": fss,
            "aT": df_T,
            "aCa": df_Ca,
            "aCb": df_Cb,
        })
    lin_df = pd.DataFrame(lin_params).sort_values("region_id").reset_index(drop=True)
    lin_df.to_csv("lin_params.csv", index=False)
    
    # 4) build A, B, b for 1D input:
    #    aT*T + aCa*Ca + aCb*Cb = b
    AB_rows = []
    for _, r in lin_df.iterrows():
        b_val = (
            -float(r["fss"])
            + float(r["aCa"]) * float(r["Cass"])
            + float(r["aCb"]) * float(r["Cbss"])
            + float(r["aT"]) * float(r["Tss"])
        )

        AB_rows.append({
            "region_id": int(r["region_id"]),
            "A_T": float(r["aT"]),
            "B_Ca": float(r["aCa"]),
            "B_Cb": float(r["aCb"]),
            "B_Cc": 0.0,
            "b": float(b_val),
        })

    AB_df = pd.DataFrame(AB_rows).sort_values("region_id").reset_index(drop=True)
    AB_df.to_csv("ABb_matrices.csv", index=False)

    print("Saved data.csv")
    print("Saved lin_params.csv")
    print("Saved ABb_matrices.csv")
    print(AB_df)


if __name__ == "__main__":
    main()

# # === 3) Lambdify for fast NumPy calls ===
# f_nl_func  = sym.lambdify((Ca, Cb, T), f,     'numpy')
# f_lin_func = sym.lambdify((Ca, Cb, T), f_lin, 'numpy')

# # === 4) Load your original data ===
# df = pd.read_csv('data.csv')
# # make sure your CSV has columns labeled exactly 'T', 'Ca', 'Cb'
# print("Data columns:", df.columns.tolist())

# T_data  = df['Temperature (T)'].values
# Ca_data = df['Ca'].values
# Cb_data = df['Cb'].values

# # === 5) Evaluate both residuals on your data ===
# res_nl  = f_nl_func(Ca_data, Cb_data, T_data)
# res_lin = f_lin_func(Ca_data, Cb_data, T_data)

# df['residual_nonlinear']  = res_nl
# df['residual_linearized'] = res_lin

# # === 6) Plot comparison ===
# plt.figure(figsize=(6,4))
# plt.plot(T_data, res_nl,  'o', label='Nonlinear residual')
# plt.plot(T_data, res_lin, 'x', label='Linearized residual')
# plt.axhline(0, color='gray', lw=0.5)
# plt.xlabel('Temperature T (K)')
# plt.ylabel('Residual f(Ca, Cb, T)')
# plt.title('Residuals at Original Data Points')
# plt.legend()
# plt.tight_layout()
# plt.show()

# # --- 1) Load & filter original data to 500 ≤ T ≤ 600 ---
# df = pd.read_csv('data.csv')          # expects columns ['T','Ca','Cb',…]
# mask = (df['Temperature (T)'] >= 500) & (df['Temperature (T)'] <= 600)
# df_range = df.loc[mask, ['Temperature (T)','Ca','Cb']].copy()

# # --- 2) Extract as NumPy arrays and evaluate both functions ---
# T_vals  = df_range['Temperature (T)'].values
# Ca_vals = df_range['Ca'].values
# Cb_vals = df_range['Cb'].values

# res_nl  = f_nl_func(Ca_vals, Cb_vals, T_vals)
# res_lin = f_lin_func(Ca_vals, Cb_vals, T_vals)

# # --- 3) Attach results to the DataFrame & print ---
# df_range['residual_nonlinear']  = res_nl
# df_range['residual_linearized'] = res_lin

# # Print the inputs and both outputs
# print(df_range.to_string(index=False))

# # (Optional) save to CSV
# df_range.to_csv('residual_comparison_500_600.csv', index=False)
# print("\nWrote residual_comparison_500_600.csv")

# # --- 4) Plot the two curves over T ∈ [500,600] ---
# plt.figure(figsize=(6,4))
# plt.plot(df_range['Temperature (T)'], df_range['residual_nonlinear'],
#          'o', label='Nonlinear residual')
# plt.plot(df_range['Temperature (T)'], df_range['residual_linearized'],
#          'x', label='Linearized residual')
# plt.axhline(0, color='gray', lw=0.5)
# plt.xlabel('Temperature, T (K)')
# plt.ylabel('Residual f(Ca, Cb, T)')
# plt.title('Residuals on Original Data (500–600 K)')
# plt.legend()
# plt.tight_layout()
# plt.show()

