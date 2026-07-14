import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sym
from sympy import symbols, Eq, solve
from scipy.optimize import fsolve
import argparse

np_dtype = np.float32

# ---------------- constants ----------------
Cao = 1 #mol/L
Cbo = 2 #mol/L
Cco = 0 #mol/L
V = 10 #L
Q = 1 #L/s
tau = V/Q #s

#Parameters to tuning to obtain "aggressive" non-linearity
Afo = 10e12
Eaf = 90000 #J/mol
Aro = 10e10
Ear = 80000 #J/mol
R = 8.314 #J/mol

XTOL = 1e-11

Tmin, Tmax = 280.0, 380.0

# keep same idea as 2D:
SEGMENT_SCENARIOS = [1, 2, 3, 5, 7, 9, 11, 30, 90]

def equations(variables, T):
    Cc, Cb, Ca = variables
    kf = Afo * np.exp(-Eaf/(R*T)) #Arrhenius eqn for forward reaction
    kr = Aro * np.exp(-Ear/(R*T)) #arrhenius eqn for reverse reaction

    eq1 = Cao - Ca + -kf*Ca*(Cb**2)*tau + kr*(Cao-Ca+Cbo-Cb)*tau
    eq2 = Cbo - Cb + -2*kf*Ca*(Cb**2)*tau + 2*kr*(Cao-Ca+Cbo-Cb)*tau
    eq3=Cc-Cao+Ca-Cbo+Cb
    return [eq1, eq2, eq3]

def solve_equilibrium(T, guess):
    sol, info, ier, mesg = fsolve(
        equations, guess, args=(T,),
        full_output=True, xtol=XTOL
    )
    return sol, (ier == 1), mesg


# ============================================================
# FIXED GLOBAL SAMPLING
# ============================================================
def build_fixed_points(n_total_points, seed):
    rng = np.random.default_rng(seed)

    center_pts = []
    for nT in SEGMENT_SCENARIOS:
        T_edges = np.linspace(Tmin, Tmax, nT + 1)
        T_centers = 0.5 * (T_edges[:-1] + T_edges[1:])
        for Tc in T_centers:
            center_pts.append(Tc)

    center_pts = np.unique(np.round(np.array(center_pts, dtype=float), 12))
    anchors = np.array([Tmin, Tmax], dtype=float)

    pts = np.concatenate([center_pts, anchors])
    pts = np.unique(np.round(pts, 12))

    if len(pts) > n_total_points:
        raise ValueError(
            f"Need at least {len(pts)} points to include all scenario centers, "
            f"but n_total_points={n_total_points}"
        )

    n_random = n_total_points - len(pts)
    if n_random > 0:
        rand_pts = rng.uniform(Tmin, Tmax, size=n_random)
        pts = np.concatenate([pts, rand_pts])
        pts = np.unique(np.round(pts, 12))

        while len(pts) < n_total_points:
            extra = rng.uniform(Tmin, Tmax, size=1)
            pts = np.concatenate([pts, extra])
            pts = np.unique(np.round(pts, 12))

    return np.sort(pts[:n_total_points])


def plot_generated_data(df, save_path="generated_data_outputs_vs_T.png"):
    T_values = df["Temperature (T)"].values
    Ca_values = df["Ca"].values
    Cb_values = df["Cb"].values
    Cc_values = df["Cc"].values

    plt.figure(figsize=(7, 5))

    plt.plot(T_values, Ca_values, "b-o", markersize=3, label="Ca")
    plt.plot(T_values, Cb_values, "r-o", markersize=3, label="Cb")
    plt.plot(T_values, Cc_values, "g-o", markersize=3, label="Cc")

    plt.xlabel("Temperature (K)")
    plt.ylabel("Concentration (mol/L)")
    plt.title("Generated CSTR Data: Outputs vs Input Temperature")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Saved plot to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_total_points", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_csv", type=str, default="data.csv")
    args = parser.parse_args()

    pts = build_fixed_points(args.n_total_points, args.seed)

    Tc = 0.5 * (Tmin + Tmax)
    guess0 = np.array([Cco, Cbo, Cao], dtype=np_dtype)

    sol_center, ok, mesg = solve_equilibrium(Tc, guess0)
    if not ok:
        raise RuntimeError(f"Center solve failed: {mesg}")

    d2 = (pts - Tc) ** 2
    order = np.argsort(d2)
    pts = pts[order]

    rows = []
    fail_rows = []
    guess = sol_center.copy()

    for Tpt in pts:
        sol, ok, mesg = solve_equilibrium(float(Tpt), guess)
        if ok:
            Cc_sol, Cb_sol, Ca_sol = sol
            rows.append({
                "Temperature (T)": float(Tpt),
                "Ca": float(Ca_sol),
                "Cb": float(Cb_sol),
                "Cc": float(Cc_sol),
            })
            guess = sol
        else:
            fail_rows.append((float(Tpt), mesg))

    df = pd.DataFrame(rows).sort_values(["Temperature (T)"]).reset_index(drop=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved fixed dataset with {len(df)} solved points to {args.out_csv}")
    
    plot_generated_data(df)

    if fail_rows:
        fail_df = pd.DataFrame(fail_rows, columns=["T", "message"])
        fail_df.to_csv("failed_points_fixed_data.csv", index=False)
        print(f"Warning: {len(fail_rows)} points failed. Saved failed_points_fixed_data.csv")

if __name__ == "__main__":
    main()