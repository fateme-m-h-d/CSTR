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

T_EDGES = np.array([280, 300, 340, 360, 400, 420, 440, 460], dtype=np_dtype)
T_CENTERS = 0.5 * (T_EDGES[:-1] + T_EDGES[1:])   # 290, 320, 350, 380, 410, 430, 450

def equations(variables, T):
    Cc, Cb, Ca = variables
    kf = Afo * np.exp(-Eaf/(R*T)) #Arrhenius eqn for forward reaction
    kr = Aro * np.exp(-Ear/(R*T)) #arrhenius eqn for reverse reaction

    eq1 = Cao - Ca + -kf*Ca*(Cb**2)*tau + kr*(Cao-Ca+Cbo-Cb)*tau
    eq2 = Cbo - Cb + -2*kf*Ca*(Cb**2)*tau + 2*kr*(Cao-Ca+Cbo-Cb)*tau
    eq3=Cc-Cao+Ca-Cbo+Cb
    return [eq1, eq2, eq3]

# # Define the range of T values
# n = 200 #number of points #I'm gonna change this one from 1000 points and decrease it to 600.
# T_values = np.linspace(280, 600, n)  # Adjust the range and number of points as needed         #change this for first half of the nonlinear equation that we have.

def build_temperature_points(n_inner_per_region, seed):
    rng = np.random.default_rng(seed)

    # Always include all edges and all exact centers
    pts = set(T_EDGES.tolist()) | set(T_CENTERS.tolist())

    # Optional extra random interior points per region
    for lo, hi in zip(T_EDGES[:-1], T_EDGES[1:]):
        if n_inner_per_region > 0:
            vals = rng.uniform(lo, hi, size=n_inner_per_region)
            pts.update(np.round(vals, 12).tolist())

    return np.array(sorted(pts), dtype=np_dtype)

def solve_dataset(T_values):
    # store results
    rows = []
    # Initial guess for fsolve
    initial_guess = [Cco, Cbo, Cao]

    for T in T_values:
        solution, info, ier, msg = fsolve(
            equations, initial_guess, args=(T,), full_output=True, xtol=1e-11
        )
        if ier == 1:  # ier == 1 indicates successful convergence
            Cc, Cb, Ca = solution
        else:
            print(f"Solver did not converge for T = {T}. Message: {msg}")
        initial_guess = [Cc, Cb, Ca]  # warm start for next T
        rows.append({
            "Temperature (T)": T,
            "Ca": Ca,
            "Cb": Cb,
            "Cc": Cc,
        })

    return pd.DataFrame(rows)

def generate_dataset(n_inner_per_region, seed, out_csv="data.csv"):
    T_values = build_temperature_points(n_inner_per_region=n_inner_per_region, seed=seed)
    df = solve_dataset(T_values)
    # Save to Excel
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} with {len(df)} rows")
    print("Exact centers included:", T_CENTERS.tolist())
    return df


# plt.plot(T_values, Ca_values,'b--',label='Ca')
# plt.plot(T_values, Cb_values,'r--',label='Cb')
# plt.plot(T_values, Cc_values,'g--',label='Cc')
    
# plt.xlabel('Temperature (K)')
# plt.ylabel('Concentration (mol/L)')
# plt.title('Original Data')
# plt.legend()
# plt.grid()
# plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_inner_per_region", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    generate_dataset(
        n_inner_per_region=args.n_inner_per_region,
        seed=args.seed,
        out_csv="data.csv"
    )


if __name__ == "__main__":
    main()


