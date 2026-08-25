import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import argparse
from .config import N_C_REGIONS as NC_REGIONS, SEGMENT_SCENARIOS

np_dtype = np.float64

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

XTOL = 1e-11

Tmin, Tmax     = 280.0, 460.0
Caomin, Caomax = 0.8, 1.2

def equations(vars_, T, Cao):
    Cc, Cb, Ca = vars_

    kf = Afo * np.exp(-Eaf / (R * T))
    kr = Aro * np.exp(-Ear / (R * T))

    eq1 = Cao - Ca - kf * Ca * (Cb**2) * tau + kr * Cc * tau
    eq2 = Cbo - Cb - 2.0 * kf * Ca * (Cb**2) * tau + 2.0 * kr * Cc * tau
    eq3 = Cc - (Cao - Ca + Cbo - Cb + Cco)

    return [eq1, eq2, eq3]

def solve_equilibrium(T, Cao, guess):
    sol, info, ier, mesg = fsolve(
        equations, guess, args=(T, Cao),
        full_output=True, xtol=XTOL
    )
    return sol, (ier == 1), mesg

def build_fixed_points(n_total_points, seed):
    rng = np.random.default_rng(seed)

    C_edges_fixed = np.linspace(Caomin, Caomax, NC_REGIONS + 1)
    C_centers = 0.5 * (C_edges_fixed[:-1] + C_edges_fixed[1:])

    center_pts = []
    for nT in SEGMENT_SCENARIOS:
        T_edges = np.linspace(Tmin, Tmax, nT + 1)
        T_centers = 0.5 * (T_edges[:-1] + T_edges[1:])
        for Tc in T_centers:
            for Cc in C_centers:
                center_pts.append([Tc, Cc])
                
    
    # Additional centers for the square-grid projection diagnostic:
    # 1x1, 2x2, 4x4, ..., 64x64
    # for n in PROJECTION_GRID_SCENARIOS:
    #     T_edges_diag = np.linspace(Tmin, Tmax, n + 1)
    #     C_edges_diag = np.linspace(Caomin, Caomax, n + 1)

    #     T_centers_diag = 0.5 * (
    #         T_edges_diag[:-1] + T_edges_diag[1:]
    #     )

    #     C_centers_diag = 0.5 * (
    #         C_edges_diag[:-1] + C_edges_diag[1:]
    #     )

    #     for Tc in T_centers_diag:
    #         for Cc in C_centers_diag:
    #             center_pts.append([Tc, Cc])

    center_pts = np.unique(np.round(np.array(center_pts, dtype=float), 12), axis=0)

    anchors = np.array([
        [Tmin, Caomin],
        [Tmin, Caomax],
        [Tmax, Caomin],
        [Tmax, Caomax],
    ], dtype=float)

    pts = np.vstack([center_pts, anchors])
    pts = np.unique(np.round(pts, 12), axis=0)

    if len(pts) > n_total_points:
        raise ValueError(
            f"Need at least {len(pts)} points to include all scenario centers, "
            f"but n_total_points={n_total_points}"
        )

    n_random = n_total_points - len(pts)
    if n_random > 0:
        T_rand = rng.uniform(Tmin, Tmax, size=n_random)
        C_rand = rng.uniform(Caomin, Caomax, size=n_random)
        rand_pts = np.column_stack([T_rand, C_rand])
        pts = np.vstack([pts, rand_pts])
        pts = np.unique(np.round(pts, 12), axis=0)

        while len(pts) < n_total_points:
            extra = np.column_stack([
                rng.uniform(Tmin, Tmax, size=1),
                rng.uniform(Caomin, Caomax, size=1)
            ])
            pts = np.vstack([pts, extra])
            pts = np.unique(np.round(pts, 12), axis=0)

    return pts[:n_total_points]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_total_points", type=int, default=170)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_csv", type=str, default="data.csv")

    args = parser.parse_args()

    pts = build_fixed_points(args.n_total_points, args.seed)

    Tc = 0.5 * (Tmin + Tmax)
    Cc0 = 0.5 * (Caomin + Caomax)
    guess0 = np.array([Cco, Cbo, Cc0], dtype=np_dtype)

    sol_center, ok, mesg = solve_equilibrium(Tc, Cc0, guess0)
    if not ok:
        raise RuntimeError(f"Center solve failed: {mesg}")

    d2 = (pts[:, 0] - Tc)**2 + (pts[:, 1] - Cc0)**2
    order = np.argsort(d2)
    pts = pts[order]

    rows = []
    fail_rows = []
    guess = sol_center.copy()

    for Tpt, Cpt in pts:
        sol, ok, mesg = solve_equilibrium(float(Tpt), float(Cpt), guess)
        if ok:
            Cc_sol, Cb_sol, Ca_sol = sol
            rows.append({
                "Temperature (T)": float(Tpt),
                "Cao": float(Cpt),
                "Ca": float(Ca_sol),
                "Cb": float(Cb_sol),
                "Cc": float(Cc_sol),
            })
            guess = sol
        else:
            fail_rows.append((float(Tpt), float(Cpt), mesg))

    df = pd.DataFrame(rows).sort_values(["Temperature (T)", "Cao"]).reset_index(drop=True)
    df.to_csv(args.out_csv, index=False, float_format="%.17g",)
    print(f"Saved fixed dataset with {len(df)} solved points to {args.out_csv}")

    if fail_rows:
        fail_df = pd.DataFrame(fail_rows, columns=["T", "Cao", "message"])
        fail_df.to_csv("failed_points_fixed_data.csv", index=False)
        print(f"Warning: {len(fail_rows)} points failed. Saved failed_points_fixed_data.csv")

if __name__ == "__main__":
    main()