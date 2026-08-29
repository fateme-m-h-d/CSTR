"""Generate the common 2D CSTR dataset used by all three methods.

This is a standalone adaptation of PL-KKT-hPINN/2D/src/generate_data.py.
The physical model, sampling bounds, default number of points (170), seed (0),
and inclusion of the PL 2D scenario centers are intentionally preserved.

IMPORTANT
---------
* Inputs:  Temperature T and inlet concentration Cao.
* Outputs: Ca, Cb, Cc.
* Output columns are exactly:
      Temperature (T), Cao, Ca, Cb, Cc
* ENFORCE and KKT-HardNet do not use PL regions or linearization. The scenario
  centers appear here only because the goal is to use exactly the same dataset
  as the PL-KKT-hPINN 2D benchmark.
"""

import argparse
import numpy as np
import pandas as pd
from scipy.optimize import fsolve

from config_cstr_2d import N_C_REGIONS as NC_REGIONS, SEGMENT_SCENARIOS

np_dtype = np.float32

V = 10.0
Q = 1.0
TAU = V / Q

AFO = 1.0e13
EAF = 90000.0
ARO = 1.0e11
EAR = 80000.0
R = 8.314

CBO = 2.0
CCO = 0.0

XTOL = 1.0e-11

TMIN, TMAX = 280.0, 460.0
CAO_MIN, CAO_MAX = 0.8, 1.2


def equations(vars_, T, Cao):
    """Steady-state nonlinear CSTR equations at one (T, Cao) input pair."""
    Cc, Cb, Ca = vars_
    kf = AFO * np.exp(-EAF / (R * T))
    kr = ARO * np.exp(-EAR / (R * T))

    eq1 = Cao - Ca - kf * Ca * (Cb**2) * TAU + kr * Cc * TAU
    eq2 = CBO - Cb - 2.0 * kf * Ca * (Cb**2) * TAU + 2.0 * kr * Cc * TAU
    eq3 = Cc - (Cao - Ca + CBO - Cb + CCO)
    return [eq1, eq2, eq3]


def solve_equilibrium(T, Cao, guess):
    sol, info, ier, mesg = fsolve(
        equations, guess, args=(T, Cao), full_output=True, xtol=XTOL
    )
    return sol, (ier == 1), mesg


def build_fixed_points(n_total_points, seed):
    """Reproduce the fixed 2D input locations used by the PL 2D generator."""
    rng = np.random.default_rng(seed)

    # These centers are included only to match PL-KKT-hPINN's 2D dataset.
    # ENFORCE and KKT-HardNet themselves have no regions/linearization.
    C_edges_fixed = np.linspace(CAO_MIN, CAO_MAX, NC_REGIONS + 1)
    C_centers = 0.5 * (C_edges_fixed[:-1] + C_edges_fixed[1:])

    center_pts = []
    for nT in SEGMENT_SCENARIOS:
        T_edges = np.linspace(TMIN, TMAX, nT + 1)
        T_centers = 0.5 * (T_edges[:-1] + T_edges[1:])
        for Tc in T_centers:
            for Cc in C_centers:
                center_pts.append([Tc, Cc])

    center_pts = np.unique(np.round(np.array(center_pts, dtype=float), 12), axis=0)
    anchors = np.array(
        [
            [TMIN, CAO_MIN],
            [TMIN, CAO_MAX],
            [TMAX, CAO_MIN],
            [TMAX, CAO_MAX],
        ],
        dtype=float,
    )

    pts = np.vstack([center_pts, anchors])
    pts = np.unique(np.round(pts, 12), axis=0)

    if len(pts) > n_total_points:
        raise ValueError(
            f"Need at least {len(pts)} points to include all scenario centers, "
            f"but n_total_points={n_total_points}"
        )

    n_random = n_total_points - len(pts)
    if n_random > 0:
        T_rand = rng.uniform(TMIN, TMAX, size=n_random)
        C_rand = rng.uniform(CAO_MIN, CAO_MAX, size=n_random)
        rand_pts = np.column_stack([T_rand, C_rand])
        pts = np.vstack([pts, rand_pts])
        pts = np.unique(np.round(pts, 12), axis=0)

        while len(pts) < n_total_points:
            extra = np.column_stack(
                [
                    rng.uniform(TMIN, TMAX, size=1),
                    rng.uniform(CAO_MIN, CAO_MAX, size=1),
                ]
            )
            pts = np.vstack([pts, extra])
            pts = np.unique(np.round(pts, 12), axis=0)

    return pts[:n_total_points]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-total-points", type=int, default=170)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-csv", default="data_cstr_2d.csv")
    args = parser.parse_args()

    pts = build_fixed_points(args.n_total_points, args.seed)

    Tc = 0.5 * (TMIN + TMAX)
    Cc0 = 0.5 * (CAO_MIN + CAO_MAX)
    guess0 = np.array([CCO, CBO, Cc0], dtype=np_dtype)
    sol_center, ok, mesg = solve_equilibrium(Tc, Cc0, guess0)
    if not ok:
        raise RuntimeError(f"Center solve failed: {mesg}")

    # Solve nearby points first so fsolve can reuse a physically sensible guess.
    d2 = (pts[:, 0] - Tc) ** 2 + (pts[:, 1] - Cc0) ** 2
    pts = pts[np.argsort(d2)]

    rows = []
    fail_rows = []
    guess = sol_center.copy()
    for Tpt, Cpt in pts:
        sol, ok, mesg = solve_equilibrium(float(Tpt), float(Cpt), guess)
        if ok:
            Cc_sol, Cb_sol, Ca_sol = sol
            rows.append(
                {
                    "Temperature (T)": float(Tpt),
                    "Cao": float(Cpt),
                    "Ca": float(Ca_sol),
                    "Cb": float(Cb_sol),
                    "Cc": float(Cc_sol),
                }
            )
            guess = sol
        else:
            fail_rows.append((float(Tpt), float(Cpt), mesg))

    df = (
        pd.DataFrame(rows)
        .sort_values(["Temperature (T)", "Cao"])
        .reset_index(drop=True)
    )
    df.to_csv(args.out_csv, index=False)
    print(f"Saved fixed 2D dataset with {len(df)} solved points to {args.out_csv}")

    if fail_rows:
        fail_df = pd.DataFrame(fail_rows, columns=["T", "Cao", "message"])
        fail_df.to_csv("failed_points_cstr_2d.csv", index=False)
        print(f"Warning: {len(fail_rows)} points failed.")


if __name__ == "__main__":
    main()
