import argparse
import numpy as np
import pandas as pd
from scipy.optimize import fsolve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import SEGMENT_SCENARIOS

np_dtype = np.float64

V = 10.0
Q = 1.0
tau = V / Q

T_ISO = 350.0

Cbo = 2.0
Cco = 0.0

Afo = 10e12
Eaf = 90000.0
Aro = 10e10
Ear = 80000.0
R = 8.314

kf_const = Afo * np.exp(-Eaf / (R * T_ISO))
kr_const = Aro * np.exp(-Ear / (R * T_ISO))

XTOL = 1e-11

Caomin, Caomax = 0.5, 1.5


def equations(variables, Cao):
    """Unknowns are ordered as (Cc, Cb, Ca)."""
    Cc, Cb, Ca = variables

    eq1 = Cao - Ca - kf_const * Ca * (Cb ** 2) * tau + kr_const * Cc * tau
    eq2 = Cbo - Cb - 2.0 * kf_const * Ca * (Cb ** 2) * tau + 2.0 * kr_const * Cc * tau
    eq3 = Cc - (Cao - Ca + Cbo - Cb + Cco)
    return [eq1, eq2, eq3]


def solve_equilibrium(Cao, guess):
    sol, info, ier, mesg = fsolve(
        equations, guess, args=(Cao,), full_output=True, xtol=XTOL
    )
    return sol, (ier == 1), mesg


def build_fixed_points(n_total_points, seed):
    rng = np.random.default_rng(seed)

    center_pts = []
    for nC in SEGMENT_SCENARIOS:
        C_edges = np.linspace(Caomin, Caomax, nC + 1)
        C_centers = 0.5 * (C_edges[:-1] + C_edges[1:])
        center_pts.extend(C_centers.tolist())

    center_pts = np.unique(np.round(np.array(center_pts, dtype=float), 12))
    anchors = np.array([Caomin, Caomax], dtype=float)
    pts = np.unique(np.round(np.concatenate([center_pts, anchors]), 12))

    if len(pts) > n_total_points:
        raise ValueError(
            f"Need at least {len(pts)} points to include all scenario centers, "
            f"but n_total_points={n_total_points}"
        )

    n_random = n_total_points - len(pts)
    if n_random > 0:
        rand_pts = rng.uniform(Caomin, Caomax, size=n_random)
        pts = np.unique(np.round(np.concatenate([pts, rand_pts]), 12))
        while len(pts) < n_total_points:
            extra = rng.uniform(Caomin, Caomax, size=1)
            pts = np.unique(np.round(np.concatenate([pts, extra]), 12))

    return np.sort(pts[:n_total_points])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_total_points", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_csv", type=str, default="data.csv")
    args = parser.parse_args()

    pts = build_fixed_points(args.n_total_points, args.seed)

    Cao_center = 0.5 * (Caomin + Caomax)
    guess0 = np.array([Cco, Cbo, Cao_center], dtype=np_dtype)

    sol_center, ok, mesg = solve_equilibrium(Cao_center, guess0)
    if not ok:
        raise RuntimeError(f"Center solve failed: {mesg}")

    order = np.argsort((pts - Cao_center) ** 2)
    pts = pts[order]

    rows = []
    fail_rows = []
    guess = sol_center.copy()

    for Caopt in pts:
        sol, ok, mesg = solve_equilibrium(float(Caopt), guess)
        if ok:
            Cc_sol, Cb_sol, Ca_sol = sol
            rows.append({
                "Cao": float(Caopt),
                "Ca": float(Ca_sol),
                "Cb": float(Cb_sol),
                "Cc": float(Cc_sol),
            })
            guess = sol
        else:
            fail_rows.append((float(Caopt), mesg))

    df = pd.DataFrame(rows).sort_values(["Cao"]).reset_index(drop=True)
    df.to_csv(args.out_csv, index=False, float_format="%.17g",)

    if fail_rows:
        fail_df = pd.DataFrame(fail_rows, columns=["Cao", "message"])
        fail_df.to_csv("failed_points_fixed_data.csv", index=False)


if __name__ == "__main__":
    main()