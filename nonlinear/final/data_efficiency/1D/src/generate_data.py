import argparse

import numpy as np
import pandas as pd
from scipy.optimize import fsolve

np_dtype = np.float32

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
    Cc, Cb, Ca = variables
    reaction = Cao - Ca - kf_const * Ca * Cb**2 * tau + kr_const * Cc * tau
    species_b = Cbo - Cb - 2.0 * kf_const * Ca * Cb**2 * tau + 2.0 * kr_const * Cc * tau
    mass_balance = Cc - (Cao - Ca + Cbo - Cb + Cco)
    return [reaction, species_b, mass_balance]


def solve_equilibrium(Cao, guess):
    solution, _, status, message = fsolve(
        equations,
        guess,
        args=(Cao,),
        full_output=True,
        xtol=XTOL,
    )
    return solution, status == 1, message


def build_sampling_points(C_edges, n_inner_per_region, seed):
    rng = np.random.default_rng(seed)
    points = []

    for lower, upper in zip(C_edges[:-1], C_edges[1:]):
        center = 0.5 * (lower + upper)
        points.extend([float(lower), float(upper), float(center)])
        if n_inner_per_region > 0:
            points.extend(
                rng.uniform(lower, upper, size=n_inner_per_region).tolist()
            )

    return np.sort(np.unique(np.round(np.asarray(points, dtype=float), 12)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nC_regions", type=int, default=30)
    parser.add_argument("--n_inner_per_region", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_csv", type=str, default="data.csv")
    args = parser.parse_args()

    if args.nC_regions < 1:
        raise ValueError("nC_regions must be at least 1.")
    if args.n_inner_per_region < 0:
        raise ValueError("n_inner_per_region must be nonnegative.")

    C_edges = np.linspace(Caomin, Caomax, args.nC_regions + 1, dtype=float)
    C_centers = 0.5 * (C_edges[:-1] + C_edges[1:])
    points = build_sampling_points(
        C_edges, args.n_inner_per_region, args.seed
    )

    Cao_mid = 0.5 * (Caomin + Caomax)
    initial_guess = np.array([Cco, Cbo, Cao_mid], dtype=np_dtype)
    middle_solution, ok, message = solve_equilibrium(Cao_mid, initial_guess)
    if not ok:
        raise RuntimeError(f"Middle-point solve failed: {message}")

    order = np.argsort((points - Cao_mid) ** 2)
    rows = []
    failed = []
    guess = middle_solution.copy()

    for Cao in points[order]:
        solution, ok, message = solve_equilibrium(float(Cao), guess)
        if ok:
            Cc, Cb, Ca = solution
            rows.append({
                "Cao": float(Cao),
                "Ca": float(Ca),
                "Cb": float(Cb),
                "Cc": float(Cc),
            })
            guess = solution
        else:
            failed.append({"Cao": float(Cao), "message": message})

    if not rows:
        raise RuntimeError("No points were solved successfully.")

    frame = pd.DataFrame(rows).sort_values("Cao").reset_index(drop=True)
    frame.to_csv(args.out_csv, index=False)

    missing_centers = [
        center for center in C_centers
        if not np.any(np.isclose(frame["Cao"], center, atol=1e-10, rtol=0.0))
    ]
    if missing_centers:
        raise RuntimeError(f"Missing region centers: {missing_centers}")

    if failed:
        pd.DataFrame(failed).to_csv("failed_points_fixed_data.csv", index=False)


if __name__ == "__main__":
    main()
