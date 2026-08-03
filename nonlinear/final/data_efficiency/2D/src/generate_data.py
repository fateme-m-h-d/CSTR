import argparse

import numpy as np
import pandas as pd
from scipy.optimize import fsolve

from .config import C_MAX, C_MIN, N_C_REGIONS, T_EDGES

np_dtype = np.float32
V = 10.0
Q = 1.0
tau = V / Q
Afo = 10e12
Eaf = 90000.0
Aro = 10e10
Ear = 80000.0
R = 8.314
Cbo = 2.0
Cco = 0.0
XTOL = 1e-11


def equations(variables, T, Cao):
    Cc, Cb, Ca = variables
    kf = Afo * np.exp(-Eaf / (R * T))
    kr = Aro * np.exp(-Ear / (R * T))
    reaction = Cao - Ca - kf * Ca * Cb**2 * tau + kr * Cc * tau
    species_b = Cbo - Cb - 2.0 * kf * Ca * Cb**2 * tau + 2.0 * kr * Cc * tau
    mass_balance = Cc - (Cao - Ca + Cbo - Cb + Cco)
    return [reaction, species_b, mass_balance]


def solve_equilibrium(T, Cao, guess):
    solution, _, status, message = fsolve(
        equations, guess, args=(T, Cao), full_output=True, xtol=XTOL
    )
    return solution, status == 1, message


def build_sampling_points(T_edges, C_edges, n_inner, seed):
    rng = np.random.default_rng(seed)
    points = []
    for i in range(len(T_edges) - 1):
        for j in range(len(C_edges) - 1):
            T0, T1 = float(T_edges[i]), float(T_edges[i + 1])
            C0, C1 = float(C_edges[j]), float(C_edges[j + 1])
            Tc, Cc = 0.5 * (T0 + T1), 0.5 * (C0 + C1)
            points.extend([(T0, C0), (T0, C1), (T1, C0), (T1, C1), (Tc, Cc)])
            Tr = rng.uniform(T0, T1, size=n_inner)
            Cr = rng.uniform(C0, C1, size=n_inner)
            points.extend(zip(Tr, Cr))

    points = np.asarray(points, dtype=np_dtype)
    points = np.unique(np.round(points, 12), axis=0)
    return points[:, 0], points[:, 1]


def generate_dataset(n_inner_per_region, seed, output_path):
    T_edges = np.asarray(T_EDGES, dtype=float)
    C_edges = np.linspace(C_MIN, C_MAX, N_C_REGIONS + 1)
    np.savez("region_edges.npz", T_edges=T_edges, C_edges=C_edges)
    T_samples, C_samples = build_sampling_points(
        T_edges, C_edges, n_inner_per_region, seed
    )

    rows = []
    failures = []
    for iT in range(len(T_edges) - 1):
        for iC in range(len(C_edges) - 1):
            region_id = iT * (len(C_edges) - 1) + iC
            T0, T1 = float(T_edges[iT]), float(T_edges[iT + 1])
            C0, C1 = float(C_edges[iC]), float(C_edges[iC + 1])
            Tc, Cc = 0.5 * (T0 + T1), 0.5 * (C0 + C1)
            mask = (
                (T_samples >= T0 - 1e-12)
                & (T_samples <= T1 + 1e-12)
                & (C_samples >= C0 - 1e-12)
                & (C_samples <= C1 + 1e-12)
            )
            region_points = np.column_stack([T_samples[mask], C_samples[mask]])

            guess = np.array([Cco, Cbo, Cc], dtype=np_dtype)
            center_solution, ok, message = solve_equilibrium(Tc, Cc, guess)
            if not ok:
                failures.append((region_id, Tc, Cc, message))
                continue

            Cc_ss, Cb_ss, Ca_ss = center_solution
            rows.append({
                "region_id": region_id,
                "iT": iT,
                "iC": iC,
                "Temperature (T)": Tc,
                "Cao": Cc,
                "Ca": Ca_ss,
                "Cb": Cb_ss,
                "Cc": Cc_ss,
                "is_center": 1,
            })

            distance = (region_points[:, 0] - Tc) ** 2 + (region_points[:, 1] - Cc) ** 2
            region_points = region_points[np.argsort(distance)]
            guess = center_solution.copy()
            for T_point, C_point in region_points:
                if abs(T_point - Tc) < 1e-12 and abs(C_point - Cc) < 1e-12:
                    continue
                solution, ok, message = solve_equilibrium(
                    float(T_point), float(C_point), guess
                )
                if not ok:
                    failures.append((region_id, float(T_point), float(C_point), message))
                    continue
                Cc_sol, Cb_sol, Ca_sol = solution
                rows.append({
                    "region_id": region_id,
                    "iT": iT,
                    "iC": iC,
                    "Temperature (T)": float(T_point),
                    "Cao": float(C_point),
                    "Ca": Ca_sol,
                    "Cb": Cb_sol,
                    "Cc": Cc_sol,
                    "is_center": 0,
                })
                guess = solution

    detailed = pd.DataFrame(rows)
    output = (
        detailed[["Temperature (T)", "Cao", "Ca", "Cb", "Cc"]]
        .sort_values(["Temperature (T)", "Cao"])
        .reset_index(drop=True)
    )
    output.to_csv(output_path, index=False)
    if failures:
        pd.DataFrame(
            failures, columns=["region_id", "T", "Cao", "message"]
        ).to_csv("failed_points.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_inner_per_region", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_csv", default="data.csv")
    args = parser.parse_args()
    generate_dataset(args.n_inner_per_region, args.seed, args.out_csv)


if __name__ == "__main__":
    main()
