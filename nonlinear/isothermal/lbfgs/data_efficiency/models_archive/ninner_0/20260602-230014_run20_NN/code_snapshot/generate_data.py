import argparse
import numpy as np
import pandas as pd
from scipy.optimize import fsolve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


np_dtype = np.float32

# ============================================================
# Isothermal CSTR constants
# ============================================================
V = 10.0      # L
Q = 1.0       # L/s
tau = V / Q   # s

# fixed temperature; change this if you want a different isothermal case
T_ISO = 350.0  # K

# feed concentrations
Cbo = 2.0
Cco = 0.0

# Original Arrhenius parameters are used only once to compute constant kf, kr
Afo = 10e12
Eaf = 90000.0
Aro = 10e10
Ear = 80000.0
R = 8.314

kf_const = Afo * np.exp(-Eaf / (R * T_ISO))
kr_const = Aro * np.exp(-Ear / (R * T_ISO))

XTOL = 1e-11

# Only Cao varies now
Caomin, Caomax = 0.5, 1.5


# ============================================================
# Nonlinear steady-state equations
# ============================================================
def equations(variables, Cao):
    """
    Unknowns are ordered as:
        variables = (Cc, Cb, Ca)

    Isothermal CSTR equations.
    """
    Cc, Cb, Ca = variables

    eq1 = Cao - Ca - kf_const * Ca * (Cb ** 2) * tau + kr_const * Cc * tau
    eq2 = Cbo - Cb - 2.0 * kf_const * Ca * (Cb ** 2) * tau + 2.0 * kr_const * Cc * tau
    eq3 = Cc - (Cao - Ca + Cbo - Cb + Cco)

    return [eq1, eq2, eq3]


def solve_equilibrium(Cao, guess):
    sol, info, ier, mesg = fsolve(
        equations,
        guess,
        args=(Cao,),
        full_output=True,
        xtol=XTOL,
    )
    return sol, (ier == 1), mesg


# ============================================================
# 1D region-based sampling
# ============================================================
def build_sampling_points_1d(C_edges, n_inner_per_region, seed):
    """
    1D version of the 2D data-efficiency sampling idea.

    For each Cao region, include:
        - left boundary
        - right boundary
        - center point
        - n_inner_per_region random interior points

    Afterward, duplicate shared boundaries are removed.

    For nC regions, the approximate number of points is:
        (nC + 1) boundaries + nC centers + nC*n_inner_per_region
    """

    rng = np.random.default_rng(seed)
    pts = []

    for i in range(len(C_edges) - 1):
        C0 = float(C_edges[i])
        C1 = float(C_edges[i + 1])
        Cc = 0.5 * (C0 + C1)

        # 1D equivalent of corners + center
        pts.extend([C0, C1, Cc])

        # random interior points inside this region
        if n_inner_per_region > 0:
            rand_pts = rng.uniform(C0, C1, size=n_inner_per_region)
            pts.extend(rand_pts.tolist())

    # remove duplicate shared boundaries
    pts = np.unique(np.round(np.array(pts, dtype=float), 12))
    return np.sort(pts)


def assign_region_id(Cao, C_edges):
    """
    Assign each Cao point to one region.
    Last boundary belongs to the last region.
    """

    idx = np.digitize(Cao, C_edges) - 1
    idx = np.clip(idx, 0, len(C_edges) - 2)
    return int(idx)


# ============================================================
# Plotting
# ============================================================
def plot_outputs_vs_input(df, C_edges, out_file="outputs_vs_Cao.png"):
    """
    Plot solved CSTR outputs Ca, Cb, and Cc versus the input Cao.
    """

    plt.figure(figsize=(8, 5))

    plt.plot(
        df["Cao"],
        df["Ca"],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label=r"$C_A$",
    )
    plt.plot(
        df["Cao"],
        df["Cb"],
        marker="s",
        markersize=3,
        linewidth=1.5,
        label=r"$C_B$",
    )
    plt.plot(
        df["Cao"],
        df["Cc"],
        marker="^",
        markersize=3,
        linewidth=1.5,
        label=r"$C_C$",
    )

    # show region boundaries lightly
    for c in C_edges:
        plt.axvline(c, linewidth=0.8, alpha=0.25)

    plt.xlabel(r"Input feed concentration $C_{A0}$")
    plt.ylabel("Output concentration")
    plt.title("Isothermal CSTR outputs versus input concentration")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"Saved output plot to {out_file}")


def plot_sampling_points(df, C_edges, out_file="sampling_points_Cao.png"):
    """
    Simple diagnostic plot showing the 1D sample locations.
    """

    plt.figure(figsize=(8, 2.8))

    y = np.zeros(len(df))
    plt.scatter(df["Cao"], y, s=18, label="sample points")

    centers = df[df["is_center"] == 1]
    plt.scatter(
        centers["Cao"],
        np.zeros(len(centers)),
        s=60,
        marker="*",
        label="region centers",
    )

    for c in C_edges:
        plt.axvline(c, linewidth=0.8, alpha=0.35)

    plt.yticks([])
    plt.xlabel(r"$C_{A0}$")
    plt.title("1D data-efficiency sampling points")
    plt.legend()
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"Saved sampling plot to {out_file}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nC_regions",
        type=int,
        default=30,
        help="Fixed number of Cao regions for the data-efficiency experiment.",
    )
    parser.add_argument(
        "--n_inner_per_region",
        type=int,
        default=0,
        help="Number of random interior points added inside each Cao region.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for interior points.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="data.csv",
        help="Output CSV file used by training/linearization.",
    )
    parser.add_argument(
        "--plot_file",
        type=str,
        default="outputs_vs_Cao.png",
        help="Output plot file for CSTR outputs.",
    )
    parser.add_argument(
        "--sampling_plot_file",
        type=str,
        default="sampling_points_Cao.png",
        help="Output plot file for sample locations.",
    )

    args = parser.parse_args()

    if args.nC_regions < 1:
        raise ValueError("nC_regions must be at least 1.")

    if args.n_inner_per_region < 0:
        raise ValueError("n_inner_per_region must be nonnegative.")

    # Fixed region edges for this scenario
    C_edges = np.linspace(Caomin, Caomax, args.nC_regions + 1, dtype=float)
    C_centers = 0.5 * (C_edges[:-1] + C_edges[1:])

    # Build points using region-wise sampling
    pts = build_sampling_points_1d(
        C_edges=C_edges,
        n_inner_per_region=args.n_inner_per_region,
        seed=args.seed,
    )

    # Save edges too; linearization.py will also save region_edges.npz,
    # but this is useful for checking consistency.
    np.savez("sampling_region_edges.npz", C_edges=C_edges)

    # Solve center of full domain first for a stable initial guess
    Cao_mid = 0.5 * (Caomin + Caomax)
    guess0 = np.array([Cco, Cbo, Cao_mid], dtype=np_dtype)

    sol_mid, ok, mesg = solve_equilibrium(Cao_mid, guess0)
    if not ok:
        raise RuntimeError(f"Middle-point solve failed: {mesg}")

    # Solve points from middle outward for stable warm-starting
    order = np.argsort((pts - Cao_mid) ** 2)
    pts_ordered = pts[order]

    rows = []
    fail_rows = []
    guess = sol_mid.copy()

    center_tol = 1e-10

    for Caopt in pts_ordered:
        sol, ok, mesg = solve_equilibrium(float(Caopt), guess)

        if ok:
            Cc_sol, Cb_sol, Ca_sol = sol

            region_id = assign_region_id(float(Caopt), C_edges)
            is_center = int(np.any(np.isclose(Caopt, C_centers, atol=center_tol, rtol=0.0)))
            is_boundary = int(np.any(np.isclose(Caopt, C_edges, atol=center_tol, rtol=0.0)))

            rows.append({
                "Cao": float(Caopt),
                "Ca": float(Ca_sol),
                "Cb": float(Cb_sol),
                "Cc": float(Cc_sol),
                "region_id": region_id,
                "is_center": is_center,
                "is_boundary": is_boundary,
            })

            # warm-start next point
            guess = sol

        else:
            fail_rows.append({
                "Cao": float(Caopt),
                "message": mesg,
            })

    if len(rows) == 0:
        raise RuntimeError("No points were solved successfully.")

    df_full = pd.DataFrame(rows).sort_values("Cao").reset_index(drop=True)

    # This file keeps diagnostic columns.
    df_full.to_csv("data_with_region_info.csv", index=False)

    # This is the file used by your current utils.py and training code.
    df_out = df_full[["Cao", "Ca", "Cb", "Cc"]].copy()
    df_out.to_csv(args.out_csv, index=False)

    plot_outputs_vs_input(df_out, C_edges, args.plot_file)
    plot_sampling_points(df_full, C_edges, args.sampling_plot_file)

    if fail_rows:
        fail_df = pd.DataFrame(fail_rows)
        fail_df.to_csv("failed_points_fixed_data.csv", index=False)
        print(f"Warning: {len(fail_rows)} points failed. Saved failed_points_fixed_data.csv")

    expected_min_points = (args.nC_regions + 1) + args.nC_regions
    expected_points = expected_min_points + args.nC_regions * args.n_inner_per_region

    print("\n=== Data generation summary ===")
    print(f"nC_regions = {args.nC_regions}")
    print(f"n_inner_per_region = {args.n_inner_per_region}")
    print(f"Expected points after duplicate boundary removal = {expected_points}")
    print(f"Solved points saved to {args.out_csv} = {len(df_out)}")
    print(f"Diagnostic file saved to data_with_region_info.csv")
    print(f"T_ISO = {T_ISO} K")
    print(f"kf_const = {kf_const:.6e}")
    print(f"kr_const = {kr_const:.6e}")
    print(f"Caomin = {Caomin}, Caomax = {Caomax}")

    # Check that all region centers are present for linearization.py
    missing_centers = []
    for c in C_centers:
        if not np.any(np.isclose(df_out["Cao"].to_numpy(), c, atol=center_tol, rtol=0.0)):
            missing_centers.append(c)

    if missing_centers:
        raise RuntimeError(
            "Some region centers are missing from data.csv. "
            f"Missing centers: {missing_centers}"
        )

    print("All region centers are present in data.csv.")


if __name__ == "__main__":
    main()