"""Generate adaptive 1D intervals and compatible PL constraint coefficients."""

import argparse
import numpy as np
import pandas as pd
from .adaptive_partition import build_interval_partition, intervals_to_edges, solve_checked
from .generate_data import Cbo, Cco, T_ISO, kf_const, kr_const, tau


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n_regions", "--nC_regions", dest="n_regions", type=int, required=True,
        help="Total adaptive intervals; the old --nC_regions option also works.",
    )
    parser.add_argument("--reference_C_points", type=int, default=1025)
    parser.add_argument("--safety_factor", type=float, default=1.10)
    return parser.parse_args()


def main():
    args = parse_args()
    intervals, curve = build_interval_partition(
        args.n_regions, args.reference_C_points, args.safety_factor
    )
    C_edges = intervals_to_edges(intervals)
    rows, AB_rows, partition_rows = [], [], []
    for rid, cell in enumerate(intervals):
        Caoss = cell.C_center
        nearest = int(np.argmin(np.abs(curve.Cao - Caoss)))
        ref = curve.state[nearest]
        Ccss, Cbss, Cass = solve_checked(
            Caoss, [[ref[3], ref[2], ref[1]], [Cco, Cbo, Caoss]]
        )
        fss = Caoss - Cass - tau * kf_const * Cass * Cbss**2 + tau * kr_const * Ccss
        # Exact first derivatives of the original 1D reaction residual.
        aCao = 1.0
        aCa = -1.0 - tau * kf_const * Cbss**2
        aCb = -2.0 * tau * kf_const * Cass * Cbss
        aCc = tau * kr_const
        b_rxn = -fss + aCao * Caoss + aCa * Cass + aCb * Cbss + aCc * Ccss
        partition = {
            "region_id": rid, "C_low": cell.C_low, "C_high": cell.C_high,
            "C_center": Caoss, "h_C": cell.h_C, "depth": cell.depth,
            "M_CC": cell.M_CC, "estimated_cell_taylor_bound": cell.estimated_bound,
        }
        partition_rows.append(partition)
        rows.append({
            **partition, "Caoss": Caoss, "Cass": Cass, "Cbss": Cbss,
            "Ccss": Ccss, "fss": fss, "aCao": aCao, "aCa": aCa,
            "aCb": aCb, "aCc": aCc, "b": b_rxn,
        })
        AB_rows.extend([
            {
                "region_id": rid, "constraint_order": 0,
                "constraint_name": "reaction_linearized", "A_Cao": aCao,
                "B_Ca": aCa, "B_Cb": aCb, "B_Cc": aCc, "b": b_rxn,
            },
            {
                "region_id": rid, "constraint_order": 1,
                "constraint_name": "mass_balance", "A_Cao": -1.0,
                "B_Ca": 1.0, "B_Cb": 1.0, "B_Cc": 1.0, "b": float(Cbo + Cco),
            },
        ])
    # Finish all center solves before replacing artifacts. Reference/center
    # solves are offline work; data.csv is never changed by this module.
    pd.DataFrame(rows).to_csv("lin_params.csv", index=False)
    pd.DataFrame(AB_rows).to_csv("ABb_matrices.csv", index=False)
    pd.DataFrame(partition_rows).to_csv("region_partition_summary.csv", index=False)
    np.savez(
        "region_edges.npz", C_edges=C_edges,
        partition=np.asarray("taylor_intervals"), n_regions=np.asarray(len(intervals)),
        reference_C_points=np.asarray(args.reference_C_points),
        safety_factor=np.asarray(args.safety_factor), T_iso=np.asarray(T_ISO),
    )
    print(f"Saved {len(intervals)} adaptive intervals at T={T_ISO:g} K")
    print(f"Interval widths: {np.min(np.diff(C_edges)):.8g} to {np.max(np.diff(C_edges)):.8g}")
    print("Saved region_edges.npz, lin_params.csv, ABb_matrices.csv, region_partition_summary.csv")


if __name__ == "__main__":
    main()
