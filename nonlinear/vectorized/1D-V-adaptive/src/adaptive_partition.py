"""One-input reduction of the 2D sampled Hessian/Taylor partition indicator.

At fixed temperature z(Cao) = [Cao, Ca, Cb, Cc] and q = z' @ H_f @ z'.
For an interval of full width h, use safety_factor * max(|q|) * h^2/8.
Bisect the interval with the largest indicator until the target count is reached.
The maximum is sampled on a separately solved reference curve; z' is estimated
by finite differences, as in the supplied 2D methodology. This is not a certified
bound for arbitrary NN outputs. Reference/center points do not enter the dataset.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .generate_data import (
    Caomin, Caomax, Cbo, Cco, equations, kf_const, solve_equilibrium, tau,
)


@dataclass(frozen=True)
class ReferenceCurve:
    Cao: np.ndarray
    state: np.ndarray  # [Cao, Ca, Cb, Cc]
    dz_dC: np.ndarray
    q_CC: np.ndarray


@dataclass
class Interval:
    C_low: float
    C_high: float
    depth: int = 0
    M_CC: float = np.nan
    estimated_bound: float = np.nan

    @property
    def h_C(self):
        return self.C_high - self.C_low

    @property
    def C_center(self):
        return 0.5 * (self.C_low + self.C_high)


def solve_checked(Cao, guesses):
    """Return [Cc, Cb, Ca], checking residuals and nonnegative concentrations."""
    messages = []
    for guess in guesses:
        sol, ok, message = solve_equilibrium(float(Cao), np.asarray(guess, dtype=float))
        sol = np.asarray(sol, dtype=float)
        residual = np.asarray(equations(sol, float(Cao)), dtype=float)
        finite = np.isfinite(sol).all() and np.isfinite(residual).all()
        # fsolve can report stagnation at an already-converged root. Require
        # physical concentrations and small equation residuals in every case.
        if finite and np.min(sol) >= -1e-10 and np.max(np.abs(residual)) <= 1e-9:
            return sol
        messages.append(f"converged={ok}: {message}")
    raise RuntimeError(f"Equilibrium solve failed at Cao={Cao:.12g}: {messages[-2:]}")


def reaction_hessian(Ca, Cb):
    """Hessian of the reaction residual in [Cao, Ca, Cb, Cc] at fixed T."""
    H = np.zeros((4, 4), dtype=float)
    H[1, 2] = H[2, 1] = -2.0 * tau * kf_const * Cb
    H[2, 2] = -2.0 * tau * kf_const * Ca
    return H


def build_reference_curve(reference_C_points=1025):
    if reference_C_points < 5:
        raise ValueError("reference_C_points must be at least 5")
    C_grid = np.linspace(Caomin, Caomax, reference_C_points, dtype=float)
    center = int(np.argmin(np.abs(C_grid - 0.5 * (Caomin + Caomax))))
    solved = np.empty((reference_C_points, 3), dtype=float)
    solved[center] = solve_checked(C_grid[center], [[Cco, Cbo, C_grid[center]]])
    for i in range(center + 1, reference_C_points):
        solved[i] = solve_checked(C_grid[i], [solved[i - 1], [Cco, Cbo, C_grid[i]]])
    for i in range(center - 1, -1, -1):
        solved[i] = solve_checked(C_grid[i], [solved[i + 1], [Cco, Cbo, C_grid[i]]])
    state = np.column_stack([C_grid, solved[:, 2], solved[:, 1], solved[:, 0]])
    dz_dC = np.gradient(state, C_grid, axis=0, edge_order=2)
    q_CC = np.array([
        dz @ reaction_hessian(s[1], s[2]) @ dz
        for s, dz in zip(state, dz_dC)
    ], dtype=float)
    if not np.isfinite(q_CC).all():
        raise RuntimeError("Nonfinite sampled curvature on the reference curve")
    return ReferenceCurve(C_grid, state, dz_dC, q_CC)


def sampled_interval_bound(interval, curve, safety_factor=1.10):
    """Use the sampled-maximum convention of the 2D rectangle method."""
    if interval.h_C <= 0.0:
        raise ValueError("Interval width must be positive")
    mask = (curve.Cao >= interval.C_low) & (curve.Cao <= interval.C_high)
    values = curve.q_CC[mask]
    if values.size == 0:
        nearest = int(np.argmin(np.abs(curve.Cao - interval.C_center)))
        values = curve.q_CC[nearest:nearest + 1]
    M_CC = float(np.max(np.abs(values)))
    return float(safety_factor * M_CC * interval.h_C**2 / 8.0), M_CC


def build_interval_partition(n_regions, reference_C_points=1025, safety_factor=1.10):
    if not isinstance(n_regions, (int, np.integer)) or n_regions < 1:
        raise ValueError("n_regions must be a positive integer")
    if not np.isfinite(safety_factor) or safety_factor <= 0.0:
        raise ValueError("safety_factor must be positive and finite")
    curve = build_reference_curve(reference_C_points)

    def score(interval):
        interval.estimated_bound, interval.M_CC = sampled_interval_bound(
            interval, curve, safety_factor
        )
        return interval

    leaves = [score(Interval(Caomin, Caomax))]
    while len(leaves) < n_regions:
        index = int(np.argmax([cell.estimated_bound for cell in leaves]))
        cell = leaves[index]
        mid = cell.C_center
        if not cell.C_low < mid < cell.C_high:
            raise ValueError("Cannot split further at this numerical precision")
        leaves[index:index + 1] = [
            score(Interval(cell.C_low, mid, cell.depth + 1)),
            score(Interval(mid, cell.C_high, cell.depth + 1)),
        ]
    leaves.sort(key=lambda cell: cell.C_low)
    return leaves, curve


def intervals_to_edges(intervals):
    """Sorted contiguous edges directly supported by 1D-V's bucketize calls."""
    if not intervals:
        raise ValueError("At least one interval is required")
    edges = np.array([intervals[0].C_low] + [cell.C_high for cell in intervals])
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("Interval edges must be strictly increasing")
    if any(a.C_high != b.C_low for a, b in zip(intervals[:-1], intervals[1:])):
        raise ValueError("Intervals must be contiguous and ordered")
    return edges
