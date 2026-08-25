"""Offline axiswise Taylor-error partitioning for the 2D CSTR case.

This module is intended for the current tensor-product rectangular partition
used by PL-KKT-hPINN.  It builds a dense physical reference surface
z*(T, Cao) = [T, Cao, Ca, Cb, Cc], estimates the full-constraint Hessian norm
and the physical-state sensitivities with respect to each input, then places
nonuniform edges so the sampled axiswise Taylor bounds are approximately equal.

Important: this is an axiswise sampled bound surrogate for a tensor grid.  It
is more complete than using only partial d2f/dT2 or d2f/dCao2 at one nominal
slice, but it is not a certified global 2D cell bound.  A certified result
would require interval bounds over every full input-output cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .generate_data import (
    Afo,
    Aro,
    Cbo,
    Cco,
    Eaf,
    Ear,
    R,
    Caomin,
    Caomax,
    Tmin,
    Tmax,
    solve_equilibrium,
    tau,
)


@dataclass(frozen=True)
class ReferenceSurface:
    T: np.ndarray
    Cao: np.ndarray
    state: np.ndarray  # shape (nT, nC, 5), columns [T, Cao, Ca, Cb, Cc]
    hessian_norm: np.ndarray
    tangent_T_norm: np.ndarray
    tangent_C_norm: np.ndarray
    monitor_T: np.ndarray
    monitor_C: np.ndarray


def _solve_checked(T: float, Cao: float, guesses: list[np.ndarray]) -> np.ndarray:
    """Solve one equilibrium point, trying several continuation guesses."""

    messages: list[str] = []
    for guess in guesses:
        sol, ok, message = solve_equilibrium(float(T), float(Cao), np.asarray(guess, dtype=float))
        if ok and np.all(np.isfinite(sol)):
            return np.asarray(sol, dtype=float)
        messages.append(str(message))
    raise RuntimeError(
        f"Equilibrium solve failed at T={T:.12g}, Cao={Cao:.12g}. "
        f"Tried {len(guesses)} guesses. Last messages: {messages[-3:]}"
    )


def _fill_row(
    solved: np.ndarray,
    i: int,
    j0: int,
    T_grid: np.ndarray,
    C_grid: np.ndarray,
) -> None:
    """Fill a T-row by continuation away from its already-solved center point."""

    for j in range(j0 + 1, len(C_grid)):
        fallback = np.array([Cco, Cbo, C_grid[j]], dtype=float)
        solved[i, j] = _solve_checked(
            T_grid[i],
            C_grid[j],
            [solved[i, j - 1], fallback],
        )

    for j in range(j0 - 1, -1, -1):
        fallback = np.array([Cco, Cbo, C_grid[j]], dtype=float)
        solved[i, j] = _solve_checked(
            T_grid[i],
            C_grid[j],
            [solved[i, j + 1], fallback],
        )


def _reaction_hessian(T: float, Ca: float, Cb: float, Cc: float) -> np.ndarray:
    """Hessian of the nonlinear reaction constraint in [T,Cao,Ca,Cb,Cc]."""

    kf = Afo * np.exp(-Eaf / (R * T))
    kr = Aro * np.exp(-Ear / (R * T))

    af = Eaf / (R * T**2)
    ar = Ear / (R * T**2)
    kf_1 = kf * af
    kr_1 = kr * ar
    kf_2 = kf * (af**2 - 2.0 * Eaf / (R * T**3))
    kr_2 = kr * (ar**2 - 2.0 * Ear / (R * T**3))

    H = np.zeros((5, 5), dtype=float)

    H[0, 0] = -tau * kf_2 * Ca * Cb**2 + tau * kr_2 * Cc
    H[0, 2] = H[2, 0] = -tau * kf_1 * Cb**2
    H[0, 3] = H[3, 0] = -2.0 * tau * kf_1 * Ca * Cb
    H[0, 4] = H[4, 0] = tau * kr_1

    H[2, 3] = H[3, 2] = -2.0 * tau * kf * Cb
    H[3, 3] = -2.0 * tau * kf * Ca

    return H


def build_reference_surface(n_T: int = 181, n_C: int = 81) -> ReferenceSurface:
    """Solve the physical surface and construct axiswise Taylor monitors."""

    if n_T < 5 or n_C < 5:
        raise ValueError("reference grid must have at least 5 points per axis")

    T_grid = np.linspace(Tmin, Tmax, n_T, dtype=float)
    C_grid = np.linspace(Caomin, Caomax, n_C, dtype=float)
    i0 = int(np.argmin(np.abs(T_grid - 0.5 * (Tmin + Tmax))))
    j0 = int(np.argmin(np.abs(C_grid - 0.5 * (Caomin + Caomax))))

    solved = np.empty((n_T, n_C, 3), dtype=float)  # [Cc, Cb, Ca]
    fallback0 = np.array([Cco, Cbo, C_grid[j0]], dtype=float)
    solved[i0, j0] = _solve_checked(T_grid[i0], C_grid[j0], [fallback0])
    _fill_row(solved, i0, j0, T_grid, C_grid)

    for i in range(i0 + 1, n_T):
        fallback = np.array([Cco, Cbo, C_grid[j0]], dtype=float)
        solved[i, j0] = _solve_checked(
            T_grid[i], C_grid[j0], [solved[i - 1, j0], fallback]
        )
        _fill_row(solved, i, j0, T_grid, C_grid)

    for i in range(i0 - 1, -1, -1):
        fallback = np.array([Cco, Cbo, C_grid[j0]], dtype=float)
        solved[i, j0] = _solve_checked(
            T_grid[i], C_grid[j0], [solved[i + 1, j0], fallback]
        )
        _fill_row(solved, i, j0, T_grid, C_grid)

    cc = solved[:, :, 0]
    cb = solved[:, :, 1]
    ca = solved[:, :, 2]

    TT, CC = np.meshgrid(T_grid, C_grid, indexing="ij")
    state = np.stack([TT, CC, ca, cb, cc], axis=-1)

    dz_dT = np.gradient(state, T_grid, axis=0, edge_order=2)
    dz_dC = np.gradient(state, C_grid, axis=1, edge_order=2)
    tangent_T_norm = np.linalg.norm(dz_dT, axis=-1)
    tangent_C_norm = np.linalg.norm(dz_dC, axis=-1)

    hessian_norm = np.empty((n_T, n_C), dtype=float)
    for i in range(n_T):
        for j in range(n_C):
            H = _reaction_hessian(T_grid[i], ca[i, j], cb[i, j], cc[i, j])
            hessian_norm[i, j] = np.linalg.norm(H, ord=2)

    # If only input j varies over half-width h/2, then
    # ||Delta z|| <= ||dz*/dx_j|| h/2 and
    # |R| <= ||H|| ||dz*/dx_j||^2 h^2 / 8.
    local_T = hessian_norm * tangent_T_norm**2
    local_C = hessian_norm * tangent_C_norm**2

    # Worst case over the other input coordinate, not one nominal slice.
    monitor_T = np.max(local_T, axis=1)
    monitor_C = np.max(local_C, axis=0)

    return ReferenceSurface(
        T=T_grid,
        Cao=C_grid,
        state=state,
        hessian_norm=hessian_norm,
        tangent_T_norm=tangent_T_norm,
        tangent_C_norm=tangent_C_norm,
        monitor_T=monitor_T,
        monitor_C=monitor_C,
    )


def _interval_max(x: np.ndarray, values: np.ndarray, left: float, right: float) -> float:
    mask = (x >= left) & (x <= right)
    candidates = [np.interp(left, x, values), np.interp(right, x, values)]
    if np.any(mask):
        candidates.extend(values[mask].tolist())
    return float(np.max(candidates))


def sampled_axis_bound(
    left: float,
    right: float,
    x: np.ndarray,
    monitor: np.ndarray,
    safety_factor: float = 1.10,
) -> float:
    if right <= left:
        return 0.0
    mu_max = _interval_max(x, monitor, left, right)
    h = right - left
    return float(safety_factor * mu_max * h**2 / 8.0)


def _right_edge_for_target(
    left: float,
    target: float,
    maximum_right: float,
    x: np.ndarray,
    monitor: np.ndarray,
    safety_factor: float,
) -> float:
    if left >= maximum_right:
        return maximum_right

    full_value = sampled_axis_bound(
        left, maximum_right, x, monitor, safety_factor
    )
    if full_value <= target:
        return maximum_right

    domain_length = x[-1] - x[0]
    tiny = max(1e-12, 1e-12 * domain_length)
    lower = min(left + tiny, maximum_right)

    def mismatch(right: float) -> float:
        return sampled_axis_bound(
            left, right, x, monitor, safety_factor
        ) - target

    return float(brentq(mismatch, lower, maximum_right, xtol=1e-11, rtol=1e-10))


def equal_error_edges_from_monitor(
    x: np.ndarray,
    monitor: np.ndarray,
    n_segments: int,
    safety_factor: float = 1.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Place 1D edges so sampled max-based Taylor bounds are equal."""

    x = np.asarray(x, dtype=float)
    monitor = np.maximum(np.asarray(monitor, dtype=float), 1e-18)
    if x.ndim != 1 or monitor.shape != x.shape:
        raise ValueError("x and monitor must be one-dimensional arrays of equal length")
    if n_segments < 1:
        raise ValueError("n_segments must be positive")

    xmin, xmax = float(x[0]), float(x[-1])
    if n_segments == 1:
        edges = np.array([xmin, xmax], dtype=float)
        bounds = np.array([
            sampled_axis_bound(xmin, xmax, x, monitor, safety_factor)
        ])
        return edges, bounds

    min_width = max(1e-10 * (xmax - xmin), np.finfo(float).eps)

    def march(target: float) -> np.ndarray:
        edges = [xmin]
        for k in range(n_segments - 1):
            remaining_after = n_segments - k - 1
            maximum_right = xmax - remaining_after * min_width
            right = _right_edge_for_target(
                edges[-1],
                target,
                maximum_right,
                x,
                monitor,
                safety_factor,
            )
            if right <= edges[-1]:
                raise RuntimeError("non-increasing adaptive edges")
            edges.append(right)
        edges.append(xmax)
        return np.asarray(edges, dtype=float)

    whole = sampled_axis_bound(xmin, xmax, x, monitor, safety_factor)

    def final_mismatch(target: float) -> float:
        edges = march(target)
        return (
            sampled_axis_bound(edges[-2], edges[-1], x, monitor, safety_factor)
            - target
        )

    lower = max(np.finfo(float).tiny, whole / (1000.0 * n_segments**2))
    upper = max(whole, lower * 10.0)
    f_lower = final_mismatch(lower)
    f_upper = final_mismatch(upper)

    for _ in range(80):
        if f_lower >= 0.0 and f_upper <= 0.0:
            break
        if f_lower < 0.0:
            lower *= 0.25
            f_lower = final_mismatch(lower)
        if f_upper > 0.0:
            upper *= 2.0
            f_upper = final_mismatch(upper)
    else:
        raise RuntimeError("could not bracket the equal-error target")

    target = float(brentq(final_mismatch, lower, upper, xtol=1e-11, rtol=1e-10))
    edges = march(target)
    bounds = np.array([
        sampled_axis_bound(edges[i], edges[i + 1], x, monitor, safety_factor)
        for i in range(n_segments)
    ])
    return edges, bounds


def build_axiswise_partition(
    n_T_regions: int,
    n_C_regions: int,
    reference_T_points: int = 181,
    reference_C_points: int = 81,
    safety_factor: float = 1.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ReferenceSurface]:
    surface = build_reference_surface(reference_T_points, reference_C_points)
    T_edges, T_bounds = equal_error_edges_from_monitor(
        surface.T, surface.monitor_T, n_T_regions, safety_factor
    )
    C_edges, C_bounds = equal_error_edges_from_monitor(
        surface.Cao, surface.monitor_C, n_C_regions, safety_factor
    )
    return T_edges, C_edges, T_bounds, C_bounds, surface
