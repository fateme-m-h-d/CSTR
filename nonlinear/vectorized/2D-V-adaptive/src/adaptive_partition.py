"""Offline full-rectangle Taylor partitioning for the 2D CSTR case.

This module replaces the previous axis-wise tensor-grid partitioner.

The physical solution surface is

    z*(T, Cao) = [T, Cao, Ca, Cb, Cc].

For the nonlinear reaction constraint f(z), a first-order Taylor model around
one cell center has the second-order remainder

    R = 1/2 * Delta z.T @ H_f(xi) @ Delta z.

Using the first-order input-to-state variation

    Delta z ~= z_T Delta T + z_C Delta Cao,

we obtain the sampled 2D quadratic indicator

    R ~= 1/2 [q_TT DeltaT^2 + 2 q_TC DeltaT DeltaC + q_CC DeltaC^2],

where

    q_TT = z_T.T H_f z_T,
    q_TC = z_T.T H_f z_C,
    q_CC = z_C.T H_f z_C.

For a rectangle of full widths h_T and h_C centered at the Taylor point,
|Delta T| <= h_T/2 and |Delta C| <= h_C/2.  A sampled-max cell indicator is
therefore

    R_cell <= safety/8 * (
        M_TT h_T^2 + 2 M_TC h_T h_C + M_CC h_C^2
    ),

with M_ab the maximum sampled |q_ab| over the rectangle.

This is a *sampled Taylor-bound surrogate*, not a certified interval bound:
the maxima are taken on the dense reference surface and Delta z is represented
by first-order state sensitivities.  It does, however, treat the two input
directions jointly and includes the mixed T-Cao term explicitly.

Partitioning uses a fixed total number of rectangles.  Starting from the whole
input domain, the rectangle with the largest current indicator is split.  A
T split and a Cao split are both tested at the rectangle midpoint, and the
split that gives the smaller worst-child indicator is chosen.  This repeats
until exactly n_regions leaves remain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
    state: np.ndarray          # (nT, nC, 5): [T, Cao, Ca, Cb, Cc]
    dz_dT: np.ndarray          # same shape as state
    dz_dC: np.ndarray          # same shape as state
    hessian_norm: np.ndarray   # (nT, nC), diagnostic
    q_TT: np.ndarray           # z_T^T H z_T
    q_TC: np.ndarray           # z_T^T H z_C
    q_CC: np.ndarray           # z_C^T H z_C


@dataclass
class Rectangle:
    T_low: float
    T_high: float
    C_low: float
    C_high: float
    estimated_bound: float = np.nan
    M_TT: float = np.nan
    M_TC: float = np.nan
    M_CC: float = np.nan
    depth: int = 0

    @property
    def h_T(self) -> float:
        return float(self.T_high - self.T_low)

    @property
    def h_C(self) -> float:
        return float(self.C_high - self.C_low)

    @property
    def T_center(self) -> float:
        return 0.5 * (self.T_low + self.T_high)

    @property
    def C_center(self) -> float:
        return 0.5 * (self.C_low + self.C_high)



def _solve_checked(T: float, Cao: float, guesses: list[np.ndarray]) -> np.ndarray:
    """Solve one equilibrium point, trying continuation/fallback guesses."""

    messages: list[str] = []
    for guess in guesses:
        sol, ok, message = solve_equilibrium(
            float(T), float(Cao), np.asarray(guess, dtype=float)
        )
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
    """Fill one T row by continuation from an already-solved center point."""

    for j in range(j0 + 1, len(C_grid)):
        fallback = np.array([Cco, Cbo, C_grid[j]], dtype=float)
        solved[i, j] = _solve_checked(
            T_grid[i], C_grid[j], [solved[i, j - 1], fallback]
        )

    for j in range(j0 - 1, -1, -1):
        fallback = np.array([Cco, Cbo, C_grid[j]], dtype=float)
        solved[i, j] = _solve_checked(
            T_grid[i], C_grid[j], [solved[i, j + 1], fallback]
        )



def reaction_hessian(T: float, Ca: float, Cb: float, Cc: float) -> np.ndarray:
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
    """Solve the physical surface and build the full 2D Taylor coefficients."""

    if n_T < 5 or n_C < 5:
        raise ValueError("reference grid must have at least 5 points per axis")

    T_grid = np.linspace(Tmin, Tmax, n_T, dtype=float)
    C_grid = np.linspace(Caomin, Caomax, n_C, dtype=float)
    i0 = int(np.argmin(np.abs(T_grid - 0.5 * (Tmin + Tmax))))
    j0 = int(np.argmin(np.abs(C_grid - 0.5 * (Caomin + Caomax))))

    # solve_equilibrium returns [Cc, Cb, Ca]
    solved = np.empty((n_T, n_C, 3), dtype=float)
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

    hessian_norm = np.empty((n_T, n_C), dtype=float)
    q_TT = np.empty((n_T, n_C), dtype=float)
    q_TC = np.empty((n_T, n_C), dtype=float)
    q_CC = np.empty((n_T, n_C), dtype=float)

    for i in range(n_T):
        for j in range(n_C):
            H = reaction_hessian(T_grid[i], ca[i, j], cb[i, j], cc[i, j])
            zT = dz_dT[i, j]
            zC = dz_dC[i, j]
            hessian_norm[i, j] = np.linalg.norm(H, ord=2)
            q_TT[i, j] = float(zT @ H @ zT)
            q_TC[i, j] = float(zT @ H @ zC)
            q_CC[i, j] = float(zC @ H @ zC)

    return ReferenceSurface(
        T=T_grid,
        Cao=C_grid,
        state=state,
        dz_dT=dz_dT,
        dz_dC=dz_dC,
        hessian_norm=hessian_norm,
        q_TT=q_TT,
        q_TC=q_TC,
        q_CC=q_CC,
    )



def _cell_mask(surface: ReferenceSurface, rect: Rectangle) -> np.ndarray:
    """Reference-grid points lying inside a rectangle (boundaries included)."""

    t_mask = (surface.T >= rect.T_low) & (surface.T <= rect.T_high)
    c_mask = (surface.Cao >= rect.C_low) & (surface.Cao <= rect.C_high)
    return t_mask[:, None] & c_mask[None, :]



def sampled_cell_bound(
    rect: Rectangle,
    surface: ReferenceSurface,
    safety_factor: float = 1.10,
) -> tuple[float, float, float, float]:
    """Return sampled full-cell Taylor indicator and its three coefficients."""

    if rect.h_T <= 0.0 or rect.h_C <= 0.0:
        raise ValueError("rectangle widths must be positive")

    mask = _cell_mask(surface, rect)
    if not np.any(mask):
        # This should not occur for the default dense grid and midpoint splits,
        # but keep a robust nearest-grid fallback.
        i = int(np.argmin(np.abs(surface.T - rect.T_center)))
        j = int(np.argmin(np.abs(surface.Cao - rect.C_center)))
        mask = np.zeros_like(surface.q_TT, dtype=bool)
        mask[i, j] = True

    M_TT = float(np.max(np.abs(surface.q_TT[mask])))
    M_TC = float(np.max(np.abs(surface.q_TC[mask])))
    M_CC = float(np.max(np.abs(surface.q_CC[mask])))

    bound = safety_factor / 8.0 * (
        M_TT * rect.h_T**2
        + 2.0 * M_TC * rect.h_T * rect.h_C
        + M_CC * rect.h_C**2
    )
    return float(bound), M_TT, M_TC, M_CC



def _with_bound(
    rect: Rectangle,
    surface: ReferenceSurface,
    safety_factor: float,
) -> Rectangle:
    bound, M_TT, M_TC, M_CC = sampled_cell_bound(rect, surface, safety_factor)
    rect.estimated_bound = bound
    rect.M_TT = M_TT
    rect.M_TC = M_TC
    rect.M_CC = M_CC
    return rect



def _split_midpoint(rect: Rectangle, axis: str) -> tuple[Rectangle, Rectangle]:
    """Bisect one rectangle along T or Cao."""

    if axis == "T":
        mid = rect.T_center
        if not (rect.T_low < mid < rect.T_high):
            raise ValueError("cannot split rectangle along T")
        return (
            Rectangle(rect.T_low, mid, rect.C_low, rect.C_high, depth=rect.depth + 1),
            Rectangle(mid, rect.T_high, rect.C_low, rect.C_high, depth=rect.depth + 1),
        )
    if axis == "C":
        mid = rect.C_center
        if not (rect.C_low < mid < rect.C_high):
            raise ValueError("cannot split rectangle along Cao")
        return (
            Rectangle(rect.T_low, rect.T_high, rect.C_low, mid, depth=rect.depth + 1),
            Rectangle(rect.T_low, rect.T_high, mid, rect.C_high, depth=rect.depth + 1),
        )
    raise ValueError("axis must be 'T' or 'C'")



def _candidate_split(
    rect: Rectangle,
    axis: str,
    surface: ReferenceSurface,
    safety_factor: float,
) -> tuple[float, tuple[Rectangle, Rectangle]]:
    children = _split_midpoint(rect, axis)
    children = tuple(_with_bound(c, surface, safety_factor) for c in children)
    score = max(c.estimated_bound for c in children)
    return float(score), children



def build_rectangle_partition(
    n_regions: int,
    reference_T_points: int = 181,
    reference_C_points: int = 81,
    safety_factor: float = 1.10,
) -> tuple[list[Rectangle], ReferenceSurface]:
    """Build exactly ``n_regions`` non-Cartesian adaptive rectangles.

    Greedy rule:
      1. Find the current rectangle with the largest full-cell indicator.
      2. Test midpoint bisection along T and along Cao.
      3. Choose the split with the smaller maximum child indicator.
      4. Repeat until exactly ``n_regions`` leaves remain.
    """

    if n_regions < 1:
        raise ValueError("n_regions must be positive")
    if safety_factor <= 0.0:
        raise ValueError("safety_factor must be positive")

    surface = build_reference_surface(reference_T_points, reference_C_points)
    root = _with_bound(
        Rectangle(Tmin, Tmax, Caomin, Caomax, depth=0),
        surface,
        safety_factor,
    )
    leaves: list[Rectangle] = [root]

    while len(leaves) < n_regions:
        # Refine the rectangle with the largest current full-cell indicator.
        worst_index = int(np.argmax([r.estimated_bound for r in leaves]))
        worst = leaves[worst_index]

        score_T, children_T = _candidate_split(
            worst, "T", surface, safety_factor
        )
        score_C, children_C = _candidate_split(
            worst, "C", surface, safety_factor
        )

        # Choose the direction that most reduces the worst child indicator.
        # A deterministic tie-break uses T.
        chosen = children_T if score_T <= score_C else children_C
        leaves[worst_index:worst_index + 1] = list(chosen)

    # Stable spatial ordering makes region IDs reproducible and easy to inspect.
    leaves.sort(key=lambda r: (r.T_low, r.C_low, r.T_high, r.C_high))
    return leaves, surface



def rectangles_to_array(rectangles: list[Rectangle]) -> np.ndarray:
    """Return bounds as [T_low, T_high, C_low, C_high] rows."""

    return np.asarray(
        [[r.T_low, r.T_high, r.C_low, r.C_high] for r in rectangles],
        dtype=float,
    )
