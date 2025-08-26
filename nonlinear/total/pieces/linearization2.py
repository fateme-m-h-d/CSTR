# taylor_auto_segments.py
# Piecewise first-order Taylor linearization on ORIGINAL data (no 1/T).
# Chooses #segments by a max-error tolerance. No least-squares is used.

import numpy as np
from typing import Callable, Dict, Optional, List

# EXACT same constants you use in linearization.py
Cao, Cbo, Cco = 1.0, 2.0, 0.0     # feed concentrations (mol/L)
V, Q = 10.0, 1.0                  # volume, flow
tau = V/Q

Afo, Eaf = 1e13, 90000.0          # forward Arrhenius
Aro, Ear = 1e11, 80000.0          # reverse Arrhenius
R = 8.314   
# ====== 1) Provide your nonlinear residual f(T, Ca, Cb, Cc) here ======
def residual_fn(T, Ca, Cb, Cc):
    """
    Nonlinear residual f(T, Ca, Cb, Cc) to be linearized.
    Copy the SAME formula you use for f in linearization.py.
    """
    T  = np.asarray(T,  float)
    Ca = np.asarray(Ca, float)
    Cb = np.asarray(Cb, float)
    # Cc is unused in this particular formula; keep it for signature consistency

    kf = Afo * np.exp(-Eaf/(R*T))
    kr = Aro * np.exp(-Ear/(R*T))

    f = (Cao - Ca) \
        - kf * Ca * (Cb**2) * tau \
        + kr * ((Cao - Ca) + (Cbo - Cb) + Cco) * tau
    return f


# (Optional) analytic derivatives at a scalar point; supply any subset you have.
# grads = {
#   "dfdT":  lambda T, Ca, Cb, Cc: ...,
#   "dfdCa": lambda T, Ca, Cb, Cc: ...,
#   "dfdCb": lambda T, Ca, Cb, Cc: ...,
#   "dfdCc": lambda T, Ca, Cb, Cc: ...,
# }
grads: Dict[str, Callable] = {}


# ====== 2) Taylor at a center point (uses analytic grads if given, else finite diff) ======
def _central_diff(f1d: Callable[[float], float], x0: float, h: float) -> float:
    return (f1d(x0 + h) - f1d(x0 - h)) / (2.0 * h)

def _taylor_coeffs_at_center(Tc: float, Ca_c: float, Cb_c: float, Cc_c: float,
                             hT: float = 1e-6, hC: float = 1e-6) -> Dict[str, float]:
    f0 = float(residual_fn(np.array([Tc]), np.array([Ca_c]), np.array([Cb_c]), np.array([Cc_c]))[0])

    # d/dT
    if "dfdT" in grads:
        A = float(grads["dfdT"](Tc, Ca_c, Cb_c, Cc_c))
    else:
        A = _central_diff(lambda t: residual_fn(np.array([t]), np.array([Ca_c]),
                                                np.array([Cb_c]), np.array([Cc_c]))[0], Tc, hT)
    # d/dCa
    if "dfdCa" in grads:
        B1 = float(grads["dfdCa"](Tc, Ca_c, Cb_c, Cc_c))
    else:
        B1 = _central_diff(lambda ca: residual_fn(np.array([Tc]), np.array([ca]),
                                                  np.array([Cb_c]), np.array([Cc_c]))[0], Ca_c, hC)
    # d/dCb
    if "dfdCb" in grads:
        B2 = float(grads["dfdCb"](Tc, Ca_c, Cb_c, Cc_c))
    else:
        B2 = _central_diff(lambda cb: residual_fn(np.array([Tc]), np.array([Ca_c]),
                                                  np.array([cb]), np.array([Cc_c]))[0], Cb_c, hC)
    # d/dCc (optional)
    if "dfdCc" in grads:
        B3 = float(grads["dfdCc"](Tc, Ca_c, Cb_c, Cc_c))
    else:
        # if Cc is unused in your constraint, keep zero so format stays A, [B1,B2,B3], b
        B3 = 0.0

    # Convert Taylor form to constraint form:  g = A*T + B·Y - b
    # f_lin = f0 + A(T-Tc) + B1(Ca-Ca_c) + B2(Cb-Cb_c) + B3(Cc-Cc_c)
    #       = A*T + B1*Ca + B2*Cb + B3*Cc - [A*Tc + B1*Ca_c + B2*Cb_c + B3*Cc_c - f0]
    b = A*Tc + B1*Ca_c + B2*Cb_c + B3*Cc_c - f0
    return dict(A=A, B1=B1, B2=B2, B3=B3, b=b, f0=f0, Tc=Tc, Ca_c=Ca_c, Cb_c=Cb_c, Cc_c=Cc_c)


# ====== 3) Fit a segment by Taylor at its center and measure max error ======
def _segment_taylor(T: np.ndarray, Ca: np.ndarray, Cb: np.ndarray, Cc: np.ndarray,
                    center: str = "median") -> Dict[str, float]:
    assert len(T) == len(Ca) == len(Cb) == len(Cc)
    idx = np.argsort(T)
    T, Ca, Cb, Cc = T[idx], Ca[idx], Cb[idx], Cc[idx]

    # choose center point from the data in this segment
    if center == "median":
        k = len(T)//2
    elif center == "mean":
        # choose actual sample closest to the mean T
        Tmean = T.mean()
        k = int(np.argmin(np.abs(T - Tmean)))
    else:
        raise ValueError("center must be 'median' or 'mean'")

    coeffs = _taylor_coeffs_at_center(T[k], Ca[k], Cb[k], Cc[k])

    # evaluate linearized approximation on all points in this segment
    A, B1, B2, B3, b = coeffs["A"], coeffs["B1"], coeffs["B2"], coeffs["B3"], coeffs["b"]
    f_true = residual_fn(T, Ca, Cb, Cc)
    f_lin  = A*T + B1*Ca + B2*Cb + B3*Cc - b
    err    = np.abs(f_true - f_lin)

    return dict(A=A, B1=B1, B2=B2, B3=B3, b=b,
                center_index=k, max_err=float(err.max()),
                mean_err=float(err.mean()))


# ====== 4) Adaptive splitting in T using a max-error tolerance ======
def taylor_adaptive_segments(T: np.ndarray, Ca: np.ndarray, Cb: np.ndarray, Cc: np.ndarray,
                             tol: float = 1e-2, min_pts: int = 25, max_depth: int = 12,
                             center: str = "median") -> List[dict]:
    """
    Recursively split a T range until first-order Taylor in each segment
    meets max |f - f_lin| <= tol on the ORIGINAL data points.

    Returns a list of segments with fields:
      T_lo, T_hi, idx_lo, idx_hi, A, B (len-3), b, max_err, mean_err.
    """
    order = np.argsort(T)
    T, Ca, Cb, Cc = T[order], Ca[order], Cb[order], Cc[order]

    segments: List[dict] = []

    def recurse(i0: int, i1: int, depth: int):
        n = i1 - i0 + 1
        if n < max(min_pts, 3) or depth == 0:
            stats = _segment_taylor(T[i0:i1+1], Ca[i0:i1+1], Cb[i0:i1+1], Cc[i0:i1+1], center=center)
            segments.append(dict(T_lo=float(T[i0]), T_hi=float(T[i1]),
                                 idx_lo=i0, idx_hi=i1,
                                 A=stats["A"], B=np.array([stats["B1"], stats["B2"], stats["B3"]]),
                                 b=stats["b"], max_err=stats["max_err"], mean_err=stats["mean_err"],
                                 n_points=n))
            return

        stats = _segment_taylor(T[i0:i1+1], Ca[i0:i1+1], Cb[i0:i1+1], Cc[i0:i1+1], center=center)
        if stats["max_err"] <= tol:
            segments.append(dict(T_lo=float(T[i0]), T_hi=float(T[i1]),
                                 idx_lo=i0, idx_hi=i1,
                                 A=stats["A"], B=np.array([stats["B1"], stats["B2"], stats["B3"]]),
                                 b=stats["b"], max_err=stats["max_err"], mean_err=stats["mean_err"],
                                 n_points=n))
            return

        # split at the sample with the worst error (compute it once)
        A, B1, B2, B3, b = stats["A"], stats["B1"], stats["B2"], stats["B3"], stats["b"]
        f_true = residual_fn(T[i0:i1+1], Ca[i0:i1+1], Cb[i0:i1+1], Cc[i0:i1+1])
        f_lin  = A*T[i0:i1+1] + B1*Ca[i0:i1+1] + B2*Cb[i0:i1+1] + B3*Cc[i0:i1+1] - b
        worst_local = int(np.argmax(np.abs(f_true - f_lin)))
        j = i0 + worst_local
        if j <= i0 or j >= i1:
            j = (i0 + i1)//2  # safety

        recurse(i0, j,      depth-1)
        recurse(j+1, i1,    depth-1)

    recurse(0, len(T)-1, max_depth)
    segments.sort(key=lambda s: s["T_lo"])
    return segments


# ====== 5) Example usage (wire to your data) ======
if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt

    # Load original data (columns: T, Ca, Cb, optional Cc)
    df = pd.read_csv("data.csv")
    T  = df["Temperature (T)"].to_numpy(float)
    Ca = df["Ca"].to_numpy(float)
    Cb = df["Cb"].to_numpy(float)
    Cc = df["Cc"].to_numpy(float) if "Cc" in df.columns else np.zeros_like(T)

    # Run adaptive Taylor
    segs = taylor_adaptive_segments(T, Ca, Cb, Cc,
                                    tol=1e-2,       # tighten for more segments
                                    min_pts=25,
                                    max_depth=12,
                                    center="median")

    # Print and save
    rows = []
    for i, s in enumerate(segs, 1):
        print(f"Region {i:2d}: {s['T_lo']:.1f}–{s['T_hi']:.1f} K | "
              f"max_err={s['max_err']:.3e} | A={s['A']:.6g} | "
              f"B=[{s['B'][0]:.6g}, {s['B'][1]:.6g}, {s['B'][2]:.6g}] | b={s['b']:.6g}")
        rows.append({
            "region": i, "T_lo": s["T_lo"], "T_hi": s["T_hi"],
            "A": s["A"], "B_Ca": s["B"][0], "B_Cb": s["B"][1], "B_Cc": s["B"][2], "b": s["b"],
            "max_err": s["max_err"], "mean_err": s["mean_err"], "n_points": s["n_points"]
        })
    pd.DataFrame(rows).to_csv("segments_taylor_auto.csv", index=False)

    # Quick visual: error on original data with accepted boundaries
    f = residual_fn(T, Ca, Cb, Cc)
    err = np.zeros_like(f)
    for s in segs:
        A, B, b = s["A"], s["B"], s["b"]
        mask = (T >= s["T_lo"]) & (T <= s["T_hi"])
        f_lin = A*T[mask] + B[0]*Ca[mask] + B[1]*Cb[mask] + B[2]*Cc[mask] - b
        err[mask] = np.abs(f[mask] - f_lin)
    plt.figure(figsize=(9,4))
    for s in segs:
        plt.axvline(s["T_lo"], color="gray", lw=0.6, ls="--")
    plt.axvline(segs[-1]["T_hi"], color="gray", lw=0.6, ls="--")
    plt.scatter(T, err, s=10)
    plt.yscale("log"); plt.xlabel("T (K)"); plt.ylabel("|f - f_lin| (raw)")
    plt.title("Taylor error on original data")
    plt.tight_layout(); plt.show()
