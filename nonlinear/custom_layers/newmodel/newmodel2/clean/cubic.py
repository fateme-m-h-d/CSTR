import torch
torch.set_default_dtype(torch.float64)

# ---- constants (match your models.py / generate_data.py) ----
Cao = 1.0
Cbo = 2.0
Cco = 0.0
V   = 10.0
Q   = 1.0
tau = V / Q

Afo = 10e12
Eaf = 90000.0
Aro = 10e10
Ear = 80000.0
R   = 8.314


def cbrt(x: torch.Tensor) -> torch.Tensor:
    """Real cube root (works for negative x)."""
    return torch.sign(x) * torch.abs(x).pow(1.0 / 3.0)


def cb_roots_from_BDH(B: torch.Tensor, D: torch.Tensor, H: torch.Tensor, eps: float = 1e-12):
    """
    Compute real roots of:
        B*cb^3 - B*D*cb^2 + cb + (H - D) = 0
    Returns:
        roots: [N,3] (NaN where a root slot is unused, e.g. only 1 real root case)
    """
    B = B.reshape(-1)
    D = D.reshape(-1)
    H = H.reshape(-1)
    N = B.numel()
    device, dtype = B.device, B.dtype

    roots = torch.full((N, 3), float("nan"), device=device, dtype=dtype)

    # ---- near-linear case: B ~ 0  -> cb + (H-D) = 0 ----
    lin = B.abs() <= eps
    cb_lin = (D - H)
    roots[lin, 0] = cb_lin[lin]

    # ---- cubic case ----
    m = ~lin
    if m.any():
        Bb = B[m]
        Dd = D[m]
        Hh = H[m]

        # Divide by Bb: cb^3 + A cb^2 + B1 cb + C = 0
        A = -Dd
        B1 = 1.0 / Bb
        C = (Hh - Dd) / Bb

        # Depress: cb = t - A/3
        shift = A / 3.0
        p = B1 - (A * A) / 3.0
        q = (2.0 * A * A * A) / 27.0 - (A * B1) / 3.0 + C
        disc = (q / 2.0) ** 2 + (p / 3.0) ** 3

        idx_m = torch.nonzero(m, as_tuple=True)[0]

        # disc > 0: one real root
        one = disc > eps
        if one.any():
            s = torch.sqrt(disc[one])
            t = cbrt(-q[one] / 2.0 + s) + cbrt(-q[one] / 2.0 - s)
            cb1 = t - shift[one]
            roots[idx_m[one], 0] = cb1

        # disc <= 0: three real roots (or repeated)
        three = ~one
        if three.any():
            pp = p[three]
            qq = q[three]
            sh = shift[three]
            idx3 = idx_m[three]

            # p ~ 0: t^3 + q = 0 (triple or repeated root)
            p0 = pp.abs() <= eps
            if p0.any():
                t = cbrt(-qq[p0])
                cb0 = t - sh[p0]
                roots[idx3[p0], 0] = cb0

            # general trig case
            gen = ~p0
            if gen.any():
                pp2 = pp[gen]
                qq2 = qq[gen]
                sh2 = sh[gen]
                idxg = idx3[gen]

                # r = torch.sqrt(-(pp2 / 3.0))
                # denom = (r ** 3).clamp_min(eps)
                val = -(pp2 / 3.0)
                val = torch.clamp(val, min=0.0)    # prevent sqrt of negative due to roundoff
                r = torch.sqrt(val)

                denom = torch.clamp(r ** 3, min=1e-12)  # avoid divide-by-zero / huge gradients

                arg = (-qq2 / 2.0) / denom
                arg = arg.clamp(-1.0, 1.0)
                phi = torch.acos(arg)
                # EPS_ACOS = 1e-6  # keep arg away from exactly +/-1

                # arg = (-qq2 / 2.0) / denom
                # arg = torch.nan_to_num(arg, nan=0.0, posinf=1.0 - EPS_ACOS, neginf=-1.0 + EPS_ACOS)
                # arg = arg.clamp(-1.0 + EPS_ACOS, 1.0 - EPS_ACOS)
                # phi = torch.acos(arg)


                two_r = 2.0 * r
                pi = torch.pi

                t0 = two_r * torch.cos((phi + 0.0 * 2.0 * pi) / 3.0)
                t1 = two_r * torch.cos((phi + 1.0 * 2.0 * pi) / 3.0)
                t2 = two_r * torch.cos((phi + 2.0 * 2.0 * pi) / 3.0)

                cb0 = t0 - sh2
                cb1 = t1 - sh2
                cb2 = t2 - sh2

                roots[idxg, 0] = cb0
                roots[idxg, 1] = cb1
                roots[idxg, 2] = cb2

    return roots


def pick_cb_by_cstr_residual(cb_candidates: torch.Tensor,
                            D: torch.Tensor,
                            kf: torch.Tensor,
                            kr: torch.Tensor,
                            eps: float = 1e-12):
    """
    Choose among candidate cb roots using the CSTR steady-state residuals.

    cb_candidates: [N,3] unscaled candidates (may include NaN)
    D:            [N,1] = Ca + Cb (unscaled)
    kf, kr:       [N,1] kinetics
    Returns:
        cb_best: [N,1] unscaled, always finite (fallback + clamp if needed)
        ok:      [N,1] True if a candidate with cb>=0 and ca>=0 existed
    """
    N = cb_candidates.shape[0]
    D = D.view(N, 1)
    kf = kf.view(N, 1)
    kr = kr.view(N, 1)

    cbs = cb_candidates  # [N,3]
    cas = D - cbs        # [N,3]

    # compute f (depends only on D, but keep formula explicit)
    S = Cao + Cbo + Cco
    f = kr * (S - D)     # [N,1], broadcasts against [N,3] below

    # compute g for each candidate
    g = kf * cas * (cbs ** 2)  # [N,3]

    # residuals of two balances (squared error, sign doesn’t matter)
    r1 = (cas - Cao) - tau * f + tau * g
    r2 = (cbs - Cbo) - 2.0 * tau * f + 2.0 * tau * g
    score = r1 ** 2 + r2 ** 2

    # valid physical region
    valid = torch.isfinite(cbs) & (cbs >= 0) & (cas >= 0)

    inf = torch.tensor(float("inf"), device=score.device, dtype=score.dtype)
    score_valid = torch.where(valid, score, inf)

    best = torch.argmin(score_valid, dim=1)  # [N]
    cb_best = cbs[torch.arange(N, device=cbs.device), best].view(N, 1)

    ok = torch.isfinite(torch.min(score_valid, dim=1).values).view(N, 1)

    # fallback: if no valid (cb>=0, ca>=0) candidate, pick smallest score among finite roots
    finite = torch.isfinite(cbs)
    score_finite = torch.where(finite, score, inf)
    best2 = torch.argmin(score_finite, dim=1)
    cb_fallback = cbs[torch.arange(N, device=cbs.device), best2].view(N, 1)

    cb_best = torch.where(ok, cb_best, cb_fallback)

    # final safety: enforce cb>=0 (so training never NaNs)
    cb_best = torch.clamp(cb_best, min=0.0)

    return cb_best, ok


def solve_cb_ca_g_from_fh(T_scaled: torch.Tensor, z5: torch.Tensor, scaler):
    """
    Uses z5[:,3]=f_s and z5[:,4]=h_s (scaled) to recover cb,ca analytically.

    Returns:
      Ca_s, Cb_s, Cc_s, Ca_un, Cb_un, g_un, ok
    """
    T_scaled = T_scaled.view(-1, 1)
    f_s = z5[:, 3:4]
    h_s = z5[:, 4:5]
    Cc_s = z5[:, 2:3]

    # scales
    sT  = scaler.scale_[0]
    sCa = scaler.scale_[1]
    sCb = scaler.scale_[2]
    sf  = scaler.scale_[4]
    sh  = scaler.scale_[5]

    # unscale input
    T = T_scaled * sT

    # kinetics
    kf = Afo * torch.exp(-Eaf / (R * T))
    kr = Aro * torch.exp(-Ear / (R * T))
    
    # kr = torch.clamp(kr, min=1e-12)


    # derived quantities from f and h
    S = Cao + Cbo + Cco
    F = (f_s * sf) / kr           # = Cao - Ca + Cbo - Cb + Cco
    D = S - F                     # = Ca + Cb
    H = h_s * sh                  # = Ca + tau*kf*Ca*Cb^2
    B = tau * kf

    # compute all candidate cb roots (analytic)
    cb_cands = cb_roots_from_BDH(B, D, H)

    # pick the root that matches the CSTR steady state best (and cb>=0, ca>=0)
    Cb_un, ok = pick_cb_by_cstr_residual(cb_cands, D, kf, kr)

    Ca_un = D - Cb_un
    g_un  = kf * Ca_un * (Cb_un ** 2)

    # back to scaled
    Ca_s = Ca_un / sCa
    Cb_s = Cb_un / sCb

    return Ca_s, Cb_s, Cc_s, Ca_un, Cb_un, g_un, ok


# -----------------------
# Standalone test
# -----------------------
if __name__ == "__main__":
    import pickle
    import pandas as pd

    SCALED_DATA_PATH = "./scaled_data.csv"
    SCALER_PATH = "./scaler.pkl"

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    df = pd.read_csv(SCALED_DATA_PATH)

    T_s  = torch.tensor(df["0"].values).view(-1, 1)
    Ca_s = torch.tensor(df["1"].values).view(-1, 1)
    Cb_s = torch.tensor(df["2"].values).view(-1, 1)
    Cc_s = torch.tensor(df["3"].values).view(-1, 1)
    f_s  = torch.tensor(df["4"].values).view(-1, 1)
    h_s  = torch.tensor(df["5"].values).view(-1, 1)

    z5 = torch.cat([Ca_s, Cb_s, Cc_s, f_s, h_s], dim=1)

    Ca_s2, Cb_s2, _, _, _, _, ok = solve_cb_ca_g_from_fh(T_s, z5, scaler)

    print("ok ratio:", ok.double().mean().item())
    print("max |Ca_s2 - Ca_s|:", (Ca_s2 - Ca_s).abs().max().item())
    print("max |Cb_s2 - Cb_s|:", (Cb_s2 - Cb_s).abs().max().item())

    err = (Ca_s2 - Ca_s).abs().view(-1)
    idx = torch.argmax(err).item()
    print("worst idx:", idx, "Ca_s:", Ca_s[idx].item(), "Ca_s2:", Ca_s2[idx].item(),
          "Cb_s:", Cb_s[idx].item(), "Cb_s2:", Cb_s2[idx].item())
