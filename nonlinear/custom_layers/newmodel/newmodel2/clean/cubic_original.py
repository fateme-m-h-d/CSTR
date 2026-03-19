import torch
torch.set_default_dtype(torch.float64)
# ---- constants (match your models.py) ----
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


def solve_cb_from_BDH(B: torch.Tensor, D: torch.Tensor, H: torch.Tensor, eps: float = 1e-12):
    """
    Solve for Cb (unscaled) from:
        B*cb^3 - B*D*cb^2 + cb + (H - D) = 0
    and select the smallest real root with cb >= 0.

    Inputs B, D, H can be shape [N] or [N,1]. Returns:
        cb: [N,1] (NaN where no nonnegative real root found)
        ok: [N,1] boolean mask
    """
    # flatten to [N]
    B = B.reshape(-1)
    D = D.reshape(-1)
    H = H.reshape(-1)

    device = B.device
    dtype = B.dtype
    N = B.numel()

    cb = torch.full((N,), float("nan"), device=device, dtype=dtype)
    ok = torch.zeros((N,), device=device, dtype=torch.bool)

    # ---- near-linear case: B ~ 0  -> cb + (H-D) = 0 ----
    lin = B.abs() <= eps
    cb_lin = (D - H)
    lin_ok = lin & (cb_lin >= 0)
    cb[lin_ok] = cb_lin[lin_ok]
    ok[lin_ok] = True

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
            good = cb1 >= 0
            cb[idx_m[one][good]] = cb1[good]
            ok[idx_m[one][good]] = True

        # disc <= 0: three real roots (or repeated)
        three = ~one
        if three.any():
            pp = p[three]
            qq = q[three]
            sh = shift[three]
            idx3 = idx_m[three]

            # p ~ 0: t^3 + q = 0
            p0 = pp.abs() <= eps
            if p0.any():
                t = cbrt(-qq[p0])
                cb0 = t - sh[p0]
                good = cb0 >= 0
                cb[idx3[p0][good]] = cb0[good]
                ok[idx3[p0][good]] = True

            # general trig case
            gen = ~p0
            if gen.any():
                pp2 = pp[gen]
                qq2 = qq[gen]
                sh2 = sh[gen]
                idxg = idx3[gen]

                r = torch.sqrt(-(pp2 / 3.0))
                denom = (r ** 3).clamp_min(eps)
                arg = (-qq2 / 2.0) / denom
                arg = arg.clamp(-1.0, 1.0)
                phi = torch.acos(arg)

                two_r = 2.0 * r
                pi = torch.pi

                t0 = two_r * torch.cos((phi + 0.0 * 2.0 * pi) / 3.0)
                t1 = two_r * torch.cos((phi + 1.0 * 2.0 * pi) / 3.0)
                t2 = two_r * torch.cos((phi + 2.0 * 2.0 * pi) / 3.0)

                cb0 = t0 - sh2
                cb1 = t1 - sh2
                cb2 = t2 - sh2

                inf = torch.tensor(float("inf"), device=device, dtype=dtype)
                c0 = torch.where(cb0 >= 0, cb0, inf)
                c1 = torch.where(cb1 >= 0, cb1, inf)
                c2 = torch.where(cb2 >= 0, cb2, inf)
                cb_min = torch.minimum(torch.minimum(c0, c1), c2)

                good = torch.isfinite(cb_min)
                cb[idxg[good]] = cb_min[good]
                ok[idxg[good]] = True

    return cb.view(-1, 1), ok.view(-1, 1)


def solve_cb_ca_g_from_fh(T_scaled: torch.Tensor, z5: torch.Tensor, scaler):
    """
    Torch version (batch) consistent with your current model definitions:

      f_s = z5[:,3]  (scaled)
      h_s = z5[:,4]  (scaled)

      f_s = (kr/sf) * (Cao - Ca + Cbo - Cb + Cco)
      h_s = (Ca + tau*g)/sh
      g   = kf * Ca * Cb^2

    Returns:
      Ca_s, Cb_s, Cc_s, Ca_un, Cb_un, g_un, ok
    All tensors are torch, batch-shaped [N,1].
    """
    # shapes
    T_scaled = T_scaled.view(-1, 1)
    f_s = z5[:, 3:4]
    h_s = z5[:, 4:5]
    Cc_s = z5[:, 2:3]

    # scaler scales (treat as constants)
    sT  = (scaler.scale_[0])
    sCa = (scaler.scale_[1])
    sCb = (scaler.scale_[2])
    sf  = (scaler.scale_[4])
    sh  = (scaler.scale_[5])

    T = T_scaled * sT

    kf = Afo * torch.exp(-Eaf / (R * T))
    kr = Aro * torch.exp(-Ear / (R * T))

    S = Cao + Cbo + Cco

    F = (f_s * sf) / kr           # = Cao - Ca + Cbo - Cb + Cco
    D = S - F                     # = Ca + Cb
    H = h_s * sh                  # = Ca + tau*kf*Ca*Cb^2
    B = tau * kf

    Cb_un, ok = solve_cb_from_BDH(B, D, H)
    Ca_un = D - Cb_un
    g_un  = kf * Ca_un * (Cb_un ** 2)

    Ca_s = Ca_un / sCa
    Cb_s = Cb_un / sCb

    return Ca_s, Cb_s, Cc_s, Ca_un, Cb_un, g_un, ok



# test_cubic_standalone.py
import pickle
import pandas as pd
import torch

from cubic import solve_cb_ca_g_from_fh  # your torch cubic solver

SCALED_DATA_PATH = "./scaled_data.csv"   # adjust path
SCALER_PATH = "./scaler.pkl"            # adjust path

torch.set_default_dtype(torch.float64)

# load scaler
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

df = pd.read_csv(SCALED_DATA_PATH)

# --- IMPORTANT: adjust these column names if yours differ ---
# Expecting scaled columns in the saved scaled_data.csv
T_s  = torch.tensor(df["0"].values).view(-1, 1)
Ca_s = torch.tensor(df["1"].values).view(-1, 1)
Cb_s = torch.tensor(df["2"].values).view(-1, 1)
Cc_s = torch.tensor(df["3"].values).view(-1, 1)
f_s  = torch.tensor(df["4"].values).view(-1, 1)
h_s  = torch.tensor(df["5"].values).view(-1, 1)
z5 = torch.cat([Ca_s, Cb_s, Cc_s, f_s, h_s], dim=1)

Ca_s2, Cb_s2, Cc_s2, Ca_un2, Cb_un2, g_un2, ok = solve_cb_ca_g_from_fh(T_s, z5, scaler)

print("ok ratio:", ok.double().mean().item())
print("max |Ca_s2 - Ca_s|:", (Ca_s2 - Ca_s).abs().max().item())
print("max |Cb_s2 - Cb_s|:", (Cb_s2 - Cb_s).abs().max().item())

print(scaler.scale_)


print("Ca_s head:", Ca_s[:5].view(-1).tolist())
print("Ca_s2 head:", Ca_s2[:5].view(-1).tolist())
print("Cb_s head:", Cb_s[:5].view(-1).tolist())
print("Cb_s2 head:", Cb_s2[:5].view(-1).tolist())


err = (Ca_s2 - Ca_s).abs().view(-1)
idx = torch.argmax(err).item()
print("worst idx:", idx, "Ca_s:", Ca_s[idx].item(), "Ca_s2:", Ca_s2[idx].item())
