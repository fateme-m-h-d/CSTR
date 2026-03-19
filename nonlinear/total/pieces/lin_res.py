import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sym
from sympy import symbols

# === 1) Constants ===
Cao, Cbo, Cco = 1.0, 2.0, 0.0
V, Q = 10.0, 1.0
tau = V / Q

Afo, Eaf = 1e13, 90000.0
Aro, Ear = 1e11, 80000.0
R = 8.314

# === 2) Symbolic nonlinear residual f(Ca,Cb,T) and its symbolic gradients ===
Ca, Cb, T = symbols('Ca Cb T', real=True)

kf = Afo * sym.exp(-Eaf/(R*T))
kr = Aro * sym.exp(-Ear/(R*T))

f = (Cao - Ca
     - kf*Ca*Cb**2 * tau
     + kr*(Cao - Ca + Cbo - Cb + Cco) * tau)

dfdCa = sym.diff(f, Ca)
dfdCb = sym.diff(f, Cb)
dfdT  = sym.diff(f, T)

# Lambdify nonlinear f once
f_nl_func = sym.lambdify((Ca, Cb, T), f, 'numpy')

# === 3) Your piecewise center points (T-range -> (Cass,Cbss,Tss)) ===
segments = [
    # (Tmin, Tmax, Cass, Cbss, Tss)
    (280, 300, 0.978245953477923, 1.95649190695584, 289.64824120603),
    (300, 340, 0.737146877322096, 1.47429375464419, 320.201005025125),
    (340, 360, 0.52075235837267, 1.04150471674534, 350.753768844221),
    (360, 400,0.462575426509417, 0.925150853018835, 379.698492462311),
    (400, 460, 0.41589194923686623, 0.8317838984736275, 430.06012024048096),
    (460, 500, 0.3839514142818392,  0.767902828563119,  480.0801603206413),
    (500, 530, 0.3660794237985876,  0.7321588475969961, 515.3507014028056),
    (530, 550, 0.3551476226232751,  0.7102952452401861, 540.3607214428857),
    (550, 565, 0.34828559395504716, 0.6965711879167573, 557.6753507014027),
    (565, 578, 0.34307151217593673, 0.686143024351111,  571.7835671342685),
    (578, 590, 0.3385990203254893,  0.6771980406362342, 584.6092184368738),
    (590, 600, 0.3349827056400738,  0.6699654112978638, 595.5110220440881),
]

# === 4) Load data ===
df = pd.read_csv("data.csv")
T_data  = df['Temperature (T)'].to_numpy()
Ca_data = df['Ca'].to_numpy()
Cb_data = df['Cb'].to_numpy()

# === 5) Nonlinear residual for all points ===
res_nl = f_nl_func(Ca_data, Cb_data, T_data)

# === 6) Piecewise linear residual (use correct linearization per range) ===
res_lin_piecewise = np.full_like(T_data, np.nan, dtype=float)

for (Tmin, Tmax, Cass, Cbss, Tss) in segments:
    # mask for this segment
    # (left-inclusive, right-exclusive) except last segment can be inclusive
    mask = (T_data >= Tmin) & (T_data < Tmax)

    # If you want the upper endpoint included for the final segment:
    # mask = (T_data >= Tmin) & (T_data <= Tmax)  # use this only on the last segment

    # numeric values at the segment center
    subs_dict = {Ca: Cass, Cb: Cbss, T: Tss}

    fss   = float(sym.N(f.subs(subs_dict)))
    aCa   = float(sym.N(dfdCa.subs(subs_dict)))
    aCb   = float(sym.N(dfdCb.subs(subs_dict)))
    aT    = float(sym.N(dfdT.subs(subs_dict)))

    # linear residual: f_lin = fss + aCa*(Ca-Cass) + aCb*(Cb-Cbss) + aT*(T-Tss)
    res_lin_piecewise[mask] = (
        fss
        + aCa*(Ca_data[mask] - Cass)
        + aCb*(Cb_data[mask] - Cbss)
        + aT *(T_data[mask]  - Tss)
    )

# Warn if anything not covered by segments
not_covered = np.isnan(res_lin_piecewise)
if np.any(not_covered):
    print("WARNING: Some points are outside the defined segments.")
    print("Example T values not covered:", np.unique(T_data[not_covered])[:10])


# === 7) ABSOLUTE RESIDUALS ===
abs_res_nl  = np.abs(res_nl)
abs_res_lin = np.abs(res_lin_piecewise)

order = np.argsort(T_data)
plt.figure(figsize=(7,4))
plt.plot(T_data[order], abs_res_nl[order], 'o', markersize=3, label="Nonlinear")
plt.plot(T_data[order], abs_res_lin[order], 'x', markersize=3, label="Piecewise linearized")
plt.axhline(0, linewidth=0.5)
plt.xlabel("Temperature T (K)")
plt.ylabel("Residual")
plt.yscale("log")
plt.title("Residuals vs Temperature (all ranges)")
plt.legend()
plt.tight_layout()
plt.show()

# === Option B (often best): plot the absolute linearization error |f_lin - f| vs T ===
abs_err = np.abs(res_lin_piecewise - res_nl)
plt.figure(figsize=(7,4))
plt.plot(T_data[order], abs_err[order], 'o', markersize=3)
plt.xlabel("Temperature T (K)")
plt.ylabel("|f_lin - f|")
plt.title("Absolute linearization error vs Temperature (all ranges)")
plt.tight_layout()
plt.show()