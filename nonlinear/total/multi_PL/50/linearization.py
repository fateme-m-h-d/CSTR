import numpy as np
import pandas as pd
import sympy as sym
from scipy.optimize import fsolve
import matplotlib
matplotlib.use("Agg")  # headless backend (no window)
import matplotlib.pyplot as plt
import argparse


np_dtype = np.float32

# ============================================================
# 0) CONSTANTS
# ============================================================
V = 10.0      # L
Q = 1.0       # L/s
tau = V / Q   # s

# Arrhenius parameters (aggressive nonlinearity)
Afo = 10e12
Eaf = 90000.0   # J/mol
Aro = 10e10
Ear = 80000.0   # J/mol
R   = 8.314      # J/mol/K

# feed concentrations
Cbo = 2.0        # mol/L
Cco = 0.0        # mol/L

XTOL = 1e-11

# ============================================================
# 1) REGION DEFINITION (you choose these)
# ============================================================
Tmin,  Tmax  = 280.0, 460.0
Caomin, Caomax = 0.8, 1.2

# How many extra random interior points per region
# n_inner_per_region = 2
# seed = 0
parser = argparse.ArgumentParser()
parser.add_argument("--n_inner_per_region", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

n_inner_per_region = args.n_inner_per_region
seed = args.seed

# T_edges  = np.linspace(Tmin, Tmax, nT_regions + 1)
# T_edges = np.array([280, 300, 340, 360, 400, 460, 500, 530, 550, 565, 578, 590, 600], float)
T_edges = np.array([280, 300, 340, 360, 400, 420, 440, 460], float)
C_edges  = np.linspace(Caomin, Caomax, 4)
print("T_edges:", T_edges)
print("Cao_edges:", C_edges)
nT_regions = len(T_edges) - 1
nC_regions = len(C_edges) - 1
T_centers = 0.5 * (T_edges[:-1] + T_edges[1:])
C_centers = 0.5 * (C_edges[:-1] + C_edges[1:])

# ============================================================
# 2) SMART SAMPLING: corners + center + edge-midpoints + random
# ============================================================
def build_sampling_points(T_edges, C_edges, n_inner, seed, include_edge_midpoints=True):
    rng = np.random.default_rng(seed)
    pts = []

    for i in range(len(T_edges) - 1):
        for j in range(len(C_edges) - 1):
            T0, T1 = float(T_edges[i]), float(T_edges[i+1])
            C0, C1 = float(C_edges[j]), float(C_edges[j+1])
            Tc, Cc = 0.5*(T0+T1), 0.5*(C0+C1)

            # corners (bounds) + center
            pts += [(T0,C0),(T0,C1),(T1,C0),(T1,C1),(Tc,Cc)]

            # optional: midpoints on edges (helps blending across boundaries)
            if include_edge_midpoints:
                pts += [(Tc,C0),(Tc,C1),(T0,Cc),(T1,Cc)]

            # random interior points
            Tr = rng.uniform(T0, T1, size=n_inner)
            Cr = rng.uniform(C0, C1, size=n_inner)
            pts += list(zip(Tr, Cr))

    pts = np.array(pts, dtype=np_dtype)

    # remove duplicates (edges shared between regions) by rounding then unique
    pts = np.unique(np.round(pts, 12), axis=0)
    return pts[:, 0], pts[:, 1]

T_samp, Cao_samp = build_sampling_points(T_edges, C_edges, n_inner=n_inner_per_region, seed=seed, include_edge_midpoints=False)

# ============================================================
# 3) NONLINEAR STEADY-STATE SOLVER (solve Ca,Cb,Cc at each sample)
# ============================================================
def equations(vars_, T, Cao):
    """
    Unknowns are ordered as (Cc, Cb, Ca) to match your earlier code.
    """
    Cc, Cb, Ca = vars_
    kf = Afo * np.exp(-Eaf / (R * T))
    kr = Aro * np.exp(-Ear / (R * T))

    # CSTR balances (with reverse rate proportional to Cc)
    eq1 = Cao - Ca - kf * Ca * (Cb**2) * tau + kr * Cc * tau
    eq2 = Cbo - Cb - 2.0 * kf * Ca * (Cb**2) * tau + 2.0 * kr * Cc * tau

    # stoichiometry / component relation (consistent with your (Cao - Ca + Cbo - Cb + Cco))
    eq3 = Cc - (Cao - Ca + Cbo - Cb + Cco)

    return [eq1, eq2, eq3]

def solve_equilibrium(T, Cao, guess):
    sol, info, ier, mesg = fsolve(
        equations, guess, args=(T, Cao),
        full_output=True, xtol=XTOL
    )
    return sol, (ier == 1), mesg

# ============================================================
# 4) REGION ID ASSIGNMENT FOR EACH POINT
# ============================================================
def assign_region_id(T, Cao, T_edges, C_edges, nC_regions):
    iT = np.digitize(T, T_edges) - 1
    iC = np.digitize(Cao, C_edges) - 1
    iT = np.clip(iT, 0, len(T_edges)-2)
    iC = np.clip(iC, 0, len(C_edges)-2)
    return iT * nC_regions + iC, iT, iC

# ============================================================
# 5) SOLVE DATA REGION-BY-REGION (center first, then rest)
# ============================================================
rows = []
fail_rows = []

for iT in range(nT_regions):
    for iC in range(nC_regions):
        rid = iT * nC_regions + iC

        # region bounds + center
        T0, T1 = float(T_edges[iT]), float(T_edges[iT+1])
        C0, C1 = float(C_edges[iC]), float(C_edges[iC+1])
        Tc, Cc_ = 0.5*(T0+T1), 0.5*(C0+C1)

        # collect all sample points that fall in this region
        mask_T = (T_samp >= T0 - 1e-12) & (T_samp <= T1 + 1e-12)
        mask_C = (Cao_samp >= C0 - 1e-12) & (Cao_samp <= C1 + 1e-12)
        mask = mask_T & mask_C

        pts_region = np.column_stack([T_samp[mask], Cao_samp[mask]])

        # solve center first (guaranteed to exist)
        guess0 = np.array([Cco, Cbo, Cc_], dtype=np_dtype)  # (Cc,Cb,Ca) initial guess
        sol_center, ok, mesg = solve_equilibrium(Tc, Cc_, guess0)
        if not ok:
            print(f"[Region {rid}] CENTER solve failed at (T={Tc:.3f}, Cao={Cc_:.3f}): {mesg}")
            continue

        Cc_ss, Cb_ss, Ca_ss = sol_center

        rows.append({
            "region_id": rid, "iT": iT, "iC": iC,
            "Temperature (T)": Tc, "Cao": Cc_,
            "Ca": Ca_ss, "Cb": Cb_ss, "Cc": Cc_ss,
            "is_center": 1
        })

        # solve the rest of region points, warm-start from center then previous
        # sort by distance from center for stable warm-start
        d2 = (pts_region[:,0] - Tc)**2 + (pts_region[:,1] - Cc_)**2
        order = np.argsort(d2)
        pts_region = pts_region[order]

        guess = sol_center.copy()
        for (Tpt, Cpt) in pts_region:
            # skip the center point (already added)
            if abs(Tpt - Tc) < 1e-12 and abs(Cpt - Cc_) < 1e-12:
                continue

            sol, ok, mesg = solve_equilibrium(float(Tpt), float(Cpt), guess)
            if ok:
                Cc_sol, Cb_sol, Ca_sol = sol
                rows.append({
                    "region_id": rid, "iT": iT, "iC": iC,
                    "Temperature (T)": float(Tpt), "Cao": float(Cpt),
                    "Ca": Ca_sol, "Cb": Cb_sol, "Cc": Cc_sol,
                    "is_center": 0
                })
                guess = sol  # warm-start next
            else:
                fail_rows.append((rid, float(Tpt), float(Cpt), mesg))
                # keep old guess (center/previous)

data = pd.DataFrame(rows)
data.to_csv("data_smart.csv", index=False)
print(f"\nSaved {len(data)} solved points to data_smart.csv")
if fail_rows:
    print(f"WARNING: {len(fail_rows)} points failed to converge (not saved).")

if fail_rows:
    fail_df = pd.DataFrame(fail_rows, columns=["region_id", "T", "Cao", "message"])
    fail_df.to_csv("failed_points.csv", index=False)
    print("Saved failed points to failed_points.csv")

data_out = (
    data.rename(columns={"Temperature (T)": "Temperature (T)"})  # keeps same name; optional
        [["Temperature (T)", "Cao", "Ca", "Cb", "Cc"]]
        .sort_values(["Temperature (T)", "Cao"], ascending=[True, True])
        .reset_index(drop=True)
)

data_out.to_csv("data.csv", index=False)
print("Saved data.csv with columns:", data_out.columns.tolist())
# ============================================================
# 6) LINEARIZE ONLY THIS EQUATION f(T,Cao,Ca,Cb)=0 PER REGION
#     f = Cao - Ca - kf*Ca*Cb^2*tau + kr*(Cao - Ca + Cbo - Cb + Cco)*tau
# ============================================================
T_sym, Cao_sym, Ca_sym, Cb_sym = sym.symbols("T Cao Ca Cb", real=True)

kf_sym = sym.Float(Afo) * sym.exp(-sym.Float(Eaf) / (sym.Float(R) * T_sym))
kr_sym = sym.Float(Aro) * sym.exp(-sym.Float(Ear) / (sym.Float(R) * T_sym))

f_sym = (Cao_sym - Ca_sym
         - kf_sym * Ca_sym * (Cb_sym**2) * sym.Float(tau)
         + kr_sym * (Cao_sym - Ca_sym + sym.Float(Cbo) - Cb_sym + sym.Float(Cco)) * sym.Float(tau))

df_Ca_sym  = sym.diff(f_sym, Ca_sym)
df_Cb_sym  = sym.diff(f_sym, Cb_sym)
df_T_sym   = sym.diff(f_sym, T_sym)
df_Cao_sym = sym.diff(f_sym, Cao_sym)

# lambdify for fast numeric evaluation
f_fun      = sym.lambdify((T_sym, Cao_sym, Ca_sym, Cb_sym), f_sym, "numpy")
df_Ca_fun  = sym.lambdify((T_sym, Cao_sym, Ca_sym, Cb_sym), df_Ca_sym, "numpy")
df_Cb_fun  = sym.lambdify((T_sym, Cao_sym, Ca_sym, Cb_sym), df_Cb_sym, "numpy")
df_T_fun   = sym.lambdify((T_sym, Cao_sym, Ca_sym, Cb_sym), df_T_sym, "numpy")
df_Cao_fun = sym.lambdify((T_sym, Cao_sym, Ca_sym, Cb_sym), df_Cao_sym, "numpy")

# ============================================================
# 7) PRINT LINEARIZED EQUATION FOR EACH REGION (around center)
# ============================================================
print("\n" + "="*80)
print("Linearization of f(T,Cao,Ca,Cb)=0 in each region (around region center steady-state)")
print("="*80)

region_centers = data[data["is_center"] == 1].copy()
region_centers = region_centers.sort_values(["region_id"])

for _, row in region_centers.iterrows():
    rid = int(row["region_id"])

    Tss   = float(row["Temperature (T)"])
    Caoss = float(row["Cao"])
    Cass  = float(row["Ca"])
    Cbss  = float(row["Cb"])

    fss    = float(f_fun(Tss, Caoss, Cass, Cbss))
    aCa    = float(df_Ca_fun(Tss, Caoss, Cass, Cbss))
    aCb    = float(df_Cb_fun(Tss, Caoss, Cass, Cbss))
    aT     = float(df_T_fun(Tss, Caoss, Cass, Cbss))
    aCao   = float(df_Cao_fun(Tss, Caoss, Cass, Cbss))

    # Taylor form:
    # 0 ≈ fss + aCa*(Ca-Cass) + aCb*(Cb-Cbss) + aT*(T-Tss) + aCao*(Cao-Caoss)
    #
    # Rearranged scalar linear equation:
    # aT*T + aCao*Cao + aCa*Ca + aCb*Cb = b
    b = (-fss + aCa*Cass + aCb*Cbss + aT*Tss + aCao*Caoss)

    print(f"\n[Region {rid}]  Center: Tss={Tss:.4f}, Caoss={Caoss:.4f}, Cass={Cass:.6f}, Cbss={Cbss:.6f}")
    print(f"  fss = {fss:.6e}")
    # print("  Taylor (first-order):")
    # print(f"    0 ≈ {fss:.6e}"
    #       f" + ({aCa:.6e})*(Ca - {Cass:.6e})"
    #       f" + ({aCb:.6e})*(Cb - {Cbss:.6e})"
    #       f" + ({aT:.6e})*(T - {Tss:.6e})"
    #       f" + ({aCao:.6e})*(Cao - {Caoss:.6e})")

    print("  Rearranged linear form:")
    print(f"    ({aT:.6e})*T + ({aCao:.6e})*Cao + ({aCa:.6e})*Ca + ({aCb:.6e})*Cb = {b:.6e}")

print("\nDone.")

# --- store linearization params per region_id ---
lin_params = []  # list of dicts

for _, row in region_centers.iterrows():
    rid = int(row["region_id"])
    Tss   = float(row["Temperature (T)"])
    Caoss = float(row["Cao"])
    Cass  = float(row["Ca"])
    Cbss  = float(row["Cb"])

    fss    = float(f_fun(Tss, Caoss, Cass, Cbss))
    aCa    = float(df_Ca_fun(Tss, Caoss, Cass, Cbss))
    aCb    = float(df_Cb_fun(Tss, Caoss, Cass, Cbss))
    aT     = float(df_T_fun(Tss, Caoss, Cass, Cbss))
    aCao   = float(df_Cao_fun(Tss, Caoss, Cass, Cbss))

    lin_params.append({
        "region_id": rid,
        "Tss": Tss, "Caoss": Caoss, "Cass": Cass, "Cbss": Cbss,
        "fss": fss, "aCa": aCa, "aCb": aCb, "aT": aT, "aCao": aCao
    })

lin_df = pd.DataFrame(lin_params)
lin_df.to_csv("lin_params.csv", index=False)

import matplotlib.pyplot as plt

df = pd.read_csv("data_smart.csv")      # solved points
lin = pd.read_csv("lin_params.csv")     # region center linearization coefficients (you saved earlier)

df = df.merge(lin, on="region_id", how="left")

T   = df["Temperature (T)"].to_numpy()
Cao = df["Cao"].to_numpy()
Ca  = df["Ca"].to_numpy()
Cb  = df["Cb"].to_numpy()

# ---- f_nl (nonlinear) ----
kf = Afo * np.exp(-Eaf/(R*T))
kr = Aro * np.exp(-Ear/(R*T))
f_nl = (Cao - Ca
        - kf*Ca*(Cb**2)*tau
        + kr*(Cao - Ca + Cbo - Cb + Cco)*tau)

# ---- f_lin using region center coefficients ----
f_lin = (df["fss"].to_numpy()
         + df["aCa"].to_numpy()*(Ca - df["Cass"].to_numpy())
         + df["aCb"].to_numpy()*(Cb - df["Cbss"].to_numpy())
         + df["aT"].to_numpy() *(T  - df["Tss"].to_numpy())
         + df["aCao"].to_numpy()*(Cao - df["Caoss"].to_numpy()))

df["f_nl"] = f_nl
df["f_lin"] = f_lin
df["abs_f_lin"] = np.abs(f_lin)

print("Sanity: max |f_nl| =", np.nanmax(np.abs(f_nl)))   # should be tiny for solved points

# ---- Region summary: how good is the linearization inside each region ----
summary = df.groupby("region_id")["abs_f_lin"].agg(
    n="count",
    mean_abs="mean",
    max_abs="max",
    p95_abs=lambda s: np.percentile(s, 95)
).reset_index().sort_values("max_abs", ascending=False)

summary.to_csv("lin_accuracy_by_region.csv", index=False)
print(summary.head(10))

# ---- Optional: visualize worst region ----
worst = int(summary.iloc[0]["region_id"])
sub = df[df["region_id"] == worst]

plt.figure()
plt.hist(sub["abs_f_lin"], bins=30)
plt.title(f"Region {worst}: |f_lin| distribution")
plt.xlabel("|f_lin|")
plt.ylabel("count")
plt.savefig(f"worst Region {worst}: |f_lin| distribution", dpi=300, bbox_inches="tight")
plt.close()


# --- Build A, B, b for Ax + By = b ---
# x = [T, Cao]
# y = [Ca, Cb, Cc]  (Cc coefficient is zero for this constraint)

AB_rows = []
for _, r in lin_df.iterrows():
    aT   = float(r["aT"])
    aCao = float(r["aCao"])
    aCa  = float(r["aCa"])
    aCb  = float(r["aCb"])

    # compute b for this region (same as before)
    b_val = (-float(r["fss"])
             + aCa  * float(r["Cass"])
             + aCb  * float(r["Cbss"])
             + aT   * float(r["Tss"])
             + aCao * float(r["Caoss"]))

    AB_rows.append({
        "region_id": int(r["region_id"]),
        "A_T": aT,
        "A_Cao": aCao,
        "B_Ca": aCa,
        "B_Cb": aCb,
        "B_Cc": 0.0,
        "b": b_val
    })

AB_df = pd.DataFrame(AB_rows).sort_values("region_id").reset_index(drop=True)
AB_df.to_csv("ABb_matrices.csv", index=False)
print("Saved ABb_matrices.csv")

A = AB_df[["A_T", "A_Cao"]].to_numpy(dtype=np_dtype)          # (30, 2)
B = AB_df[["B_Ca", "B_Cb", "B_Cc"]].to_numpy(dtype=np_dtype)  # (30, 3)
b_vec = AB_df[["b"]].to_numpy(dtype=np_dtype)                # (30, 1)

print("A shape:", A.shape)
print("B shape:", B.shape)
print("b shape:", b_vec.shape)

import torch

torch_dtype = torch.float32  # or torch.float64 if you prefer higher precision

A_torch = torch.tensor(A, dtype=torch_dtype)
B_torch = torch.tensor(B, dtype=torch_dtype)
b_torch = torch.tensor(b_vec, dtype=torch_dtype)

# # if you want to paste a literal Python list into your model file:
# print("A_list =", A_torch.tolist())
# print("B_list =", B_torch.tolist())
# print("b_list =", b_torch.tolist())


import pandas as pd
import torch

# Read the linearization coefficients you already saved
lin_df = pd.read_csv("lin_params.csv").sort_values("region_id").reset_index(drop=True)

# Compute b for each region (same formula you used when printing)
lin_df["b"] = (-lin_df["fss"]
               + lin_df["aCa"]*lin_df["Cass"]
               + lin_df["aCb"]*lin_df["Cbss"]
               + lin_df["aT"] *lin_df["Tss"]
               + lin_df["aCao"]*lin_df["Caoss"])

# -------- Print A_list (1x2 each) --------
print("self.A_list = [")
for _, r in lin_df.iterrows():
    aT   = float(r["aT"])
    aCao = float(r["aCao"])
    print(f"    torch.tensor([[{aT:.15e}, {aCao:.15e}]], dtype=torch.float32),")
print("]")

# -------- Print B_list (1x3 each) --------
print("\nself.B_list = [")
for _, r in lin_df.iterrows():
    aCa = float(r["aCa"])
    aCb = float(r["aCb"])
    print(f"    torch.tensor([[{aCa:.15e}, {aCb:.15e}, {0.0:.15e}]], dtype=torch.float32),")
print("]")

# -------- Print b_list (1x1 each) --------
print("\nself.b_list = [")
for _, r in lin_df.iterrows():
    b = float(r["b"])
    print(f"    torch.tensor([[{b:.15e}]], dtype=torch.float32),")
print("]")


# plots

plt.figure(figsize=(8,6))
plt.scatter(df["Temperature (T)"], df["Cao"], s=10, label="solved points")

# draw region boundaries
for t in T_edges:
    plt.axvline(t, linewidth=1)
for c in C_edges:
    plt.axhline(c, linewidth=1)

# plot region centers
cent = df[df["is_center"] == 1]
plt.scatter(cent["Temperature (T)"], cent["Cao"], s=80, marker="*", label="region centers")

# plot failed points (if file exists)
# try:
#     fail = pd.read_csv("failed_points.csv")
#     plt.scatter(fail["T"], fail["Cao"], s=60, marker="x", label="failed points")
# except FileNotFoundError:
#     pass

plt.xlabel("T (K)")
plt.ylabel("Cao (mol/L)")
plt.title("Data points + regions + centers")
plt.legend()
plt.tight_layout()
plt.savefig(f"Data generated", dpi=300, bbox_inches="tight")
plt.close()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load solved data + linearization params
df = pd.read_csv("data_smart.csv")
lin = pd.read_csv("lin_params.csv")
df = df.merge(lin, on="region_id", how="left")

# Compute nonlinear f_nl on solved points
T   = df["Temperature (T)"].to_numpy()
Cao = df["Cao"].to_numpy()
Ca  = df["Ca"].to_numpy()
Cb  = df["Cb"].to_numpy()

kf = Afo * np.exp(-Eaf/(R*T))
kr = Aro * np.exp(-Ear/(R*T))

f_nl = (Cao - Ca
        - kf*Ca*(Cb**2)*tau
        + kr*(Cao - Ca + Cbo - Cb + Cco)*tau)

# Compute linearized f_lin on the SAME solved points (region-based Taylor)
f_lin = (df["fss"].to_numpy()
         + df["aCa"].to_numpy()*(Ca - df["Cass"].to_numpy())
         + df["aCb"].to_numpy()*(Cb - df["Cbss"].to_numpy())
         + df["aT"].to_numpy() *(T  - df["Tss"].to_numpy())
         + df["aCao"].to_numpy()*(Cao - df["Caoss"].to_numpy()))

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

# IMPORTANT: store these in df so you can reuse later if you want
df["f_nl"] = f_nl
df["f_lin"] = f_lin

T = df["Temperature (T)"].to_numpy(dtype=np_dtype)
C = df["Cao"].to_numpy(dtype=np_dtype)

Z_nl  = np.abs(df["f_nl"].to_numpy(dtype=np_dtype))
Z_lin = np.abs(df["f_lin"].to_numpy(dtype=np_dtype))

tri = mtri.Triangulation(T, C)

def trisurf_plot(Z, title, zlabel):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(tri, Z, linewidth=0.2, antialiased=True)
    ax.set_xlabel("T (K)")
    ax.set_ylabel("Cao (mol/L)")
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f"{zlabel} residuals", dpi=300, bbox_inches="tight")
    plt.close()

# Same style as your screenshot (absolute value)
trisurf_plot(Z_nl,  "Surface: |f_nl(T, Cao)|",  "|f_nl|")
trisurf_plot(Z_lin, "Surface: |f_lin(T, Cao)|", "|f_lin|")

# Optional (often nicer): log-scale surfaces
EPS = 1e-18
trisurf_plot(np.log10(Z_nl + EPS),  "Surface: log10(|f_nl(T, Cao)|)",  "log10(|f_nl|)")
trisurf_plot(np.log10(Z_lin + EPS), "Surface: log10(|f_lin(T, Cao)|)", "log10(|f_lin|)")


import numpy as np
import matplotlib.pyplot as plt

T = df["Temperature (T)"].to_numpy(dtype=np_dtype)
C = df["Cao"].to_numpy(dtype=np_dtype)
Z = np.abs(df["f_lin"].to_numpy(dtype=np_dtype))

plt.figure(figsize=(9,5))
hb = plt.hexbin(T, C, C=Z, gridsize=40, reduce_C_function=np.mean)
plt.colorbar(hb, label="mean |f_lin| per bin")
plt.xlabel("T (K)")
plt.ylabel("Cao (mol/L)")
plt.title("Hexbin heatmap")
plt.tight_layout()
plt.savefig(f"Hexbin heatmap", dpi=300, bbox_inches="tight")
plt.close()


plt.figure(figsize=(9,5))
hb = plt.hexbin(T, C, C=np.log10(Z + 1e-18), gridsize=40, reduce_C_function=np.mean)
plt.colorbar(hb, label="mean |f_lin| per bin")
plt.xlabel("T (K)")
plt.ylabel("Cao (mol/L)")
plt.title("Hexbin heatmap")
plt.tight_layout()
plt.savefig(f"Hexbin logarithmic heatmap", dpi=300, bbox_inches="tight")
plt.close()


df["abs_err"] = np.abs(df["f_lin"] - df["f_nl"])

Zerr = (df.groupby(["iC","iT"])["abs_err"]
          .max()
          .unstack("iT"))

# plt.figure(figsize=(8,5))
# plt.pcolormesh(T_edges, C_edges, Zerr.to_numpy(), shading="flat")
# plt.colorbar(label="max |f_lin - f_nl| per region")
# plt.xlabel("T (K)")
# plt.ylabel("Cao (mol/L)")
# plt.title("Region heatmap: max |f_lin - f_nl|")
# plt.tight_layout()
# plt.savefig(f"{zlabel} heatmap", dpi=300, bbox_inches="tight")
# plt.close()