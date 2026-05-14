import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figs", exist_ok=True)

# -----------------------------
# Fixed constants
# -----------------------------
Cao = 1.0  # mol/L
Cbo = 2.0  # mol/L

V = 10.0   # L
Q = 1.0    # L/s
tau = V / Q

# Arrhenius parameters
Afo = 10e12
Eaf = 90000.0  # J/mol

Aro = 10e10
Ear = 80000.0  # J/mol

R = 8.314  # J/mol/K


# -----------------------------
# Artificial constraint equations
# -----------------------------
def equations(variables, T):
    Ca, Cb = variables

    kf = Afo * np.exp(-Eaf / (R * T))
    kr = Aro * np.exp(-Ear / (R * T))

    # Nonlinear artificial Arrhenius-based constraint
    # -kf * Cb^2 * tau + kr * Ca^2 * tau = 0
    eq1 = -kf * (Cb**2) * tau + kr * (Ca**2) * tau

    # Simple linear constraint
    # Cao + Ca - Cbo + Cb = 0
    # with Cao=1, Cbo=2 gives Ca + Cb = 1
    eq2 = Cao + Ca - Cbo + Cb

    return [eq1, eq2]


# -----------------------------
# Temperature range
# -----------------------------
n = 200
T_values = np.linspace(280, 600, n)

# -----------------------------
# Storage
# -----------------------------
Ca_values = np.zeros(n)
Cb_values = np.zeros(n)

eq1_residuals = np.zeros(n)
eq2_residuals = np.zeros(n)

# Initial guess
initial_guess = [0.5, 0.5]

# -----------------------------
# Solve for Ca and Cb at each T
# -----------------------------
for i, T in enumerate(T_values):

    solution, infodict, ier, mesg = fsolve(
        equations,
        initial_guess,
        args=(T,),
        full_output=True,
        xtol=1.0e-11
    )

    Ca, Cb = solution

    if ier == 1 and Ca > 0 and Cb > 0:
        Ca_values[i] = Ca
        Cb_values[i] = Cb

        eq1, eq2 = equations([Ca, Cb], T)
        eq1_residuals[i] = eq1
        eq2_residuals[i] = eq2

        # Use current solution as next initial guess
        # This helps fsolve converge smoothly along the temperature grid
        initial_guess = [Ca, Cb]

    else:
        print(f"Solver did not converge for T = {T:.2f} K")
        print(f"Message: {mesg}")

        Ca_values[i] = np.nan
        Cb_values[i] = np.nan
        eq1_residuals[i] = np.nan
        eq2_residuals[i] = np.nan


# -----------------------------
# Remove failed points if any
# -----------------------------
valid_mask = ~np.isnan(Ca_values)

T_values_valid = T_values[valid_mask]
Ca_values_valid = Ca_values[valid_mask]
Cb_values_valid = Cb_values[valid_mask]
eq1_residuals_valid = eq1_residuals[valid_mask]
eq2_residuals_valid = eq2_residuals[valid_mask]

# -----------------------------
# Lifted / nonlinear output columns
# -----------------------------
kf_arr = Afo * np.exp(-Eaf / (R * T_values_valid))
kr_arr = Aro * np.exp(-Ear / (R * T_values_valid))

kfCb2_values_valid = kf_arr * (Cb_values_valid**2)
krCa2_values_valid = kr_arr * (Ca_values_valid**2)


# -----------------------------
# Save data
# -----------------------------
data = pd.DataFrame({
    "Temperature (T)": T_values_valid,
    "Ca": Ca_values_valid,
    "Cb": Cb_values_valid,
    "kfCb2": kfCb2_values_valid,
    "krCa2": krCa2_values_valid
})

data.to_csv("data.csv", index=False)
print("Data saved to data.csv")


# -----------------------------
# Save constraint check
# -----------------------------
check_data = pd.DataFrame({
    "Temperature (T)": T_values_valid,
    "Ca": Ca_values_valid,
    "Cb": Cb_values_valid,
    "kfCb2": kfCb2_values_valid,
    "krCa2": krCa2_values_valid,
    "eq1_residual": eq1_residuals_valid,
    "eq2_residual": eq2_residuals_valid
})

check_data.to_csv("constraint_check.csv", index=False)
print("Constraint check saved to constraint_check.csv")

print("Maximum absolute eq1 residual:")
print(np.nanmax(np.abs(eq1_residuals_valid)))

print("Maximum absolute eq2 residual:")
print(np.nanmax(np.abs(eq2_residuals_valid)))


# -----------------------------
# Plot generated data: Ca and Cb
# -----------------------------
plt.figure()
plt.plot(T_values_valid, Ca_values_valid, "b--", label="Ca")
plt.plot(T_values_valid, Cb_values_valid, "r--", label="Cb")
plt.xlabel("Temperature (K)")
plt.ylabel("Concentration")
plt.title("Generated Data")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("figs/generated_data.png", dpi=200)
plt.close()


# -----------------------------
# Plot lifted outputs: kfCb2 and krCa2
# -----------------------------
plt.figure()
plt.plot(T_values_valid, kfCb2_values_valid, "b--", label="kfCb2")
plt.plot(T_values_valid, krCa2_values_valid, "r--", label="krCa2")
plt.xlabel("Temperature (K)")
plt.ylabel("Lifted output value")
plt.title("Lifted Output Data")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("figs/generated_lifted_data.png", dpi=200)
plt.close()


# -----------------------------
# Plot nonlinear constraint residual
# -----------------------------
plt.figure()
plt.plot(T_values_valid, eq1_residuals_valid, marker=".", linestyle="none", label="eq1 residual")
plt.xlabel("Temperature (K)")
plt.ylabel("Residual")
plt.title("Residual of Nonlinear Arrhenius-Based Constraint")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("figs/eq1_residual_vs_T.png", dpi=200)
plt.close()


# -----------------------------
# Plot lifted-output constraint residual
# -----------------------------
plt.figure()
plt.plot(
    T_values_valid,
    eq1_residuals_valid,
    marker=".",
    linestyle="none",
    label="lifted eq1 residual"
)
plt.xlabel("Temperature (K)")
plt.ylabel("Residual")
plt.title("Residual from Lifted Outputs: -tau*kfCb2 + tau*krCa2")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("figs/lifted_eq1_residual_vs_T.png", dpi=200)
plt.close()


# -----------------------------
# Plot linear constraint residual
# -----------------------------
plt.figure()
plt.plot(T_values_valid, eq2_residuals_valid, marker=".", linestyle="none", label="eq2 residual")
plt.xlabel("Temperature (K)")
plt.ylabel("Residual")
plt.title("Residual of Linear Constraint")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("figs/eq2_residual_vs_T.png", dpi=200)
plt.close()