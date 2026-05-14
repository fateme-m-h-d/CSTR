# import numpy as np
# import pandas as pd
# from scipy.optimize import fsolve
# import os
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# os.makedirs("figs", exist_ok=True)

# # -----------------------------
# # Fixed constants
# # -----------------------------
# Cao = 1.0  # mol/L
# Cbo = 0.0  # mol/L

# V = 10.0   # L
# Q = 1.0    # L/s
# tau = V / Q

# # Arrhenius parameters
# Afo = 10e12
# Eaf = 90000.0  # J/mol

# Aro = 10e10
# Ear = 80000.0  # J/mol

# R = 8.314  # J/mol/K


# # -----------------------------
# # constraint equations
# # -----------------------------
# def equations(variables, T):
#     Ca, Cb = variables

#     kf = Afo * np.exp(-Eaf / (R * T))
#     kr = Aro * np.exp(-Ear / (R * T))

#     # Nonlinear Arrhenius-based constraint
#     # -kf * Cb^2 * tau + kr * Ca^2 * tau = 0
#     eq1 = -kf * Ca * tau + kr * (Cb**2) * tau + Cao - Ca

#     # Simple linear constraint
#     # Cao + Ca - Cbo + Cb = 0
#     # with Cao=1, Cbo=0 gives Ca + Cb = 1
#     eq2 = -Cao + Ca - Cbo + Cb

#     return [eq1, eq2]


# # -----------------------------
# # Temperature range
# # -----------------------------
# n = 200
# T_values = np.linspace(280, 600, n)

# # -----------------------------
# # Storage
# # -----------------------------
# Ca_values = np.zeros(n)
# Cb_values = np.zeros(n)

# eq1_residuals = np.zeros(n)
# eq2_residuals = np.zeros(n)

# # Initial guess
# initial_guess = [0.5, 0.5]

# # -----------------------------
# # Solve for Ca and Cb at each T
# # -----------------------------
# for i, T in enumerate(T_values):

#     solution, infodict, ier, mesg = fsolve(
#         equations,
#         initial_guess,
#         args=(T,),
#         full_output=True,
#         xtol=1.0e-11
#     )

#     Ca, Cb = solution

#     if ier == 1 and Ca > 0 and Cb > 0:
#         Ca_values[i] = Ca
#         Cb_values[i] = Cb

#         eq1, eq2 = equations([Ca, Cb], T)
#         eq1_residuals[i] = eq1
#         eq2_residuals[i] = eq2

#         # Use current solution as next initial guess
#         # This helps fsolve converge smoothly along the temperature grid
#         initial_guess = [Ca, Cb]

#     else:
#         print(f"Solver did not converge for T = {T:.2f} K")
#         print(f"Message: {mesg}")

#         Ca_values[i] = np.nan
#         Cb_values[i] = np.nan
#         eq1_residuals[i] = np.nan
#         eq2_residuals[i] = np.nan


# # -----------------------------
# # Remove failed points if any
# # -----------------------------
# valid_mask = ~np.isnan(Ca_values)

# T_values_valid = T_values[valid_mask]
# Ca_values_valid = Ca_values[valid_mask]
# Cb_values_valid = Cb_values[valid_mask]
# eq1_residuals_valid = eq1_residuals[valid_mask]
# eq2_residuals_valid = eq2_residuals[valid_mask]

# # -----------------------------
# # Lifted / nonlinear output columns
# # -----------------------------
# kf_arr = Afo * np.exp(-Eaf / (R * T_values_valid))
# kr_arr = Aro * np.exp(-Ear / (R * T_values_valid))

# kfCa_values_valid = (kf_arr * tau + 1) * (Ca_values_valid) 
# krCb2_values_valid = kr_arr * (Cb_values_valid**2)


# # -----------------------------
# # Save data
# # -----------------------------
# data = pd.DataFrame({
#     "Temperature (T)": T_values_valid,
#     "Ca": Ca_values_valid,
#     "Cb": Cb_values_valid,
#     "kfCa": kfCa_values_valid,
#     "krCb2": krCb2_values_valid
# })

# data.to_csv("data.csv", index=False)
# print("Data saved to data.csv")


# # -----------------------------
# # Save constraint check
# # -----------------------------
# check_data = pd.DataFrame({
#     "Temperature (T)": T_values_valid,
#     "Ca": Ca_values_valid,
#     "Cb": Cb_values_valid,
#     "kfCa": kfCa_values_valid,
#     "krCb2": krCb2_values_valid,
#     "eq1_residual": eq1_residuals_valid,
#     "eq2_residual": eq2_residuals_valid
# })

# check_data.to_csv("constraint_check.csv", index=False)
# print("Constraint check saved to constraint_check.csv")

# print("Maximum absolute eq1 residual:")
# print(np.nanmax(np.abs(eq1_residuals_valid)))

# print("Maximum absolute eq2 residual:")
# print(np.nanmax(np.abs(eq2_residuals_valid)))


# # -----------------------------
# # Plot generated data: Ca and Cb
# # -----------------------------
# plt.figure()
# plt.plot(T_values_valid, Ca_values_valid, "b--", label="Ca")
# plt.plot(T_values_valid, Cb_values_valid, "r--", label="Cb")
# plt.xlabel("Temperature (K)")
# plt.ylabel("Concentration")
# plt.title("Generated Data")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("figs/generated_data.png", dpi=200)
# plt.close()


# # -----------------------------
# # Plot lifted outputs: kfCb2 and krCa2
# # -----------------------------
# plt.figure()
# plt.plot(T_values_valid, kfCa_values_valid, "b--", label="kfCa")
# plt.plot(T_values_valid, krCb2_values_valid, "r--", label="krCb2")
# plt.xlabel("Temperature (K)")
# plt.ylabel("Lifted output value")
# plt.title("Lifted Output Data")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("figs/generated_lifted_data.png", dpi=200)
# plt.close()


# # -----------------------------
# # Plot nonlinear constraint residual
# # -----------------------------
# plt.figure()
# plt.plot(T_values_valid, eq1_residuals_valid, marker=".", linestyle="none", label="eq1 residual")
# plt.xlabel("Temperature (K)")
# plt.ylabel("Residual")
# plt.title("Residual of Nonlinear Arrhenius-Based Constraint")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("figs/eq1_residual_vs_T.png", dpi=200)
# plt.close()


# # -----------------------------
# # Plot lifted-output constraint residual
# # -----------------------------
# plt.figure()
# plt.plot(
#     T_values_valid,
#     eq1_residuals_valid,
#     marker=".",
#     linestyle="none",
#     label="lifted eq1 residual"
# )
# plt.xlabel("Temperature (K)")
# plt.ylabel("Residual")
# plt.title("Residual from Lifted Outputs: -tau*kfCb2 + tau*krCa2")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("figs/lifted_eq1_residual_vs_T.png", dpi=200)
# plt.close()


# # -----------------------------
# # Plot linear constraint residual
# # -----------------------------
# plt.figure()
# plt.plot(T_values_valid, eq2_residuals_valid, marker=".", linestyle="none", label="eq2 residual")
# plt.xlabel("Temperature (K)")
# plt.ylabel("Residual")
# plt.title("Residual of Linear Constraint")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("figs/eq2_residual_vs_T.png", dpi=200)
# plt.close()

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
Cbo = 0.0  # mol/L

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
# Constraint equations
# -----------------------------
def equations(variables, T):
    Ca, Cb = variables

    kf = Afo * np.exp(-Eaf / (R * T))
    kr = Aro * np.exp(-Ear / (R * T))

    # Nonlinear Arrhenius-based constraint
    # -(kf*tau + 1)*Ca + tau*kr*Cb^2 + Cao = 0
    eq1 = -kf * Ca * tau + kr * (Cb**2) * tau + Cao - Ca

    # Linear constraint
    # -Cao + Ca - Cbo + Cb = 0
    # with Cao=1, Cbo=0 gives Ca + Cb = 1
    eq2 = -Cao + Ca - Cbo + Cb

    return [eq1, eq2]


# -----------------------------
# Temperature range
# -----------------------------
n = 200
T_values = np.linspace(350, 600, n)

# -----------------------------
# Storage for positive/converged training data
# -----------------------------
Ca_values = np.zeros(n)
Cb_values = np.zeros(n)

eq1_residuals = np.zeros(n)
eq2_residuals = np.zeros(n)

# Storage for every fsolve output, including negative/failed ones
raw_rows = []

# Initial guess
initial_guess = [1, 2]


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

    # Always evaluate residuals for the returned fsolve solution
    eq1, eq2 = equations([Ca, Cb], T)

    # Always calculate kinetic terms and lifted variables
    kf = Afo * np.exp(-Eaf / (R * T))
    kr = Aro * np.exp(-Ear / (R * T))

    kfCa = (kf * tau + 1) * Ca
    krCb2 = kr * (Cb**2)

    # Save every fsolve result, even if negative or failed
    raw_rows.append({
        "Temperature (T)": T,
        "Ca_raw": Ca,
        "Cb_raw": Cb,
        "kf": kf,
        "kr": kr,
        "kfCa_raw": kfCa,
        "krCb2_raw": krCb2,
        "eq1_residual": eq1,
        "eq2_residual": eq2,
        "converged": ier == 1,
        "ier": ier,
        "positive_Ca": Ca > 0,
        "positive_Cb": Cb > 0,
        "both_positive": (Ca > 0) and (Cb > 0),
        "message": mesg
    })

    # Keep only converged positive points for the actual training dataset
    if ier == 1 and Ca > 0 and Cb > 0:
        Ca_values[i] = Ca
        Cb_values[i] = Cb
        eq1_residuals[i] = eq1
        eq2_residuals[i] = eq2

        # Use current physical solution as next initial guess
        initial_guess = [Ca, Cb]

    else:
        print(f"Non-positive or failed solution for T = {T:.2f} K")
        print(f"Ca = {Ca:.6e}, Cb = {Cb:.6e}, ier = {ier}")
        print(f"Message: {mesg}")

        Ca_values[i] = np.nan
        Cb_values[i] = np.nan
        eq1_residuals[i] = np.nan
        eq2_residuals[i] = np.nan


# -----------------------------
# Save all raw fsolve outputs
# -----------------------------
raw_data = pd.DataFrame(raw_rows)
raw_data.to_csv("raw_fsolve_solutions.csv", index=False)

print("Raw fsolve solutions saved to raw_fsolve_solutions.csv")
print("Number of total fsolve calls:")
print(len(raw_data))

print("Number of converged solutions:")
print(raw_data["converged"].sum())

print("Number of negative Ca values:")
print((raw_data["Ca_raw"] < 0).sum())

print("Number of negative Cb values:")
print((raw_data["Cb_raw"] < 0).sum())

print("Number of non-positive Ca or Cb values:")
print((~raw_data["both_positive"]).sum())


# -----------------------------
# Remove failed / non-positive points from training data
# -----------------------------
valid_mask = ~np.isnan(Ca_values)

T_values_valid = T_values[valid_mask]
Ca_values_valid = Ca_values[valid_mask]
Cb_values_valid = Cb_values[valid_mask]
eq1_residuals_valid = eq1_residuals[valid_mask]
eq2_residuals_valid = eq2_residuals[valid_mask]


# -----------------------------
# Lifted / nonlinear output columns for valid training data
# -----------------------------
kf_arr = Afo * np.exp(-Eaf / (R * T_values_valid))
kr_arr = Aro * np.exp(-Ear / (R * T_values_valid))

kfCa_values_valid = (kf_arr * tau + 1) * Ca_values_valid
krCb2_values_valid = kr_arr * (Cb_values_valid**2)


# -----------------------------
# Save positive/converged training data
# -----------------------------
data = pd.DataFrame({
    "Temperature (T)": T_values_valid,
    "Ca": Ca_values_valid,
    "Cb": Cb_values_valid,
    "kfCa": kfCa_values_valid,
    "krCb2": krCb2_values_valid
})

data.to_csv("data.csv", index=False)
print("Positive/converged training data saved to data.csv")
print("Number of rows saved to data.csv:")
print(len(data))


# -----------------------------
# Save constraint check for valid training data
# -----------------------------
check_data = pd.DataFrame({
    "Temperature (T)": T_values_valid,
    "Ca": Ca_values_valid,
    "Cb": Cb_values_valid,
    "kfCa": kfCa_values_valid,
    "krCb2": krCb2_values_valid,
    "eq1_residual": eq1_residuals_valid,
    "eq2_residual": eq2_residuals_valid
})

check_data.to_csv("constraint_check.csv", index=False)
print("Constraint check saved to constraint_check.csv")

if len(eq1_residuals_valid) > 0:
    print("Maximum absolute eq1 residual:")
    print(np.nanmax(np.abs(eq1_residuals_valid)))

    print("Maximum absolute eq2 residual:")
    print(np.nanmax(np.abs(eq2_residuals_valid)))
else:
    print("No valid positive/converged points were saved.")


# -----------------------------
# Plot generated data: Ca and Cb
# -----------------------------
if len(T_values_valid) > 0:
    plt.figure()
    plt.plot(T_values_valid, Ca_values_valid, "b--", label="Ca")
    plt.plot(T_values_valid, Cb_values_valid, "r--", label="Cb")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Concentration")
    plt.title("Generated Data: Positive/Converged Points")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("figs/generated_data.png", dpi=200)
    plt.close()


# -----------------------------
# Plot lifted outputs: kfCa and krCb2
# -----------------------------
if len(T_values_valid) > 0:
    plt.figure()
    plt.plot(T_values_valid, kfCa_values_valid, "b--", label="kfCa")
    plt.plot(T_values_valid, krCb2_values_valid, "r--", label="krCb2")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Lifted output value")
    plt.title("Lifted Output Data")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("figs/generated_lifted_data.png", dpi=200)
    plt.close()


# -----------------------------
# Plot nonlinear constraint residual for valid points
# -----------------------------
if len(T_values_valid) > 0:
    plt.figure()
    plt.plot(
        T_values_valid,
        eq1_residuals_valid,
        marker=".",
        linestyle="none",
        label="eq1 residual"
    )
    plt.xlabel("Temperature (K)")
    plt.ylabel("Residual")
    plt.title("Residual of Nonlinear Arrhenius-Based Constraint")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("figs/eq1_residual_vs_T.png", dpi=200)
    plt.close()


# -----------------------------
# Plot lifted-output constraint residual for valid points
# -----------------------------
if len(T_values_valid) > 0:
    lifted_residual_valid = -kfCa_values_valid + tau * krCb2_values_valid + Cao

    plt.figure()
    plt.plot(
        T_values_valid,
        lifted_residual_valid,
        marker=".",
        linestyle="none",
        label="lifted residual: -kfCa + tau*krCb2 + Cao"
    )
    plt.xlabel("Temperature (K)")
    plt.ylabel("Residual")
    plt.title("Residual from Lifted Outputs")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("figs/lifted_eq1_residual_vs_T.png", dpi=200)
    plt.close()


# -----------------------------
# Plot linear constraint residual for valid points
# -----------------------------
if len(T_values_valid) > 0:
    plt.figure()
    plt.plot(
        T_values_valid,
        eq2_residuals_valid,
        marker=".",
        linestyle="none",
        label="eq2 residual"
    )
    plt.xlabel("Temperature (K)")
    plt.ylabel("Residual")
    plt.title("Residual of Linear Constraint")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("figs/eq2_residual_vs_T.png", dpi=200)
    plt.close()


# -----------------------------
# Plot raw fsolve Ca/Cb, including negative values
# -----------------------------
plt.figure()
plt.plot(raw_data["Temperature (T)"], raw_data["Ca_raw"], "b.", label="Ca raw")
plt.plot(raw_data["Temperature (T)"], raw_data["Cb_raw"], "r.", label="Cb raw")
plt.axhline(0.0, linestyle="--", linewidth=1)
plt.xlabel("Temperature (K)")
plt.ylabel("Raw fsolve solution")
plt.title("Raw fsolve Ca and Cb Values")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("figs/raw_fsolve_Ca_Cb.png", dpi=200)
plt.close()


# -----------------------------
# Plot raw fsolve residuals
# -----------------------------
plt.figure()
plt.plot(
    raw_data["Temperature (T)"],
    raw_data["eq1_residual"],
    marker=".",
    linestyle="none",
    label="eq1 residual"
)
plt.plot(
    raw_data["Temperature (T)"],
    raw_data["eq2_residual"],
    marker=".",
    linestyle="none",
    label="eq2 residual"
)
plt.xlabel("Temperature (K)")
plt.ylabel("Residual")
plt.title("Raw fsolve Residuals")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("figs/raw_fsolve_residuals.png", dpi=200)
plt.close()