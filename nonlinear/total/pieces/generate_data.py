import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sym
from sympy import symbols, Eq, solve
from scipy.optimize import fsolve


Cao = 1 #mol/L
Cbo = 2 #mol/L
Cco = 0 #mol/L
V = 10 #L
Q = 1 #L/s
tau = V/Q #s

#Parameters to tuning to obtain "aggressive" non-linearity
Afo = 10e12
Eaf = 90000 #J/mol
Aro = 10e10
Ear = 80000 #J/mol
R = 8.314 #J/mol


def equations(variables, T):
    Cc, Cb, Ca = variables
    kf = Afo * np.exp(-Eaf/(R*T)) #Arrhenius eqn for forward reaction
    kr = Aro * np.exp(-Ear/(R*T)) #arrhenius eqn for reverse reaction

    eq1 = Cao - Ca + -kf*Ca*(Cb**2)*tau + kr*(Cao-Ca+Cbo-Cb)*tau
    eq2 = Cbo - Cb + -2*kf*Ca*(Cb**2)*tau + 2*kr*(Cao-Ca+Cbo-Cb)*tau
    eq3=Cc-Cao+Ca-Cbo+Cb
    return [eq1, eq2, eq3]

# Define the range of T values
n = 30 #number of points #I'm gonna change this one from 1000 points and decrease it to 600.
T_values = np.linspace(280, 600, n)  # Adjust the range and number of points as needed         #change this for first half of the nonlinear equation that we have.


# Initial guess for fsolve
initial_guess = [Cco, Cbo, Cao]

# Lists to store results
Ca_values = np.ones(n)*Cao
Cb_values = np.ones(n)*Cbo
Cc_values = np.ones(n)*Cco
i=0
# Loop over each value of T and solve for Ca and Cb
for T in T_values:
    solution, infodict, ier, mesg = fsolve(equations, initial_guess, args=(T,), full_output=True, xtol=1e-11)
    #solution, mesg = fsolve(equations, initial_guess, args=(T,))
    if ier == 1:  # ier == 1 indicates successful convergence
       Cc_values[i], Cb_values[i], Ca_values[i] = solution[0], solution[1], solution[2]
    else:
        print(f"Solver did not converge for T = {T}. Message: {mesg}")
    i+=1


#kf = Afo * np.exp(-Eaf/(R*T_values)) #Arrhenius eqn for forward reaction
#kr = Aro * np.exp(-Ear/(R*T_values)) #arrhenius eqn for reverse reaction
#ra=-kf*Ca*(Cb**2)*tau + kr*(Cao-Ca+Cbo-Cb)

# Print results or process them further
#for T, Ca, Cb, Cc in zip(T_values, Ca_values, Cb_values, Cc_values):
    #print(f"T: {T:.2f}, Ca: {Ca:.4f}, Cb: {Cb:.4f}, Cc: {Cc:.4f}")


# Create a DataFrame
data = pd.DataFrame({
    'Temperature (T)': T_values,
    'Ca': Ca_values,
    'Cb': Cb_values,
    'Cc': Cc_values
})

# Save to Excel
data.to_csv("./data.csv", index=False)
#data.to_excel("./data.xlsx", index=False)
print("Data saved to 'data.csv'")

kf = Afo * np.exp(-Eaf/(R*T_values)) #Arrhenius eqn for forward reaction
kr = Aro * np.exp(-Ear/(R*T_values)) #arrhenius eqn for reverse reaction
f2 = (Cao - Ca_values
    - kf * Ca_values * (Cb_values**2) * tau
    + kr * (Cao - Ca_values + Cbo - Cb_values + Cco) * tau)

print("\n=== Nonlinear Constraint Values ===")
for i, T0 in enumerate(T_values):
    # print(f"T = {T0:.1f} K")
    # print(f"  f(T,Ca,Cb) = {f1[i]:.6e}")
    # print("-"*55)
    plt.scatter(T0, f2[i], color='blue', label='f(T,Ca,Cb)' if i == 0 else "")
plt.title("Nonlinear Constraint Values vs Temperature - prediction")
plt.xlabel("Temperature (K)")
plt.ylabel("nonlinear constraint")
plt.legend()
plt.show()

f3 = kf * Cao * (Cbo**2) * tau
plt.plot(T_values, f3, label='kf * Cao * (Cbo**2) * tau', color='orange')
plt.show()
# #added_part to generate data for check how effective can it extrapolate!
# T_limited = np.linspace(280, 600, 1000)
# data_limited = pd.DataFrame({
#     'Temperature (T)': T_limited,
#     'Ca': Ca_values,
#     'Cb': Cb_values,
#     'Cc': Cc_values
# })
# data_limited.to_csv("./data_limited.csv", index=False)
# plt.figure()
# plt.scatter(T_values, Ca_values)
# plt.scatter(T_values, Cb_values)
# plt.scatter(T_values, Cc_values)
# plt.show()


plt.plot(T_values, Ca_values,'b--',label='Ca')
plt.plot(T_values, Cb_values,'r--',label='Cb')
plt.plot(T_values, Cc_values,'g--',label='Cc')
    
plt.xlabel('Temperature (K)')
plt.ylabel('Concentration (mol/L)')
plt.title('Original Data')
plt.legend()
plt.grid()
plt.show()

# ───────────────────────── FUNCTIONS ─────────────────────────
def calc_Cb(T, Ca):
    """Solve for Cb such that eq1 = 0."""
    kf = Afo * np.exp(-Eaf / (R * T))
    kr = Aro * np.exp(-Ear / (R * T))

    def eq(Cb):
        return Cao - Ca + -kf * Ca * (Cb**2) * tau + kr * (Cao - Ca + Cbo - Cb + Cco) * tau

    Cb_guess = 1.0
    Cb_solution, = fsolve(eq, Cb_guess, xtol=1e-11)
    return Cb_solution

# ───────────────────────── MAIN ─────────────────────────
def main():
    # Load WITHOUT mutating original data
    df = pd.read_csv("data.csv")
    T = df["Temperature (T)"].to_numpy(dtype=np.float64)
    Ca = df["Ca"].to_numpy(dtype=np.float64)
    Cb_dataset = df["Cb"].to_numpy(dtype=np.float64)

    # Compute Cb_calc separately (no in-place edits)
    Cb_calc = np.array([calc_Cb(t, ca) for t, ca in zip(T, Ca)], dtype=np.float64)
    Cb_diff = Cb_dataset - Cb_calc

    # Build separate results table (still not modifying df)
    results = pd.DataFrame({
        "T": T,
        "Ca": Ca,
        "Cb_dataset": Cb_dataset,
        "Cb_calc": Cb_calc,
        "Cb_diff": Cb_diff
    })

    # Print all rows
    with pd.option_context("display.float_format", "{:,.12g}".format,
                           "display.max_rows", None,
                           "display.max_columns", None,
                           "display.width", 160):
        print(results)

    # Save results (optional)
    results.to_csv("cb_comparison.csv", index=False)

    # Plot 1: Cb vs T
    plt.figure(figsize=(9,5))
    plt.plot(T, Cb_dataset, linewidth=2, label="Dataset Cb")
    plt.plot(T, Cb_calc, "--", linewidth=2, label="Calculated Cb")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Cb (mol/L)")
    plt.title("Dataset Cb vs Calculated Cb")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cb_vs_t.png", dpi=150)
    plt.show()

    # Plot 2: Difference vs T
    plt.figure(figsize=(9,5))
    plt.plot(T, Cb_diff, linewidth=2, label="Cb_diff = Cb_dataset - Cb_calc")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Difference (mol/L)")
    plt.title("Difference between Dataset and Calculated Cb")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cb_diff_vs_t.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()

