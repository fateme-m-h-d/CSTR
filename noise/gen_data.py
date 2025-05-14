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
n = 1000  # number of points
n1 = n // 2  # number of points in the first range
n2 = n - n1  # number of points in the second range

# Generate T_values excluding the 400 to 500 range
T_values1 = np.linspace(280, 400, n1, endpoint=False)  # First range
T_values2 = np.linspace(500, 600, n2)  # Second range
T_values = np.concatenate((T_values1, T_values2))

# Initial guess for fsolve
initial_guess = [Cco, Cbo, Cao]

# Lists to store results
Ca_values = np.ones(n) * Cao
Cb_values = np.ones(n) * Cbo
Cc_values = np.ones(n) * Cco

# Loop over each value of T and solve for Ca and Cb
for i, T in enumerate(T_values):
    solution, infodict, ier, mesg = fsolve(equations, initial_guess, args=(T,), full_output=True)
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

#added_part to generate data for check how effective can it extrapolate!
T_limited = np.linspace(280, 600, 1000)
data_limited = pd.DataFrame({
    'Temperature (T)': T_limited,
    'Ca': Ca_values,
    'Cb': Cb_values,
    'Cc': Cc_values
})
data_limited.to_csv("./data_limited.csv", index=False)
# plt.figure()
# plt.scatter(T_values, Ca_values)
# plt.scatter(T_values, Cb_values)
# plt.scatter(T_values, Cc_values)
# plt.show()