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
n = 500 #number of points #I'm gonna change this one from 1000 points and decrease it to 600.
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
    solution, infodict, ier, mesg = fsolve(equations, initial_guess, args=(T,), full_output=True)
    #solution, mesg = fsolve(equations, initial_guess, args=(T,))
    if ier == 1:  # ier == 1 indicates successful convergence
       Cc_values[i], Cb_values[i], Ca_values[i] = solution[0], solution[1], solution[2]
    else:
        print(f"Solver did not converge for T = {T}. Message: {mesg}")
    i+=1


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

kf_arr = Afo * np.exp(-Eaf/(R*T_values))
kr_arr = Aro * np.exp(-Ear/(R*T_values))

#Linearize MB on A
# rA = -kf*Ca*Cb**2 + kr*(Cao + Cbo + Cco - Ca - Cb)
# f = Cao - Ca + rA*tau
f = kr_arr*(Cao-Ca_values+Cbo-Cb_values+Cco)
g = kf_arr*Ca_values*(Cb_values**2)
fg = f - g
# Create a DataFrame
data = pd.DataFrame({
    'Temperature (T)': T_values,
    'Ca': Ca_values,
    'Cb': Cb_values,
    'Cc': Cc_values,
    'fg': fg
})

# Save to Excel
data.to_csv("./data.csv", index=False)
#data.to_excel("./data.xlsx", index=False)
print("Data saved to 'data.csv'")

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