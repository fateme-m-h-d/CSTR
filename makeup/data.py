#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sym
from sympy import symbols, Eq, solve
from scipy.optimize import fsolve

Cao = 1 #mol/L
Cbo = 2 #mol/L
V = 10 #L
Q = 1 #L/s
tau = V/Q #s

#Parameters to tuning to obtain "aggressive" non-linearity
Afo = 10e12
Eaf = 100000 #J/mol
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
T_values = np.linspace(280, 800, 1000)  # Adjust the range and number of points as needed


# Initial guess for fsolve
initial_guess = [0, 2, 1]

# Lists to store results
Ca_values = []
Cb_values = []
Cc_values = []

# Loop over each value of T and solve for Ca and Cb
for T in T_values:
    solution, infodict, ier, mesg = fsolve(equations, initial_guess, args=(T,), full_output=True)
    #solution, mesg = fsolve(equations, initial_guess, args=(T,))
    if ier == 1:  # ier == 1 indicates successful convergence
       Cc, Cb, Ca = solution
       Ca_values.append(Ca)
       Cb_values.append(Cb)
       Cc_values.append(Cc)
    else:
        print(f"Solver did not converge for T = {T}. Message: {mesg}")

kf = Afo * np.exp(-Eaf/(R*T_values)) #Arrhenius eqn for forward reaction
kr = Aro * np.exp(-Ear/(R*T_values)) #arrhenius eqn for reverse reaction
ra=-kf*Ca*(Cb**2)*tau + kr*(Cao-Ca+Cbo-Cb)

# Print results or process them further
for T, Ca, Cb, Cc in zip(T_values, Ca_values, Cb_values, Cc_values):
    print(f"T: {T:.2f}, Ca: {Ca:.4f}, Cb: {Cb:.4f}, Cc: {Cc:.4f}")


# Create a DataFrame
data = pd.DataFrame({
    'Temperature (T)': T_values,
    'Ca': Ca_values,
    'Cb': Cb_values,
    'Cc': Cc_values
})

# Save to Excel
data.to_csv("C:/Users/Fateme/Desktop/Research/CSTR/makeup/data.csv", index=False)
data.to_excel("C:/Users/Fateme/Desktop/Research/CSTR/makeup/data.xlsx", index=False)
print("Data saved to 'data.csv'")

from matplotlib import pyplot as plt

plt.figure()
plt.scatter(T_values, Ca_values)
plt.scatter(T_values, Cb_values)
plt.scatter(T_values, Cc_values)
plt.show()


plt.figure()
plt.plot(T_values, ra)
plt.show()



#Steady-state values
# Cass = 0.7 #mol/L
# Cbss = 0.9 #mol/L
# Tss = 300 #K
Cass = 0.602496402376611 #mol/L
Cbss = 1.20499280475946 #mol/L
Tss = 560.04004004004 #K
Ca, Cb, T = sym.symbols('Ca Cb T')
kf = Afo * sym.exp(-Eaf/(R*T)) #Arrhenius eqn for forward reaction
kr = Aro * sym.exp(-Ear/(R*T)) #arrhenius eqn for reverse reaction

#Linearize MB on A
f = Cao - Ca + -kf*Ca*(Cb**2)*tau + kr*(Cao-Ca+Cbo-Cb)*tau
df_Ca = f.diff(Ca)
df_Cb = f.diff(Cb)
df_T = f.diff(T)
fss = f.subs([(Ca,Cass),(Cb,Cbss),(T,Tss)])
df_Cass=df_Ca.subs([(Ca,Cass),(Cb,Cbss),(T,Tss)])
df_Cbss=df_Cb.subs([(Ca,Cass),(Cb,Cbss),(T,Tss)])
df_Tss=df_T.subs([(Ca,Cass),(Cb,Cbss),(T,Tss)])
f_linearized = fss + df_Cass*(Ca-Cass) + df_Cbss*(Cb-Cbss) + df_Tss*(T-Tss) #linearized equation
print(f_linearized)

#Linearize MB on B
g = Cbo - Cb + -2*kf*Ca*(Cb**2)*tau + 2*kr*(Cao-Ca+Cbo-Cb)*tau
dg_Ca = g.diff(Ca)
dg_Cb = g.diff(Cb)
dg_T = g.diff(T)
gss = g.subs([(Ca,Cass),(Cb,Cbss),(T,Tss)])
dg_Cass=dg_Ca.subs([(Ca,Cass),(Cb,Cbss),(T,Tss)])
dg_Cbss=dg_Cb.subs([(Ca,Cass),(Cb,Cbss),(T,Tss)])
dg_Tss=dg_T.subs([(Ca,Cass),(Cb,Cbss),(T,Tss)])
g_linearized = gss + dg_Cass*(Ca-Cass) + dg_Cbss*(Cb-Cbss) + dg_Tss*(T-Tss) #linearized equation
print(g_linearized)




#Extract coefficients in the linearized equations
a = f_linearized.coeff(Ca)
b = f_linearized.coeff(Cb)
c = f_linearized.coeff(T)
d = f_linearized.subs({Ca: 0, Cb:0, T: 0})

e = g_linearized.coeff(Ca)
f = g_linearized.coeff(Cb)
g = g_linearized.coeff(T)
h = g_linearized.subs({Ca: 0, Cb:0, T: 0})

# Create a DataFrame
data = pd.DataFrame({
    'a': [a],
    'b': [b],
    'c': [c],
    'd': [d],
    'e': [e],
    'f': [f],
    'g': [g],
    'h': [h]
})

# Save to Excel
data.to_excel("C:/Users/Fateme/Desktop/Research/CSTR/makeup/coefficients.xlsx", index=False)
print("Data saved to 'co.csv'")

n=1000
Ca_values_linear = np.zeros(n)
Cb_values_linear = np.zeros(n)
Cc_values_linear= np.zeros(n)

def linear_balance(x, T):
    Ca, Cb = x
    
    #Define the equations
    eqn1 = a*Ca + b*Cb + c*T + d
    eqn2 = e*Ca + f*Cb + g*T + h
    
    return ([eqn1, eqn2])

xguess = [Cao, Cbo]

print("Linear result")
for i in range(n):
    sol = fsolve(linear_balance, xguess, args=(T_values[i],))
    Ca_values_linear[i], Cb_values_linear[i] = sol[0], sol[1]
    Cc_values_linear[i]=Cao-Ca_values_linear[i]+Cbo-Cb_values_linear[i]
    print( "T:",T_values[i], "Ca:", Ca_values_linear[i], "Cb:", Cb_values_linear[i], "Cc:", Cc_values_linear[i])
    

want_plot = True

if want_plot:
    plt.figure()
    plt.scatter(T_values, Ca_values, label="Ca")
    plt.scatter(T_values, Cb_values, label="Cb")
    plt.scatter(T_values, Cc_values, label="Cc")
    plt.plot(T_values, Ca_values_linear, "k--")
    plt.plot(T_values, Cb_values_linear, "k--")
    plt.plot(T_values, Cc_values_linear, "k--")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Concentration (mol/L)")
    plt.title("Linear plot")
    plt.legend()
    plt.show()
    
separate_linear_plot = False
    
if separate_linear_plot:
    
    # Create a figure and a 1x2 grid of subplots (2 subplots in 1 row)
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    # Plot data on the first subplot
    axs[0].plot(T_values, Ca_values_linear, 'b-', label='Ca')
    axs[0].set_xlabel('T')
    axs[0].set_ylabel('Ca')
    axs[0].legend()

    # Plot data on the second subplot
    axs[1].plot(T_values, Cb_values_linear, 'r-', label='Cb')
    axs[1].set_xlabel('T')
    axs[1].set_ylabel('Cb')
    axs[1].legend()

    # Plot data on the third subplot
    axs[2].plot(T_values, Cc_values_linear, 'g-', label='Cc')
    axs[2].set_xlabel('T')
    axs[2].set_ylabel('Cc')
    axs[2].legend()

    #   Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()
    
    
#Import the data into a csv file    
"""
import pandas as pd
nonlinear_data = {'Temperature (K)': T_values, 'CA (mol/L)': Ca_val, 'CC (mol/L)': Cc_val}
dnl = pd.DataFrame(nonlinear_data)
dnl.to_csv('nonlinear_data.csv', index=False)
linear_data = {'Temperature (K)': T_values, 'CA (mol/L)': Ca_values, 'CC (mol/L)': Cc_values}
dl = pd.DataFrame(linear_data)
dl.to_csv('linear_data.csv', index=False)
"""
    


# %%
