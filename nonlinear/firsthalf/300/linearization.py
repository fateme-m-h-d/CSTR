import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sym
from sympy import symbols, Eq, solve
from scipy.optimize import fsolve
#Linearization of the Material Balance

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


#Steady-state values
#print(T_values)
#print(Ca_values)


# index = np.where((T_values < 500.5) & (T_values > 500))
# Tss = float(T_values[index]) #K
# Cass = float(Ca_values[index]) #mol/L
# Cbss = float(Cb_values[index]) #mol/L
# Ccss = float(Cc_values[index]) #mol/L

# print("Tss:", Tss, "Cass:", Cass, "Cbss", Cbss, "Ccss", Ccss)

# Cass = 0.602496402376611 #mol/L
# Cbss = 1.20499280475946 #mol/L
# Tss = 560.04004004004 #K

# Cass = 0.75624563472165 #mol/L
# Cbss = 1.5124912694433#mol/L
# Tss = 440.16016016016 #K

Cass = 0.737879797 #mol/L
Cbss = 1.475759594 #mol/L
Tss = 320.1337793 #K

Ca, Cb, T = sym.symbols('Ca Cb T')
kf = Afo * sym.exp(-Eaf/(R*T))
kr = Aro * sym.exp(-Ear/(R*T))

#Linearize MB on A
# rA = -kf*Ca*Cb**2 + kr*(Cao + Cbo + Cco - Ca - Cb)
# f = Cao - Ca + rA*tau
f = Cao - Ca + -kf*Ca*(Cb**2)*tau + kr*(Cao-Ca+Cbo-Cb)*tau
df_Ca = f.diff(Ca)
df_Cb = f.diff(Cb)
df_T = f.diff(T)
fss = f.subs([(Ca, Cass), (Cb, Cbss), (T, Tss)])
df_Cass = df_Ca.subs([(Ca, Cass), (Cb, Cbss), (T, Tss)])
df_Cbss = df_Cb.subs([(Ca, Cass), (Cb, Cbss), (T, Tss)])
df_Tss = df_T.subs([(Ca, Cass), (Cb, Cbss), (T, Tss)])
f_linearized = fss + df_Cass*(Ca-Cass) + df_Cbss*(Cb-Cbss) + df_Tss*(T-Tss)
print("Linearized MB on A is", f_linearized)

#Linearize MB on B
# rB = 2*rA
# g = Cao - Ca + rB*tau
g = Cbo - Cb + -2*kf*Ca*(Cb**2)*tau + 2*kr*(Cao-Ca+Cbo-Cb)*tau
dg_Ca = g.diff(Ca)
dg_Cb = g.diff(Cb)
dg_T = g.diff(T)
gss = g.subs([(Ca, Cass), (Cb, Cbss), (T, Tss)])
dg_Cass = dg_Ca.subs([(Ca, Cass), (Cb, Cbss), (T, Tss)])
dg_Cbss = dg_Cb.subs([(Ca, Cass), (Cb, Cbss), (T, Tss)])
dg_Tss = dg_T.subs([(Ca, Cass), (Cb, Cbss), (T, Tss)])
g_linearized = gss + dg_Cass*(Ca-Cass) + dg_Cbss*(Cb-Cbss) + dg_Tss*(T-Tss)
print("Linearized MB on A is", g_linearized)

#Extract the coefficients of the linearized equations
# a = f_linearized.coeff(Ca)
# c = f_linearized.coeff(Cb)
# d = f_linearized.coeff(T)
# e = f_linearized.subs({Ca:0, Cb:0, T:0})
# print('type',type(a))
a = float(f_linearized.coeff(Ca).evalf())
c = float(f_linearized.coeff(Cb).evalf())
d = float(f_linearized.coeff(T).evalf())
e = float(f_linearized.subs({Ca: 0, Cb: 0, T: 0}).evalf())
print(type(a), type(c), type(d), type(e))
print(a, c, d, e)
# f = g_linearized.coeff(Ca)
# g = g_linearized.coeff(Cb)
# h = g_linearized.coeff(T)
# i = g_linearized.subs({Ca:0, Cb:0, T:0})
n = float(g_linearized.coeff(Ca).evalf())
g = float(g_linearized.coeff(Cb).evalf())
h = float(g_linearized.coeff(T).evalf())
i = float(g_linearized.subs({Ca: 0, Cb: 0, T: 0}).evalf())
print(n, g, h, i)
print(type(n), type(g), type(h), type(i))

coefficients = pd.DataFrame({'a':[a], 'c':[c], 'd':[d], 'e':[e], 'f':[f], 'g':[g], 'h':[h], 'i':[i]})
coefficients.to_csv("./coefficients.csv",index=False)

#Estimate the linearized data and plot to see if the steady-state values are good
"""
def linear_balance(x,T):
    Ca, Cb = x
    eqn1 = a*Ca + b*Cb + c*T + d
    eqn2 = e*Ca + f*Cb + g*T + h
    return ([eqn1, eqn2])

Ca_linear = np.zeros(n)
Cb_linear = np.zeros(n)


i = 0
for T in T_values:
    solution, infodict, ier, mesg = fsolve(linear_balance, initial_guess, args=(T,), full_output=True)
    #solution, mesg = fsolve(equations, initial_guess, args=(T,))
    if ier == 1:  # ier == 1 indicates successful convergence
       Ca_values[i], Cb_values[i] = solution[0], solution[1]
    else:
        print(f"Solver did not converge for T = {T}. Message: {mesg}")
    i+=1
Cc_linear = (Cao + Cbo + Cco)*np.ones(n) - Ca_linear - Cb_linear
"""
