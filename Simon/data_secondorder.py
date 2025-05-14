#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 21:21:27 2024

@author: simonnguyen
"""

#Reaction: A + 2B <-> C
#rA = 2*rB = -kf*Ca*Cb**2 + kr*Cc
#MB on A: Cao - Ca + rA*tau = 0
#MB on B: Cbo - Cb + 2*rA*tau = 0
#Overall balabce: Cc = Cao + Cbo + Cco - Ca - Cb

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
import sympy as sym
import pandas as pd

#Plot the Arrhenius equation
#Parameters of tunning to obtain "aggressive" non-linearity
Afo = 10e12
Eaf = 100000 #J/mol
Aro = 10e10
Ear = 80000 #J/mol
R = 8.314 #J/mol

V = 10 #L
Q = 1 #L/s
tau = V/Q #s

n = 1000
T_values = np.linspace(300,800,n)
#print(T_values)
Cao = 1 #mol/L
Cbo = 2 #mol/L
Cco = 0 #mol/L

kf_values = Afo * np.exp(-Eaf/R/T_values)
kr_values = Aro * np.exp(-Ear/R/T_values)

arrhenius_plot = False
if arrhenius_plot:
    plt.figure()
    plt.plot(T_values, -kf_values + kr_values)
    plt.xlabel("Temperature (K)")
    plt.ylabel("-kf + kr")
    plt.title("Arrhenius plot of nonlinearity check")
    plt.show()

#Find the solution of non-linear results
Ca_nonlinear = np.ones(n)*Cao
Cb_nonlinear = np.ones(n)*Cbo

initial_guess = [Cao, Cbo]

def balance(x, T):
    Ca, Cb = x
    Cc = Cao + Cbo + Cco - Ca - Cb
    kf = Afo * np.exp(-Eaf/R/T)
    kr = Aro * np.exp(-Ear/R/T)
    rA = -kf*Ca*Cb**2 + kr*Cc
    eqn1 = Cao - Ca + rA*tau
    eqn2 = Cbo - Cb + 2*rA*tau
    return ([eqn1, eqn2])

#solve for Ca, Cb, Cc
for T in range(n):
    sol, infodict, ier, mesg = fsolve(balance, initial_guess, args=(T_values[T],), full_output=True)
    if ier == 1:  # ier == 1 indicates successful convergence
       Ca_nonlinear[T], Cb_nonlinear[T] = sol[0], sol[1]
    else:
        print(f"Solver did not converge for T = {T}. Message: {mesg}")

Cc_nonlinear = (Cao + Cbo + Cco)*np.ones(n) - Ca_nonlinear - Cb_nonlinear


#for T, Ca, Cb, Cc in zip(T_values, Ca_val, Cb_val, Cc_val):
    #print(f"T: {T:.2f}, Ca: {Ca:.4f}, Cb: {Cb:.4f}, Cc: {Cc:.4f}")
    
#Create the data frame and convert to csv file
nonlinear_data = pd.DataFrame({'Temperature (K)': T_values, 'Ca (mol/L)': Ca_nonlinear, 'Cb (mol/L)': Cb_nonlinear})
nonlinear_data.to_csv("nonlinearized_data.csv", index=False)
    
nonlinear_only = False
if nonlinear_only:
    plt.figure()
    plt.plot(T_values, Ca_nonlinear,'b', label="Ca")
    plt.plot(T_values, Cb_nonlinear, 'r', label="Cb")
    plt.plot(T_values, Cc_nonlinear, "g", label="Cc")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Concentration (mol/L)")
    plt.legend()
    plt.title("Concentration versus Temperature")
    plt.show()

#Linearization of the Material Balance

#Steady-state values
print(T_values)
index = np.where((T_values < 500.5) & (T_values > 500))
Tss = float(T_values[index]) #K
Cass = float(Ca_nonlinear[index]) #mol/L
Cbss = float(Cb_nonlinear[index]) #mol/L
Ccss = float(Cc_nonlinear[index]) #mol/L

print("Tss:", Tss, "Cass:", Cass, "Cbss", Cbss, "Ccss", Ccss)

Ca, Cb, T = sym.symbols('Ca Cb T')
kf = Afo * sym.exp(-Eaf/(R*T))
kr = Aro * sym.exp(-Ear/(R*T))

#Linearize MB on A
rA = -kf*Ca*Cb**2 + kr*(Cao + Cbo + Cco - Ca - Cb)
f = Cao - Ca + rA*tau
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
rB = 2*rA
g = Cao - Ca + rB*tau
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
a = f_linearized.coeff(Ca)
b = f_linearized.coeff(Cb)
c = f_linearized.coeff(T)
d = f_linearized.subs({Ca:0, Cb:0, T:0})

e = g_linearized.coeff(Ca)
f = g_linearized.coeff(Cb)
g = g_linearized.coeff(T)
h = g_linearized.subs({Ca:0, Cb:0, T:0})

coefficients = pd.DataFrame({'a':[a], 'b':[b], 'c':[c], 'd':[d], 'e':[e], 'f':[f], 'g':[g], 'h':[h]})
coefficients.to_csv("coefficients.csv",index=False)
def linear_balance(x,T):
    Ca, Cb = x
    eqn1 = a*Ca + b*Cb + c*T + d
    eqn2 = e*Ca + f*Cb + g*T + h
    return ([eqn1, eqn2])

Ca_linear = np.zeros(n)
Cb_linear = np.zeros(n)

for i in range(n):
    sol = fsolve(linear_balance, initial_guess, args=(T_values[i],))
    Ca_linear[i], Cb_linear[i] = sol[0], sol[1]

Cc_linear = (Cao + Cbo + Cco)*np.ones(n) - Ca_linear - Cb_linear
combine_plot = True
if combine_plot:
    plt.figure()
    plt.plot(T_values, Ca_nonlinear,'b', label="Ca")
    plt.plot(T_values, Cb_nonlinear, 'r', label="Cb")
    plt.plot(T_values, Cc_nonlinear, "g", label="Cc")
    plt.plot(T_values, Ca_linear, "b--", label='Linearized Ca')
    plt.plot(T_values, Cb_linear, 'r--', label='Linearized Cb')
    plt.plot(T_values, Cc_linear, 'g--', label='Linearized Cc')
    plt.xlabel("Temperature (K)")
    plt.ylabel("Concentration (mol/L)")
    plt.legend()
    plt.title("Concentration versus Temperature")
    plt.show()



    