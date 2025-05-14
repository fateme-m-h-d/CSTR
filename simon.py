#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:17:06 2024

@author: simonnguyen
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
import sympy as sym

#Plot the Arrhenius equation
#Parameters to tuning to obtain "aggressive" non-linearity
Afo = 10e15
Eaf = 160000 #J/mol
Aro = 10e13
Ear = 130000 #J/mol
R = 8.314 #J/mol

V = 10 #L
Q = 1 #L/s
tau = V/Q #s

n = 100
T_values = np.linspace(300,600,n)
Cao = 2 #mol/L
Cco = 0 #mol/L
Cbo = 2 #mol/L

kf_values = np.zeros(n)
kr_values = np.zeros(n)

for i in range(n):
    kf = Afo * np.exp(-Eaf/R/T_values[i])
    kr = Aro * np.exp(-Ear/R/T_values[i])
    kf_values[i], kr_values[i] = kf, kr

#Arrhenius parameter tuning test
arrhenius_plot = True
if arrhenius_plot:
    plt.figure()
    plt.plot(T_values, -kf_values + kr_values)
    plt.title("Arrhenius plot of nonlinearity check")
    plt.show()
    
Ca_val = np.ones(n)*Cao
Cc_val = np.ones(n)*Cco
def balance(x,T):
    Ca, Cc = x
    kf = Afo * np.exp(-Eaf/R/T)
    kr = Aro * np.exp(-Ear/R/T)
    rA = -kf*Ca*(Cao-Ca+Cbo-Cc) + kr*Cc
    eqn1 = Cao - Ca + rA*tau
    eqn2 = Cco - Cc - rA*tau
    return ([eqn1, eqn2])

print("Non-linear result")
for j in range(n):
    sln = fsolve(balance, [Cao, Cco], args=(T_values[j],))
    Ca_val[j], Cc_val[j] = sln[0], sln[1]
    print(Ca_val[j], Cc_val[j])
print("----"*20)

nonlinear_balance_plot = True
if nonlinear_balance_plot:
    plt.figure()
    plt.plot(T_values, Ca_val, label="Ca")
    plt.plot(T_values, Cc_val, label='Cc')
    plt.xlabel("Temperature (K)")
    plt.ylabel("Concentration (mol/L)")
    plt.legend()
    plt.title("Non-linear plot")
    plt.show()

"""
Linearization of the Material balance
"""
"""
Reaction: A + B <-> C
MB on A: CAo - CA + rA*tau = 0 (Equation 1)
MB on B: CBo - CB + rB*tau = 0 (Equation 2)
Overall MB: CAo + CBo = CA + CB + CC (Equation 3)

Reaction rate:
rA = rB = -rC = -kf*CA*CB + kr*CC (Equation 4)

Rearrange Eqn (3) to isolate for CC, replace CC in Eqn (4) by the expression, and substitute into Equation (1):

CAo - CA + (-kf*CA*CB + kr*(CAo-CA+CBo-CB))*tau = 0 (Equation 5)
CBo - CB + (-kf*CA*CB + kr*(CAo-CA+CBo-CB))*tau = 0 (Equation 6)
Equation (5) and (6) are to be linearized with f(CA, CB, T) and g(CA, CB, T)

"""
#Steady-state values
Cass = 0.5 #mol/L
Ccss = 0 #mol/L
Tss = 400 #K
Ca, Cc, T = sym.symbols('Ca Cc T')
kf = Afo * sym.exp(-Eaf/(R*T)) #Arrhenius eqn for forward reaction
kr = Aro * sym.exp(-Ear/(R*T)) #arrhenius eqn for reverse reaction

#Linearize MB on A
rA = -kf*Ca*(Cao-Ca+Cbo-Cc) + kr*Cc
f = Cao - Ca + rA*tau
df_Ca = f.diff(Ca)
df_Cc = f.diff(Cc)
df_T = f.diff(T)
fss = f.subs([(Ca,Cass),(Cc,Ccss),(T,Tss)])
df_Cass = df_Ca.subs([(Ca,Cass),(Cc,Ccss),(T,Tss)])
df_Ccss = df_Cc.subs([(Ca,Cass),(Cc,Ccss),(T,Tss)])
df_Tss = df_T.subs([(Ca,Cass),(Cc,Ccss),(T,Tss)])
f_linearized = fss + df_Cass*(Ca-Cass) + df_Ccss*(Cc-Ccss) + df_Tss*(T-Tss) #linearized equation
print("Linearized MB on A is", f_linearized)

#Linearize MB on B
g = Cco - Cc - rA*tau
dg_Ca = g.diff(Ca)
dg_Cc = g.diff(Cc)
dg_T = g.diff(T)
gss = g.subs([(Ca,Cass),(Cc,Ccss),(T,Tss)])
dg_Cass = dg_Ca.subs([(Ca,Cass),(Cc,Ccss),(T,Tss)])
dg_Ccss = dg_Cc.subs([(Ca,Cass),(Cc,Ccss),(T,Tss)])
dg_Tss = dg_T.subs([(Ca,Cass),(Cc,Ccss),(T,Tss)])
g_linearized = gss + dg_Cass*(Ca-Cass) + dg_Ccss*(Cc-Ccss) + dg_Tss*(T-Tss) #linearized equation
print("Linearized MB on B is", g_linearized)

#Extract coefficients in the linearized equations
a = f_linearized.coeff(Ca)
b = f_linearized.coeff(Cc)
c = f_linearized.coeff(T)
d = f_linearized.subs({Ca: 0, Cc:0, T: 0})

e = g_linearized.coeff(Ca)
f = g_linearized.coeff(Cc)
g = g_linearized.coeff(T)
h = g_linearized.subs({Ca: 0, Cc:0, T: 0})

Ca_values = np.zeros(n)
Cc_values = np.zeros(n)

def linear_balance(x, T):
    Ca, Cc = x
    
    #Define the equations
    eqn1 = a*Ca + b*Cc + c*T + d
    eqn2 = e*Ca + f*Cc + g*T + h
    
    return ([eqn1, eqn2])

xguess = [Cao, Cbo]

print("Linear result")
for i in range(n):
    sol = fsolve(linear_balance, xguess, args=(T_values[i],))
    Ca_values[i], Cc_values[i] = sol[0], sol[1]
    print(Ca_values[i], Cc_values[i])
    

want_plot = True

if want_plot:
    plt.figure()
    plt.plot(T_values, Ca_values, "g--", label="Ca")
    plt.plot(T_values, Cc_values, "k--", label="Cc")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Concentration (mol/L)")
    plt.title("Linear plot")
    plt.legend()
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
    

