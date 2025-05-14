import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sym


Cao = 1 #mol/L
Cbo = 2 #mol/L
V = 10 #L
Q = 1 #L/s
tau = V/Q #s

#Parameters to tuning to obtain "aggressive" non-linearity
Afo = 10e12
Eaf = 30000 #J/mol
Aro = 10e10
Ear = 15000 #J/mol
R = 8.314 #J/mol


#Steady-state values
Cass = 0.7 #mol/L
Cbss = 0.9 #mol/L
Tss = 300 #K
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


