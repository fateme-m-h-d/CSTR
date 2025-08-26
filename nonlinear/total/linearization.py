# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import sympy as sym
# from sympy import symbols, Eq, solve
# from scipy.optimize import fsolve
# #Linearization of the Material Balance

# Cao = 1 #mol/L
# Cbo = 2 #mol/L
# Cco = 0 #mol/L
# V = 10 #L
# Q = 1 #L/s
# tau = V/Q #s

# #Parameters to tuning to obtain "aggressive" non-linearity
# Afo = 10e12
# Eaf = 90000 #J/mol
# Aro = 10e10
# Ear = 80000 #J/mol
# R = 8.314 #J/mol

# Cass=0.3510168217392415
# Cbss=0.7020336434749113
# Tss=550.6212424849699
# # Ca, Cb, T = sym.symbols('Ca Cb T')
# # kf = Afo * sym.exp(-Eaf/(R*T))
# # kr = Aro * sym.exp(-Ear/(R*T))
# Ca, Cb, u = sym.symbols('Ca Cb u')  # u = 1000/T
# T_from_u = 1000/u
# kf = Afo * sym.exp(-Eaf/(R*T_from_u))
# kr = Aro * sym.exp(-Ear/(R*T_from_u))
# uss = 1000/Tss
# #Linearize MB on A

# f = Cao - Ca + -kf*Ca*(Cb**2)*tau + kr*(Cao-Ca+Cbo-Cb+Cco)*tau
# df_Ca = f.diff(Ca)
# df_Cb = f.diff(Cb)
# df_T = f.diff(u)
# fss = f.subs([(Ca, Cass), (Cb, Cbss), (u, uss)])
# df_Cass = df_Ca.subs([(Ca, Cass), (Cb, Cbss), (u, uss)])
# df_Cbss = df_Cb.subs([(Ca, Cass), (Cb, Cbss), (u, uss)])
# df_Tss = df_T.subs([(Ca, Cass), (Cb, Cbss), (u, uss)])
# f_linearized = fss + df_Cass*(Ca-Cass) + df_Cbss*(Cb-Cbss) + df_Tss*(u-uss)
# print("Linearized MB on A is", f_linearized)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sym
from sympy import symbols

# === 1) Define constants and steady‐state for linearization ===
Cao, Cbo, Cco = 1.0, 2.0, 0.0   # mol/L
V, Q = 10.0, 1.0               # L, L/s
tau = V/Q                     # s

Afo, Eaf = 1e13, 90000.0      # forward Arrhenius
Aro, Ear = 1e11, 80000.0      # reverse Arrhenius
R = 8.314                     # J/(mol·K)

# (replace with your actual steady‐state point)

430.06012024048096,0.41589194923686623,0.8317838984736275,1.752324152289506
Cass, Cbss, Tss = 0.41589194923686623, 0.8317838984736275, 430.06012024048096

# 480.0801603206413,0.3839514142818392,0.767902828563119,1.8481457571550417
# Cass, Cbss, Tss = 0.3839514142818392, 0.767902828563119, 480.0801603206413

# 515.3507014028056,0.3660794237985876,0.7321588475969961,1.9017617286044162
# Cass, Cbss, Tss = 0.3660794237985876, 0.7321588475969961, 515.3507014028056

# 540.3607214428857,0.3551476226232751,0.7102952452401861,1.934557132136539
# Cass, Cbss, Tss = 0.3551476226232751, 0.7102952452401861, 540.3607214428857

# 557.6753507014027,0.34828559395504716,0.6965711879167573,1.9551432181281954
# Cass, Cbss, Tss = 0.34828559395504716, 0.6965711879167573, 557.6753507014027

# 571.7835671342685,0.34307151217593673,0.686143024351111,1.9707854634729522
# Cass, Cbss, Tss = 0.34307151217593673, 0.686143024351111, 571.7835671342685

# 584.6092184368738,0.3385990203254893,0.6771980406362342,1.9842029390382765
# Cass, Cbss, Tss = 0.3385990203254893, 0.6771980406362342, 584.6092184368738

# 595.5110220440881,0.3349827056400738,0.6699654112978638,1.9950518830620625
# Cass, Cbss, Tss = 0.3349827056400738, 0.6699654112978638, 595.5110220440881

# === 2) Build symbolic f and its linearization ===
Ca, Cb, T = symbols('Ca Cb T')
kf = Afo * sym.exp(-Eaf/(R*T))
kr = Aro * sym.exp(-Ear/(R*T))

f       = (Cao - Ca
           - kf*Ca*Cb**2 * tau
           + kr*(Cao - Ca + Cbo - Cb + Cco) * tau)

fss     = f.subs({Ca: Cass, Cb: Cbss, T: Tss})
df_Ca   = sym.diff(f, Ca).subs({Ca: Cass, Cb: Cbss, T: Tss})
df_Cb   = sym.diff(f, Cb).subs({Ca: Cass, Cb: Cbss, T: Tss})
df_T    = sym.diff(f, T).subs({Ca: Cass, Cb: Cbss, T: Tss})

f_lin   = fss \
          + df_Ca*(Ca - Cass) \
          + df_Cb*(Cb - Cbss) \
          + df_T *(T  - Tss)
          
print("Linearized MB on A is", f_lin)

# === 3) Lambdify for fast NumPy calls ===
f_nl_func  = sym.lambdify((Ca, Cb, T), f,     'numpy')
f_lin_func = sym.lambdify((Ca, Cb, T), f_lin, 'numpy')

# === 4) Load your original data ===
df = pd.read_csv('data.csv')
# make sure your CSV has columns labeled exactly 'T', 'Ca', 'Cb'
print("Data columns:", df.columns.tolist())

T_data  = df['Temperature (T)'].values
Ca_data = df['Ca'].values
Cb_data = df['Cb'].values

# === 5) Evaluate both residuals on your data ===
res_nl  = f_nl_func(Ca_data, Cb_data, T_data)
res_lin = f_lin_func(Ca_data, Cb_data, T_data)

df['residual_nonlinear']  = res_nl
df['residual_linearized'] = res_lin

# === 6) Plot comparison ===
plt.figure(figsize=(6,4))
plt.plot(T_data, res_nl,  'o', label='Nonlinear residual')
plt.plot(T_data, res_lin, 'x', label='Linearized residual')
plt.axhline(0, color='gray', lw=0.5)
plt.xlabel('Temperature T (K)')
plt.ylabel('Residual f(Ca, Cb, T)')
plt.title('Residuals at Original Data Points')
plt.legend()
plt.tight_layout()
plt.show()

# # # --- 1) Load & filter original data to 500 ≤ T ≤ 600 ---
# # df = pd.read_csv('data.csv')          # expects columns ['T','Ca','Cb',…]
# # mask = (df['Temperature (T)'] >= 500) & (df['Temperature (T)'] <= 600)
# # df_range = df.loc[mask, ['Temperature (T)','Ca','Cb']].copy()

# # # --- 2) Extract as NumPy arrays and evaluate both functions ---
# # T_vals  = df_range['Temperature (T)'].values
# # Ca_vals = df_range['Ca'].values
# # Cb_vals = df_range['Cb'].values

# # res_nl  = f_nl_func(Ca_vals, Cb_vals, T_vals)
# # res_lin = f_lin_func(Ca_vals, Cb_vals, T_vals)

# # # --- 3) Attach results to the DataFrame & print ---
# # df_range['residual_nonlinear']  = res_nl
# # df_range['residual_linearized'] = res_lin

# # # Print the inputs and both outputs
# # print(df_range.to_string(index=False))

# # # (Optional) save to CSV
# # df_range.to_csv('residual_comparison_500_600.csv', index=False)
# # print("\nWrote residual_comparison_500_600.csv")

# # # --- 4) Plot the two curves over T ∈ [500,600] ---
# # plt.figure(figsize=(6,4))
# # plt.plot(df_range['Temperature (T)'], df_range['residual_nonlinear'],
# #          'o', label='Nonlinear residual')
# # plt.plot(df_range['Temperature (T)'], df_range['residual_linearized'],
# #          'x', label='Linearized residual')
# # plt.axhline(0, color='gray', lw=0.5)
# # plt.xlabel('Temperature, T (K)')
# # plt.ylabel('Residual f(Ca, Cb, T)')
# # plt.title('Residuals on Original Data (500–600 K)')
# # plt.legend()
# # plt.tight_layout()
# # plt.show()

