import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use('Agg')      # must come before importing pyplot
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# === Given constants ===
Cao = 1.0   # mol/L
Cbo = 2.0   # mol/L
Cco = 0.0   # mol/L
V   = 10.0  # L
Q   = 1.0   # L/s
tau = V / Q # s

Afo = 1e13      # forward pre‐exponential factor (L·mol⁻¹·s⁻¹)
Eaf = 90000.0   # forward activation energy (J/mol)
Aro = 1e11      # reverse pre‐exponential factor (L·mol⁻¹·s⁻¹)
Ear = 80000.0   # reverse activation energy (J/mol)
R   = 8.314     # gas constant (J/(mol·K))

# === Load your measured dataset ===
df = pd.read_csv('data.csv')  # make sure this file is in your working directory

# Extract columns (note exact column names in your CSV)
T_values  = df['Temperature (T)'].values  # in K
Ca_values = df['Ca'].values               # in mol/L
Cb_values = df['Cb'].values               # in mol/L

# === Compute rate constants for each data point ===
kf = Afo * np.exp(-Eaf / (R * T_values))
kr = Aro * np.exp(-Ear / (R * T_values))

# === Evaluate the constraint function f(T, Ca, Cb) ===
# f = Cao – Ca – kf*Ca*Cb^2*tau + kr*(Cao–Ca + Cbo–Cb + Cco)*tau
f = (
    Cao 
    - Ca_values 
    - kf * Ca_values * (Cb_values**2) * tau 
    + kr * (Cao - Ca_values + Cbo - Cb_values + Cco) * tau
)

# === Select points approximately satisfying the constraint ===
epsilon = 1e-2
mask    = np.abs(f) < epsilon
T_surf  = T_values[mask]
Ca_surf = Ca_values[mask]
Cb_surf = Cb_values[mask]

# === Plot the result ===
fig = plt.figure(figsize=(8,6))
ax  = fig.add_subplot(111, projection='3d')

# ax.scatter(T_surf, Ca_surf, Cb_surf, c='teal', s=10, alpha=0.7)
# ax.scatter(T_values, Ca_values, Cb_values, c='teal', s=10, alpha=0.7)
# ax.set_xlabel('Temperature (K)')
# ax.set_ylabel('Ca (mol/L)')
# ax.set_zlabel('Cb (mol/L)')
# ax.set_title('Constraint Surface from data.csv: f(T, Ca, Cb) ≈ 0')

# plt.tight_layout()
# plt.show()
# plt.savefig('constraint_surface.png', dpi=150, bbox_inches='tight')

sc = ax.scatter(
    T_values,
    Ca_values,
    Cb_values,
    c=f,           # color by f!
    cmap='coolwarm',
    s=10, alpha=0.8
)
plt.colorbar(sc, label='Constraint Residual f')
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Ca (mol/L)')
ax.set_zlabel('Cb (mol/L)')
ax.set_title('Constraint Surface from data.csv: f(T, Ca, Cb) ≈ 0')

plt.tight_layout()
plt.show()
# plt.savefig('constraint_surface.png', dpi=150, bbox_inches='tight')
