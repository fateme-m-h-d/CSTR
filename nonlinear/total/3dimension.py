import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection

# 1. Load your data
df = pd.read_csv('data.csv')  
# assume df has columns: 'Ca', 'Cb', 'Cc', 'T'

# 2. Constants
Afo = 10e12
Eaf = 90000    # J/mol
Aro = 10e10
Ear = 80000    # J/mol
R   = 8.314    # J/mol·K

Cao = 1.0      # mol/L
Cbo = 2.0      # mol/L
Cco = 0.0      # mol/L
V   = 10.0     # L
Q   = 1.0      # L/s
tau = V / Q    # s

# 3. Compute temperature‑dependent rate constants (vectorized)
T = df['Temperature (T)'].values
kf = Afo * np.exp(-Eaf / (R * T))
kr = Aro * np.exp(-Ear / (R * T))

# 4. Extract concentrations
Ca = df['Ca'].values
Cb = df['Cb'].values
# Cc = df['Cc'].values  # not used in the formula as written

# 5. Compute the constraint f
f = (
    Cao - Ca
    - kf * Ca * (Cb**2) * tau
    + kr * (Cao - Ca + Cbo - Cb + Cco) * tau
)

# 6. Make a 3D scatter plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Cb, T, f, c=f, cmap='viridis', marker='o')
ax.set_xlabel('C$_B$ (mol/L)')
ax.set_ylabel('T (K)')
ax.set_zlabel('f')
ax.set_title('Constraint surface: f vs C$_B$ and T')
plt.show()
