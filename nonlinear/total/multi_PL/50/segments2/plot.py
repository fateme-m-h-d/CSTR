import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

edges = np.load("region_edges.npz")
T_edges = edges["T_edges"]
C_edges = edges["C_edges"]

T_centers = 0.5 * (T_edges[:-1] + T_edges[1:])
C_centers = 0.5 * (C_edges[:-1] + C_edges[1:])

TT, CC = np.meshgrid(T_centers, C_centers)

plt.figure(figsize=(8,6))
plt.scatter(df["Temperature (T)"], df["Cao"], s=10, label="solved points")

for t in T_edges:
    plt.axvline(t, linewidth=1)

for c in C_edges:
    plt.axhline(c, linewidth=1)

plt.scatter(TT.ravel(), CC.ravel(), s=80, marker="*", label="region centers")

plt.xlabel("T (K)")
plt.ylabel("Cao (mol/L)")
plt.title("Data points + regions + centers")
plt.legend()
plt.tight_layout()
plt.savefig("generated_data_plot.png", dpi=300, bbox_inches="tight")
plt.show()