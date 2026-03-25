import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

SCENARIOS = [0, 1, 2, 5, 10, 15, 20, 25]

x_vals = []
nn_rmse_mean, nn_rmse_ci = [], []
kkt_rmse_mean, kkt_rmse_ci = [], []

nn_viol_mean, nn_viol_ci = [], []
kkt_viol_mean, kkt_viol_ci = [], []

nn_time_mean, nn_time_ci = [], []
kkt_time_mean, kkt_time_ci = [], []

for n in SCENARIOS:
    df = pd.read_csv(f"experiment_epoch_errors_ninner_{n}.csv")
    data_df = pd.read_csv(f"data_ninner_{n}.csv")   # actual number of sample points
    x_vals.append(len(data_df))

    # ----- RMSE -----
    nn = df["NN_Experiment_RMSE"].dropna().to_numpy()
    kkt = df["KKThPINN_Experiment_RMSE"].dropna().to_numpy()

    nn_ci = ttest_1samp(nn, popmean=0).confidence_interval(0.95)
    kkt_ci = ttest_1samp(kkt, popmean=0).confidence_interval(0.95)

    nn_rmse_mean.append(nn.mean())
    kkt_rmse_mean.append(kkt.mean())
    nn_rmse_ci.append((nn_ci.high - nn_ci.low) / 2)
    kkt_rmse_ci.append((kkt_ci.high - kkt_ci.low) / 2)

    # ----- Violation -----
    nn = df["NN_Experiment_VIOL"].dropna().to_numpy()
    kkt = df["KKThPINN_Experiment_VIOL"].dropna().to_numpy()

    nn_ci = ttest_1samp(nn, popmean=0).confidence_interval(0.95)
    kkt_ci = ttest_1samp(kkt, popmean=0).confidence_interval(0.95)

    nn_viol_mean.append(nn.mean())
    kkt_viol_mean.append(kkt.mean())
    nn_viol_ci.append((nn_ci.high - nn_ci.low) / 2)
    kkt_viol_ci.append((kkt_ci.high - kkt_ci.low) / 2)

    # ----- Inference time -----
    nn = df["NN_Experiment_Time_sec"].dropna().to_numpy()
    kkt = df["KKThPINN_Experiment_Time_sec"].dropna().to_numpy()

    nn_ci = ttest_1samp(nn, popmean=0).confidence_interval(0.95)
    kkt_ci = ttest_1samp(kkt, popmean=0).confidence_interval(0.95)

    nn_time_mean.append(nn.mean())
    kkt_time_mean.append(kkt.mean())
    nn_time_ci.append((nn_ci.high - nn_ci.low) / 2)
    kkt_time_ci.append((kkt_ci.high - kkt_ci.low) / 2)


# ===== extra case: n_inner_per_region=0 and include_edge_midpoints=False =====
df = pd.read_csv("experiment_epoch_errors.csv")
data_df = pd.read_csv("data.csv")
x_vals.append(len(data_df))

# RMSE
nn = df["NN_Experiment_RMSE"].dropna().to_numpy()
kkt = df["KKThPINN_Experiment_RMSE"].dropna().to_numpy()

nn_ci = ttest_1samp(nn, popmean=0).confidence_interval(0.95)
kkt_ci = ttest_1samp(kkt, popmean=0).confidence_interval(0.95)

nn_rmse_mean.append(nn.mean())
kkt_rmse_mean.append(kkt.mean())
nn_rmse_ci.append((nn_ci.high - nn_ci.low) / 2)
kkt_rmse_ci.append((kkt_ci.high - kkt_ci.low) / 2)

# Violation
nn = df["NN_Experiment_VIOL"].dropna().to_numpy()
kkt = df["KKThPINN_Experiment_VIOL"].dropna().to_numpy()

nn_ci = ttest_1samp(nn, popmean=0).confidence_interval(0.95)
kkt_ci = ttest_1samp(kkt, popmean=0).confidence_interval(0.95)

nn_viol_mean.append(nn.mean())
kkt_viol_mean.append(kkt.mean())
nn_viol_ci.append((nn_ci.high - nn_ci.low) / 2)
kkt_viol_ci.append((kkt_ci.high - kkt_ci.low) / 2)

# Inference time
nn = df["NN_Experiment_Time_sec"].dropna().to_numpy()
kkt = df["KKThPINN_Experiment_Time_sec"].dropna().to_numpy()

nn_ci = ttest_1samp(nn, popmean=0).confidence_interval(0.95)
kkt_ci = ttest_1samp(kkt, popmean=0).confidence_interval(0.95)

nn_time_mean.append(nn.mean())
kkt_time_mean.append(kkt.mean())
nn_time_ci.append((nn_ci.high - nn_ci.low) / 2)
kkt_time_ci.append((kkt_ci.high - kkt_ci.low) / 2)

# ===== sort by number of sample points =====
idx = np.argsort(x_vals)

x_vals = np.array(x_vals)[idx]

nn_rmse_mean = np.array(nn_rmse_mean)[idx]
nn_rmse_ci   = np.array(nn_rmse_ci)[idx]
kkt_rmse_mean = np.array(kkt_rmse_mean)[idx]
kkt_rmse_ci   = np.array(kkt_rmse_ci)[idx]

nn_viol_mean = np.array(nn_viol_mean)[idx]
nn_viol_ci   = np.array(nn_viol_ci)[idx]
kkt_viol_mean = np.array(kkt_viol_mean)[idx]
kkt_viol_ci   = np.array(kkt_viol_ci)[idx]

nn_time_mean = np.array(nn_time_mean)[idx]
nn_time_ci   = np.array(nn_time_ci)[idx]
kkt_time_mean = np.array(kkt_time_mean)[idx]
kkt_time_ci   = np.array(kkt_time_ci)[idx]
# =========================
# Plot 1: RMSE
# =========================
# plt.figure(figsize=(7,5))
# plt.errorbar(x_vals, nn_rmse_mean, yerr=nn_rmse_ci, marker='o', capsize=4, label='NN')
# plt.errorbar(x_vals, kkt_rmse_mean, yerr=kkt_rmse_ci, marker='o', capsize=4, label='KKT')
# plt.xlabel("Number of sample points")
# plt.ylabel("RMSE")
# plt.title("Piece-wise Linearization KKT vs Standard NN")
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.savefig("rmse_with_ci.png", dpi=300)
# plt.show()

# # =========================
# # Plot 2: Violation
# # =========================
# plt.figure(figsize=(7,5))
# plt.errorbar(x_vals, nn_viol_mean, yerr=nn_viol_ci, marker='o', capsize=4, label='NN')
# plt.errorbar(x_vals, kkt_viol_mean, yerr=kkt_viol_ci, marker='o', capsize=4, label='KKT')
# plt.xlabel("Number of sample points")
# plt.ylabel("Violation")
# plt.title("Piece-wise Linearization KKT vs Standard NN")
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.savefig("violation_with_ci.png", dpi=300)
# plt.show()

# # =========================
# # Plot 3: Inference Time
# # =========================
# plt.figure(figsize=(7,5))
# plt.errorbar(x_vals, nn_time_mean, yerr=nn_time_ci, marker='o', capsize=4, label='NN')
# plt.errorbar(x_vals, kkt_time_mean, yerr=kkt_time_ci, marker='o', capsize=4, label='KKT')
# plt.xlabel("Number of sample points")
# plt.ylabel("Inference Time (s)")
# plt.title("Piece-wise Linearization KKT vs Standard NN")
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.savefig("time_with_ci.png", dpi=300)
# plt.show()

# ---------------- RMSE ----------------
plt.figure(figsize=(7,5))
plt.plot(x_vals, nn_rmse_mean, marker='o', linewidth=2, label='NN')
plt.fill_between(
    x_vals,
    np.array(nn_rmse_mean) - np.array(nn_rmse_ci),
    np.array(nn_rmse_mean) + np.array(nn_rmse_ci),
    alpha=0.2, label='NN 95% CI'
)

plt.plot(x_vals, kkt_rmse_mean, marker='o', linewidth=2, label='KKT')
plt.fill_between(
    x_vals,
    np.array(kkt_rmse_mean) - np.array(kkt_rmse_ci),
    np.array(kkt_rmse_mean) + np.array(kkt_rmse_ci),
    alpha=0.2, label='KKT 95% CI'
)

plt.xlabel("Number of sample points")
plt.ylabel("RMSE")
plt.title("Piece-wise Linearization KKT vs Standard NN")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("rmse_with_ci_band.png", dpi=300)
plt.show()


# ---------------- Violation ----------------
plt.figure(figsize=(7,5))
plt.plot(x_vals, nn_viol_mean, marker='o', linewidth=2, label='NN')
plt.fill_between(
    x_vals,
    np.array(nn_viol_mean) - np.array(nn_viol_ci),
    np.array(nn_viol_mean) + np.array(nn_viol_ci),
    alpha=0.2
)

plt.plot(x_vals, kkt_viol_mean, marker='o', linewidth=2, label='KKT')
plt.fill_between(
    x_vals,
    np.array(kkt_viol_mean) - np.array(kkt_viol_ci),
    np.array(kkt_viol_mean) + np.array(kkt_viol_ci),
    alpha=0.2
)

plt.xlabel("Number of sample points")
plt.ylabel("Violation")
plt.title("Piece-wise Linearization KKT vs Standard NN")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("violation_with_ci_band.png", dpi=300)
plt.show()


# ---------------- Inference time ----------------
plt.figure(figsize=(7,5))
plt.plot(x_vals, nn_time_mean, marker='o', linewidth=2, label='NN')
plt.fill_between(
    x_vals,
    np.array(nn_time_mean) - np.array(nn_time_ci),
    np.array(nn_time_mean) + np.array(nn_time_ci),
    alpha=0.2
)

plt.plot(x_vals, kkt_time_mean, marker='o', linewidth=2, label='KKT')
plt.fill_between(
    x_vals,
    np.array(kkt_time_mean) - np.array(kkt_time_ci),
    np.array(kkt_time_mean) + np.array(kkt_time_ci),
    alpha=0.2
)

plt.xlabel("Number of sample points")
plt.ylabel("Inference Time (s)")
plt.title("Piece-wise Linearization KKT vs Standard NN")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("time_with_ci_band.png", dpi=300)
plt.show()