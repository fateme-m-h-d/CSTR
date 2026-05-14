# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import ttest_1samp

# SCENARIOS = [1, 2, 3, 5, 7, 9, 11]

# x_vals = SCENARIOS
# nn_rmse_mean, nn_rmse_ci = [], []
# kkt_rmse_mean, kkt_rmse_ci = [], []

# nn_viol_mean, nn_viol_ci = [], []
# kkt_viol_mean, kkt_viol_ci = [], []

# nn_time_mean, nn_time_ci = [], []
# kkt_time_mean, kkt_time_ci = [], []

# for n in SCENARIOS:
#     df = pd.read_csv(f"experiment_epoch_errors_nseg_{n}.csv")
    

#     # ----- RMSE -----
#     nn = df["NN_Experiment_RMSE"].dropna().to_numpy()
#     kkt = df["KKThPINN_Experiment_RMSE"].dropna().to_numpy()

#     nn_ci = ttest_1samp(nn, popmean=0).confidence_interval(0.95)
#     kkt_ci = ttest_1samp(kkt, popmean=0).confidence_interval(0.95)

#     nn_rmse_mean.append(nn.mean())
#     kkt_rmse_mean.append(kkt.mean())
#     nn_rmse_ci.append((nn_ci.high - nn_ci.low) / 2)
#     kkt_rmse_ci.append((kkt_ci.high - kkt_ci.low) / 2)

#     # ----- Violation -----
#     nn = df["NN_Experiment_VIOL_NL"].dropna().to_numpy()
#     kkt = df["KKThPINN_Experiment_VIOL_NL"].dropna().to_numpy()

#     nn_ci = ttest_1samp(nn, popmean=0).confidence_interval(0.95)
#     kkt_ci = ttest_1samp(kkt, popmean=0).confidence_interval(0.95)

#     nn_viol_mean.append(nn.mean())
#     kkt_viol_mean.append(kkt.mean())
#     nn_viol_ci.append((nn_ci.high - nn_ci.low) / 2)
#     kkt_viol_ci.append((kkt_ci.high - kkt_ci.low) / 2)

#     # ----- Inference time -----
#     nn = df["NN_Experiment_Time_sec"].dropna().to_numpy()
#     kkt = df["KKThPINN_Experiment_Time_sec"].dropna().to_numpy()

#     nn_ci = ttest_1samp(nn, popmean=0).confidence_interval(0.95)
#     kkt_ci = ttest_1samp(kkt, popmean=0).confidence_interval(0.95)

#     nn_time_mean.append(nn.mean())
#     kkt_time_mean.append(kkt.mean())
#     nn_time_ci.append((nn_ci.high - nn_ci.low) / 2)
#     kkt_time_ci.append((kkt_ci.high - kkt_ci.low) / 2)

# # =========================
# # Plot 1: RMSE
# # =========================
# # plt.figure(figsize=(7,5))
# # plt.errorbar(x_vals, nn_rmse_mean, yerr=nn_rmse_ci, marker='o', capsize=4, label='NN')
# # plt.errorbar(x_vals, kkt_rmse_mean, yerr=kkt_rmse_ci, marker='o', capsize=4, label='KKT')
# # plt.xlabel("Number of sample points")
# # plt.ylabel("RMSE")
# # plt.title("Piece-wise Linearization KKT vs Standard NN")
# # plt.grid(True, alpha=0.3)
# # plt.legend()
# # plt.tight_layout()
# # plt.savefig("rmse_with_ci.png", dpi=300)
# # plt.show()

# # # =========================
# # # Plot 2: Violation
# # # =========================
# # plt.figure(figsize=(7,5))
# # plt.errorbar(x_vals, nn_viol_mean, yerr=nn_viol_ci, marker='o', capsize=4, label='NN')
# # plt.errorbar(x_vals, kkt_viol_mean, yerr=kkt_viol_ci, marker='o', capsize=4, label='KKT')
# # plt.xlabel("Number of sample points")
# # plt.ylabel("Violation")
# # plt.title("Piece-wise Linearization KKT vs Standard NN")
# # plt.grid(True, alpha=0.3)
# # plt.legend()
# # plt.tight_layout()
# # plt.savefig("violation_with_ci.png", dpi=300)
# # plt.show()

# # # =========================
# # # Plot 3: Inference Time
# # # =========================
# # plt.figure(figsize=(7,5))
# # plt.errorbar(x_vals, nn_time_mean, yerr=nn_time_ci, marker='o', capsize=4, label='NN')
# # plt.errorbar(x_vals, kkt_time_mean, yerr=kkt_time_ci, marker='o', capsize=4, label='KKT')
# # plt.xlabel("Number of sample points")
# # plt.ylabel("Inference Time (s)")
# # plt.title("Piece-wise Linearization KKT vs Standard NN")
# # plt.grid(True, alpha=0.3)
# # plt.legend()
# # plt.tight_layout()
# # plt.savefig("time_with_ci.png", dpi=300)
# # plt.show()

# # ---------------- RMSE ----------------
# plt.figure(figsize=(7,5))
# plt.plot(x_vals, nn_rmse_mean, marker='o', linewidth=2, label='NN')
# plt.fill_between(
#     x_vals,
#     np.array(nn_rmse_mean) - np.array(nn_rmse_ci),
#     np.array(nn_rmse_mean) + np.array(nn_rmse_ci),
#     alpha=0.2, label='NN 95% CI'
# )

# plt.plot(x_vals, kkt_rmse_mean, marker='o', linewidth=2, label='KKT')
# plt.fill_between(
#     x_vals,
#     np.array(kkt_rmse_mean) - np.array(kkt_rmse_ci),
#     np.array(kkt_rmse_mean) + np.array(kkt_rmse_ci),
#     alpha=0.2, label='KKT 95% CI'
# )

# plt.xlabel("Number of segments")
# plt.ylabel("RMSE")
# plt.title("Piece-wise Linearization KKT vs Standard NN")
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.savefig("rmse_with_ci_band.png", dpi=300)
# plt.show()


# # ---------------- Violation ----------------
# plt.figure(figsize=(7,5))
# plt.plot(x_vals, nn_viol_mean, marker='o', linewidth=2, label='NN')
# plt.fill_between(
#     x_vals,
#     np.array(nn_viol_mean) - np.array(nn_viol_ci),
#     np.array(nn_viol_mean) + np.array(nn_viol_ci),
#     alpha=0.2
# )

# plt.plot(x_vals, kkt_viol_mean, marker='o', linewidth=2, label='KKT')
# plt.fill_between(
#     x_vals,
#     np.array(kkt_viol_mean) - np.array(kkt_viol_ci),
#     np.array(kkt_viol_mean) + np.array(kkt_viol_ci),
#     alpha=0.2
# )

# plt.xlabel("Number of segments")
# plt.ylabel("Violation")
# plt.yscale("log")
# plt.title("Piece-wise Linearization KKT vs Standard NN")
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.savefig("violation_with_ci_band.png", dpi=300)
# plt.show()


# # ---------------- Inference time ----------------
# plt.figure(figsize=(7,5))
# plt.plot(x_vals, nn_time_mean, marker='o', linewidth=2, label='NN')
# plt.fill_between(
#     x_vals,
#     np.array(nn_time_mean) - np.array(nn_time_ci),
#     np.array(nn_time_mean) + np.array(nn_time_ci),
#     alpha=0.2
# )

# plt.plot(x_vals, kkt_time_mean, marker='o', linewidth=2, label='KKT')
# plt.fill_between(
#     x_vals,
#     np.array(kkt_time_mean) - np.array(kkt_time_ci),
#     np.array(kkt_time_mean) + np.array(kkt_time_ci),
#     alpha=0.2
# )

# plt.xlabel("Number of segments")
# plt.ylabel("Inference Time (s)")
# plt.title("Piece-wise Linearization KKT vs Standard NN")
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.savefig("time_with_ci_band.png", dpi=300)
# plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

SCENARIOS = [1, 2, 3, 5, 7, 9, 11]
MODE = "hard"   # change to "sigmoid" when you rerun that case

x_vals = SCENARIOS
nn_rmse_mean, nn_rmse_ci = [], []
kkt_rmse_mean, kkt_rmse_ci = [], []

nn_viol_mean, nn_viol_ci = [], []
kkt_viol_mean, kkt_viol_ci = [], []

nn_time_mean, nn_time_ci = [], []
kkt_time_mean, kkt_time_ci = [], []


def mean_and_ci(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]

    if len(arr) == 0:
        return np.nan, np.nan
    elif len(arr) == 1:
        return float(arr.mean()), 0.0
    else:
        ci = ttest_1samp(arr, popmean=0).confidence_interval(0.95)
        return float(arr.mean()), float((ci.high - ci.low) / 2)


for n in SCENARIOS:
    df = pd.read_csv(f"experiment_epoch_errors_nseg_{n}.csv")

    # ----- RMSE -----
    nn = df["NN_Experiment_RMSE"].dropna().to_numpy()
    kkt = df["KKThPINN_Experiment_RMSE"].dropna().to_numpy()

    nn_mean, nn_ci = mean_and_ci(nn)
    kkt_mean, kkt_ci = mean_and_ci(kkt)

    nn_rmse_mean.append(nn_mean)
    kkt_rmse_mean.append(kkt_mean)
    nn_rmse_ci.append(nn_ci)
    kkt_rmse_ci.append(kkt_ci)

    # ----- Nonlinear Violation -----
    nn = df["NN_Experiment_VIOL_NL"].dropna().to_numpy()
    kkt = df["KKThPINN_Experiment_VIOL_NL"].dropna().to_numpy()

    nn_mean, nn_ci = mean_and_ci(nn)
    kkt_mean, kkt_ci = mean_and_ci(kkt)

    nn_viol_mean.append(nn_mean)
    kkt_viol_mean.append(kkt_mean)
    nn_viol_ci.append(nn_ci)
    kkt_viol_ci.append(kkt_ci)

    # ----- Inference time -----
    nn = df["NN_Experiment_Time_sec"].dropna().to_numpy()
    kkt = df["KKThPINN_Experiment_Time_sec"].dropna().to_numpy()

    nn_mean, nn_ci = mean_and_ci(nn)
    kkt_mean, kkt_ci = mean_and_ci(kkt)

    nn_time_mean.append(nn_mean)
    kkt_time_mean.append(kkt_mean)
    nn_time_ci.append(nn_ci)
    kkt_time_ci.append(kkt_ci)


# =========================
# Save summary CSV
# =========================
summary_df = pd.DataFrame({
    "n_segments": x_vals,

    "NN_RMSE_mean": nn_rmse_mean,
    "KKThPINN_RMSE_mean": kkt_rmse_mean,
   
    "NN_VIOL_NL_mean": nn_viol_mean,
    "KKThPINN_VIOL_NL_mean": kkt_viol_mean,
    

    "NN_TIME_mean_sec": nn_time_mean,
    "KKThPINN_TIME_mean_sec": kkt_time_mean,
})

summary_df.to_csv(f"confidence_interval_summary_{MODE}.csv", index=False)
print(f"Saved summary CSV: confidence_interval_summary_{MODE}.csv")


# ---------------- RMSE ----------------
plt.figure(figsize=(7, 5))
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

plt.xlabel("Number of segments")
plt.ylabel("RMSE")
plt.title("Piece-wise Linearization KKT vs Standard NN")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"rmse_with_ci_band_{MODE}.png", dpi=300)
plt.show()


# ---------------- Violation ----------------
plt.figure(figsize=(7, 5))
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

plt.xlabel("Number of segments")
plt.ylabel("Violation")
plt.yscale("log")
plt.title("Piece-wise Linearization KKT vs Standard NN")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"violation_with_ci_band_{MODE}.png", dpi=300)
plt.show()


# ---------------- Inference time ----------------
plt.figure(figsize=(7, 5))
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

plt.xlabel("Number of segments")
plt.ylabel("Inference Time (s)")
plt.title("Piece-wise Linearization KKT vs Standard NN")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"time_with_ci_band_{MODE}.png", dpi=300)
plt.show()