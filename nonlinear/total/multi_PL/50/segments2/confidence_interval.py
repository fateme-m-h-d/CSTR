# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import ttest_1samp

# SCENARIOS = [1, 2, 3, 5, 7, 9, 11]
# NC_REGIONS = 3

# x_vals = []

# nn_rmse_mean, nn_rmse_ci = [], []
# kkt_rmse_mean, kkt_rmse_ci = [], []

# nn_violnl_mean, nn_violnl_ci = [], []
# kkt_violnl_mean, kkt_violnl_ci = [], []

# nn_time_mean, nn_time_ci = [], []
# kkt_time_mean, kkt_time_ci = [], []


# def mean_and_ci(arr):
#     arr = np.asarray(arr, dtype=float)
#     ci = ttest_1samp(arr, popmean=0).confidence_interval(0.95)
#     mean = arr.mean()
#     half_width = (ci.high - ci.low) / 2
#     return mean, half_width


# for nT in SCENARIOS:
#     fname = f"experiment_epoch_errors_nseg_{nT}.csv"
#     if not os.path.exists(fname):
#         print(f"Skipping missing file: {fname}")
#         continue

#     df = pd.read_csv(fname)

#     # x-axis = number of T segments
#     x_vals.append(nT)
#     # If you want total number of regions instead, use:
#     # x_vals.append(nT * NC_REGIONS)

#     # ---------------- RMSE ----------------
#     nn = df["NN_Experiment_RMSE"].dropna().to_numpy()
#     kkt = df["KKThPINN_Experiment_RMSE"].dropna().to_numpy()

#     m, c = mean_and_ci(nn)
#     nn_rmse_mean.append(m)
#     nn_rmse_ci.append(c)

#     m, c = mean_and_ci(kkt)
#     kkt_rmse_mean.append(m)
#     kkt_rmse_ci.append(c)

#     # ---------------- Nonlinear violation ----------------
#     nn = df["NN_Experiment_VIOL_NL"].dropna().to_numpy()
#     kkt = df["KKThPINN_Experiment_VIOL_NL"].dropna().to_numpy()

#     m, c = mean_and_ci(nn)
#     nn_violnl_mean.append(m)
#     nn_violnl_ci.append(c)

#     m, c = mean_and_ci(kkt)
#     kkt_violnl_mean.append(m)
#     kkt_violnl_ci.append(c)

#     # ---------------- Experiment time ----------------
#     nn = df["NN_Experiment_Time_sec"].dropna().to_numpy()
#     kkt = df["KKThPINN_Experiment_Time_sec"].dropna().to_numpy()

#     m, c = mean_and_ci(nn)
#     nn_time_mean.append(m)
#     nn_time_ci.append(c)

#     m, c = mean_and_ci(kkt)
#     kkt_time_mean.append(m)
#     kkt_time_ci.append(c)


# # Convert to arrays
# x_vals = np.array(x_vals)

# nn_rmse_mean = np.array(nn_rmse_mean)
# nn_rmse_ci = np.array(nn_rmse_ci)
# kkt_rmse_mean = np.array(kkt_rmse_mean)
# kkt_rmse_ci = np.array(kkt_rmse_ci)

# nn_violnl_mean = np.array(nn_violnl_mean)
# nn_violnl_ci = np.array(nn_violnl_ci)
# kkt_violnl_mean = np.array(kkt_violnl_mean)
# kkt_violnl_ci = np.array(kkt_violnl_ci)

# nn_time_mean = np.array(nn_time_mean)
# nn_time_ci = np.array(nn_time_ci)
# kkt_time_mean = np.array(kkt_time_mean)
# kkt_time_ci = np.array(kkt_time_ci)

# # Sort by x
# idx = np.argsort(x_vals)
# x_vals = x_vals[idx]

# nn_rmse_mean = nn_rmse_mean[idx]
# nn_rmse_ci = nn_rmse_ci[idx]
# kkt_rmse_mean = kkt_rmse_mean[idx]
# kkt_rmse_ci = kkt_rmse_ci[idx]

# nn_violnl_mean = nn_violnl_mean[idx]
# nn_violnl_ci = nn_violnl_ci[idx]
# kkt_violnl_mean = kkt_violnl_mean[idx]
# kkt_violnl_ci = kkt_violnl_ci[idx]

# nn_time_mean = nn_time_mean[idx]
# nn_time_ci = nn_time_ci[idx]
# kkt_time_mean = kkt_time_mean[idx]
# kkt_time_ci = kkt_time_ci[idx]


# # ================= RMSE plot =================
# plt.figure(figsize=(7, 5))

# plt.plot(x_vals, nn_rmse_mean, marker='o', linewidth=2, label='NN')
# plt.fill_between(
#     x_vals,
#     nn_rmse_mean - nn_rmse_ci,
#     nn_rmse_mean + nn_rmse_ci,
#     alpha=0.2,
#     label='NN 95% CI'
# )

# plt.plot(x_vals, kkt_rmse_mean, marker='o', linewidth=2, label='KKT')
# plt.fill_between(
#     x_vals,
#     kkt_rmse_mean - kkt_rmse_ci,
#     kkt_rmse_mean + kkt_rmse_ci,
#     alpha=0.2,
#     label='KKT 95% CI'
# )

# plt.xlabel("Number of T segments")
# # plt.xlabel("Total number of regions")
# plt.ylabel("RMSE")
# plt.title("RMSE vs Number of Segments")
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.savefig("rmse_vs_segments.png", dpi=300)
# plt.show()


# # ================= Nonlinear violation plot =================
# plt.figure(figsize=(7, 5))

# plt.plot(x_vals, nn_violnl_mean, marker='o', linewidth=2, label='NN')
# plt.fill_between(
#     x_vals,
#     nn_violnl_mean - nn_violnl_ci,
#     nn_violnl_mean + nn_violnl_ci,
#     alpha=0.2,
#     label='NN 95% CI'
# )

# plt.plot(x_vals, kkt_violnl_mean, marker='o', linewidth=2, label='KKT')
# plt.fill_between(
#     x_vals,
#     kkt_violnl_mean - kkt_violnl_ci,
#     kkt_violnl_mean + kkt_violnl_ci,
#     alpha=0.2,
#     label='KKT 95% CI'
# )

# plt.xlabel("Number of T segments")
# # plt.xlabel("Total number of regions")
# plt.ylabel("Nonlinear violation")
# plt.title("Nonlinear Violation vs Number of Segments")
# plt.yscale("log")
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.savefig("viol_nl_vs_segments.png", dpi=300)
# plt.show()


# # ================= Experiment time plot =================
# plt.figure(figsize=(7, 5))

# plt.plot(x_vals, nn_time_mean, marker='o', linewidth=2, label='NN')
# plt.fill_between(
#     x_vals,
#     nn_time_mean - nn_time_ci,
#     nn_time_mean + nn_time_ci,
#     alpha=0.2,
#     label='NN 95% CI'
# )

# plt.plot(x_vals, kkt_time_mean, marker='o', linewidth=2, label='KKT')
# plt.fill_between(
#     x_vals,
#     kkt_time_mean - kkt_time_ci,
#     kkt_time_mean + kkt_time_ci,
#     alpha=0.2,
#     label='KKT 95% CI'
# )

# plt.xlabel("Number of T segments")
# # plt.xlabel("Total number of regions")
# plt.ylabel("Experiment time (sec)")
# plt.title("Experiment Time vs Number of Segments")
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.savefig("time_vs_segments.png", dpi=300)
# plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

SCENARIOS = [1, 2, 3, 5, 7, 9, 11]

PLOTS = [
    {
        "nn_col": "NN_Experiment_RMSE",
        "kkt_col": "KKThPINN_Experiment_RMSE",
        "ylabel": "RMSE",
        "title": "RMSE vs Number of Segments",
        "save": "rmse_vs_segments_pooled_nn.png",
        "logy": False,
    },
    {
        "nn_col": "NN_Experiment_Time_sec",
        "kkt_col": "KKThPINN_Experiment_Time_sec",
        "ylabel": "Experiment time (sec)",
        "title": "Experiment Time vs Number of Segments",
        "save": "time_vs_segments_pooled_nn.png",
        "logy": False,
    },
    {
        "nn_col": "NN_Experiment_VIOL_NL",
        "kkt_col": "KKThPINN_Experiment_VIOL_NL",
        "ylabel": "Nonlinear violation",
        "title": "Nonlinear Violation vs Number of Segments",
        "save": "viol_nl_vs_segments_pooled_nn.png",
        "logy": True,
    },
]

for cfg in PLOTS:
    nn_all = []
    kkt_mean = []
    kkt_ci = []

    for nT in SCENARIOS:
        df = pd.read_csv(f"experiment_epoch_errors_nseg_{nT}.csv")

        nn_vals = df[cfg["nn_col"]].dropna().to_numpy()
        kkt_vals = df[cfg["kkt_col"]].dropna().to_numpy()

        nn_all.extend(nn_vals.tolist())

        kkt_conf = ttest_1samp(kkt_vals, popmean=0).confidence_interval(0.95)
        kkt_mean.append(kkt_vals.mean())
        kkt_ci.append((kkt_conf.high - kkt_conf.low) / 2)

    nn_all = np.array(nn_all)

    nn_conf = ttest_1samp(nn_all, popmean=0).confidence_interval(0.95)
    nn_mean = nn_all.mean()
    nn_ci = (nn_conf.high - nn_conf.low) / 2

    x_vals = np.array(SCENARIOS)
    kkt_mean = np.array(kkt_mean)
    kkt_ci = np.array(kkt_ci)

    plt.figure(figsize=(9, 6))

    # pooled NN baseline
    plt.plot(
        x_vals,
        np.full_like(x_vals, nn_mean, dtype=float),
        marker="o",
        linewidth=2.5,
        label="NN"
    )
    plt.fill_between(
        x_vals,
        nn_mean - nn_ci,
        nn_mean + nn_ci,
        alpha=0.2,
        label="NN 95% CI"
    )

    # KKT curve
    plt.plot(
        x_vals,
        kkt_mean,
        marker="o",
        linewidth=2.5,
        label="KKT"
    )
    plt.fill_between(
        x_vals,
        kkt_mean - kkt_ci,
        kkt_mean + kkt_ci,
        alpha=0.2,
        label="KKT 95% CI"
    )

    plt.xlabel("Number of T segments")
    plt.ylabel(cfg["ylabel"])
    plt.title(cfg["title"])
    plt.grid(True, alpha=0.3)

    if cfg["logy"]:
        plt.yscale("log")

    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg["save"], dpi=300)
    plt.show()

    print(f"\n{cfg['title']}")
    print(f"NN pooled mean = {nn_mean:.6f}")
    print(f"NN pooled 95% CI half-width = {nn_ci:.6f}")