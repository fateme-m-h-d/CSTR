# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import ttest_1samp

# SCENARIOS = [1, 2, 3, 5, 7, 9, 11]

# PLOTS = [
#     {
#         "nn_col": "NN_Experiment_RMSE",
#         "kkt_col": "KKThPINN_Experiment_RMSE",
#         "ylabel": "RMSE",
#         "save": "rmse_vs_segments_pooled_nn.pdf",
#         "logy": False,
#     },
#     {
#         "nn_col": "NN_Experiment_Time_sec",
#         "kkt_col": "KKThPINN_Experiment_Time_sec",
#         "ylabel": "Experiment time (sec)",
#         "save": "time_vs_segments_pooled_nn.pdf",
#         "logy": False,
#     },
#     {
#         "nn_col": "NN_Experiment_VIOL_NL",
#         "kkt_col": "KKThPINN_Experiment_VIOL_NL",
#         "ylabel": "Nonlinear violation",
#         "save": "viol_nl_vs_segments_pooled_nn.pdf",
#         "logy": True,
#     },
# ]

# for cfg in PLOTS:
#     nn_all = []
#     kkt_mean = []
#     kkt_ci = []

#     for nT in SCENARIOS:
#         df = pd.read_csv(f"experiment_epoch_errors_nseg_{nT}.csv")

#         nn_vals = df[cfg["nn_col"]].dropna().to_numpy()
#         kkt_vals = df[cfg["kkt_col"]].dropna().to_numpy()

#         nn_all.extend(nn_vals.tolist())

#         kkt_conf = ttest_1samp(kkt_vals, popmean=0).confidence_interval(0.95)
#         kkt_mean.append(kkt_vals.mean())
#         kkt_ci.append((kkt_conf.high - kkt_conf.low) / 2)

#     nn_all = np.array(nn_all)

#     nn_conf = ttest_1samp(nn_all, popmean=0).confidence_interval(0.95)
#     nn_mean = nn_all.mean()
#     nn_ci = (nn_conf.high - nn_conf.low) / 2

#     x_vals = np.array(SCENARIOS)
#     kkt_mean = np.array(kkt_mean)
#     kkt_ci = np.array(kkt_ci)

#     plt.figure(figsize=(9, 6))

#     # pooled NN baseline
#     plt.plot(
#         x_vals,
#         np.full_like(x_vals, nn_mean, dtype=float),
#         marker="o",
#         linewidth=2.5,
#         label="NN"
#     )
#     plt.fill_between(
#         x_vals,
#         nn_mean - nn_ci,
#         nn_mean + nn_ci,
#         alpha=0.2,
#         label="NN 95% CI"
#     )

#     # KKT curve
#     plt.plot(
#         x_vals,
#         kkt_mean,
#         marker="o",
#         linewidth=2.5,
#         label="KKT"
#     )
#     plt.fill_between(
#         x_vals,
#         kkt_mean - kkt_ci,
#         kkt_mean + kkt_ci,
#         alpha=0.2,
#         label="KKT 95% CI"
#     )

#     plt.xlabel("Number of T segments")
#     plt.ylabel(cfg["ylabel"])
#     plt.grid(True, alpha=0.3)

#     if cfg["logy"]:
#         plt.yscale("log")

#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(cfg["save"], dpi=300)
#     plt.show()
    
#     print(f"NN pooled mean = {nn_mean:.6f}")
#     print(f"NN pooled 95% CI half-width = {nn_ci:.6f}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

SEGMENT_SCENARIOS = [1, 2, 3, 5, 7, 9, 11]   # number of T segments
NC_REGIONS = 3                              # number of C regions

PLOTS = [
    {
        "nn_col": "NN_Experiment_RMSE",
        "kkt_col": "KKThPINN_Experiment_RMSE",
        "ylabel": "RMSE",
        "save": "rmse_vs_regions_pooled_nn.pdf",
        "logy": False,
    },
    {
        "nn_col": "NN_Experiment_Time_sec",
        "kkt_col": "KKThPINN_Experiment_Time_sec",
        "ylabel": "Experiment time (sec)",
        "save": "time_vs_regions_pooled_nn.pdf",
        "logy": False,
    },
    {
        "nn_col": "NN_Experiment_VIOL_NL",
        "kkt_col": "KKThPINN_Experiment_VIOL_NL",
        "ylabel": "Nonlinear violation",
        "save": "viol_nl_vs_regions_pooled_nn.pdf",
        "logy": True,
    },
]

for cfg in PLOTS:
    nn_all = []
    kkt_mean = []
    kkt_ci = []
    num_regions = []

    for nT in SEGMENT_SCENARIOS:
        df = pd.read_csv(f"experiment_epoch_errors_nseg_{nT}.csv")

        nn_vals = df[cfg["nn_col"]].dropna().to_numpy()
        kkt_vals = df[cfg["kkt_col"]].dropna().to_numpy()

        nn_all.extend(nn_vals.tolist())

        kkt_conf = ttest_1samp(kkt_vals, popmean=0).confidence_interval(0.95)
        kkt_mean.append(kkt_vals.mean())
        kkt_ci.append((kkt_conf.high - kkt_conf.low) / 2)

        # total number of 2D regions
        num_regions.append(nT * NC_REGIONS)

    nn_all = np.array(nn_all)

    nn_conf = ttest_1samp(nn_all, popmean=0).confidence_interval(0.95)
    nn_mean = nn_all.mean()
    nn_ci = (nn_conf.high - nn_conf.low) / 2

    x_vals = np.array(num_regions)
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

    plt.xlabel("Number of regions")
    plt.ylabel(cfg["ylabel"])
    plt.grid(True, alpha=0.3)

    if cfg["logy"]:
        plt.yscale("log")

    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg["save"], dpi=300)
    plt.show()

    print(f"{cfg['save']}")
    print(f"NN pooled mean = {nn_mean:.6f}")
    print(f"NN pooled 95% CI half-width = {nn_ci:.6f}")