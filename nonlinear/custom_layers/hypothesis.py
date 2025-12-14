import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

def main():
    # Path to your CSV file with columns: Iteration, NN_Error, KKT_Error
    # csv_file_path = r"./results_archive/800/20250924-161548/experiment_epoch_errors.csv"
    csv_file_path = r"./experiment_epoch_errors.csv"

    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Extract the error columns
    nn_errors = df["NN_Experiment_Error"].values
    kkt_errors = df["KKThPINN_Experiment_Error"].values

    # Print raw data to verify
    print("Raw NN Errors:", nn_errors)
    print("Raw KKThPINN Errors:", kkt_errors)

    # Calculate the mean and sample variance (ddof=1 for unbiased estimate)
    nn_mean = np.mean(nn_errors)
    nn_variance = np.var(nn_errors, ddof=1)  # Sample variance
    kkt_mean = np.mean(kkt_errors)
    kkt_variance = np.var(kkt_errors, ddof=1)  # Sample variance

    # Print the mean and variance with higher precision
    print("\n=== Summary Statistics ===")
    print(f"NN Experiment Error - Mean: {nn_mean:.10f}, Variance: {nn_variance:.10f}")
    print(f"KKThPINN Experiment Error - Mean: {kkt_mean:.10f}, Variance: {kkt_variance:.10f}")

    # Perform a paired t-test
    t_stat, p_value = ttest_rel(nn_errors, kkt_errors)

    print("\n=== Paired T-Test Results ===")
    print(f"T-statistic: {t_stat}")
    print(f"P-value:     {p_value}")

    # Optional interpretation
    alpha = 0.05  # significance level
    if p_value < alpha:
        print(f"Reject H₀: KKThPINN and NN have significantly different performance (p = {p_value:.4f})")
        if np.mean(kkt_errors) < np.mean(nn_errors):
            print("KKThPINN performs significantly better.")
        else:
            print("NN performs significantly better.")
    else:
        print(f"Fail to reject H₀: No significant difference between KKThPINN and NN (p = {p_value:.4f})")

if __name__ == "__main__":
    main()

import numpy as np, pandas as pd, scipy.stats as st

df = pd.read_csv("experiment_epoch_errors.csv")
nn  = df["NN_Experiment_Error"].dropna().to_numpy()
kkt = df["KKThPINN_Experiment_Error"].dropna().to_numpy()

t, p = st.ttest_ind(kkt, nn, equal_var=False)  # Welch t-test
d = (kkt.mean() - nn.mean()) / np.sqrt(0.5*(kkt.var(ddof=1)+nn.var(ddof=1)))
print(f"Welch t-test p={p:.4g}, Cohen's d={d:.3f}")
