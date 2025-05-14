# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import ttest_rel

# def main():
#     # Path to your CSV file with columns: Iteration, NN_Error, KKT_Error
#     csv_file_path = r"C:/Users/Fateme/Desktop/Research/CSTR/noise/new12/epoch_errors.csv"

#     # Read the CSV file
#     df = pd.read_csv(csv_file_path)

#     # Extract the error columns
#     nn_errors = df["NN_Error"].values
#     kkt_errors = df["KKT_Error"].values

#     # Perform a paired t-test
#     t_stat, p_value = ttest_rel(nn_errors, kkt_errors)

#     print("=== Paired T-Test Results ===")
#     print(f"T-statistic: {t_stat}")
#     print(f"P-value:     {p_value}")

#     # Optional interpretation
#     alpha = 0.05  # significance level
#     if p_value < alpha:
#         print(f"Reject H₀: KKThPINN and NN have significantly different performance (p = {p_value:.4f})")
#         if np.mean(kkt_errors) < np.mean(nn_errors):
#             print("KKThPINN performs significantly better.")
#         else:
#             print("NN performs significantly better.")
#     else:
#         print(f"Fail to reject H₀: No significant difference between KKThPINN and NN (p = {p_value:.4f})")

# if __name__ == "__main__":
#     main()


# csv_file_path = r"C:/Users/Fateme/Desktop/Research/CSTR/noise/new12/epoch_errors.csv"    
# df = pd.read_csv(csv_file_path)
# # Drop any potential missing values
# nn_errors = df["NN_Error"].dropna()
# kkt_errors = df["KKT_Error"].dropna()

# # Create a plot with histograms for both error distributions
# plt.figure(figsize=(10, 6))
# bins = 10  # You can adjust the number of bins as needed

# plt.hist(nn_errors, bins=bins, density=True, alpha=0.5, label='NN Error')
# plt.hist(kkt_errors, bins=bins, density=True, alpha=0.5, label='KKT Error')

# plt.xlabel("Error Value")
# plt.ylabel("Density")
# plt.title("Distribution of NN and KKT Errors")
# plt.legend()
# plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

def main():
    # Path to your CSV file with columns: Iteration, NN_Error, KKT_Error
    csv_file_path = r"C:/Users/Fateme/Desktop/Research/CSTR/600/experiment_epoch_errors.csv"

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
    print(f"NN Training Error - Mean: {nn_mean:.10f}, Variance: {nn_variance:.10f}")
    print(f"KKThPINN Training Error - Mean: {kkt_mean:.10f}, Variance: {kkt_variance:.10f}")

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


# csv_file_path = r"C:/Users/Fateme/Desktop/Research/CSTR/noise/new12/epoch_errors.csv"    
# df = pd.read_csv(csv_file_path)
# # Drop any potential missing values
# nn_errors = df["NN_Error"].dropna()
# kkt_errors = df["KKT_Error"].dropna()

# # Create a plot with histograms for both error distributions
# plt.figure(figsize=(10, 6))
# bins = 10  # You can adjust the number of bins as needed

# plt.hist(nn_errors, bins=bins, density=True, alpha=0.5, label='NN Error')
# plt.hist(kkt_errors, bins=bins, density=True, alpha=0.5, label='KKT Error')

# plt.xlabel("Error Value")
# plt.ylabel("Density")
# plt.title("Distribution of NN and KKT Errors")
# plt.legend()
# plt.show()

