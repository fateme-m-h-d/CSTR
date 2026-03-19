import pandas as pd
import numpy as np
from scipy.stats import ttest_rel

def run_paired_test(df, nn_col, kkt_col, metric_name):
    nn_vals = df[nn_col].values
    kkt_vals = df[kkt_col].values

    print(f"\n================ {metric_name} ================")
    print("Raw NN values:", nn_vals)
    print("Raw KKThPINN values:", kkt_vals)

    nn_mean = np.mean(nn_vals)
    nn_variance = np.var(nn_vals, ddof=1)

    kkt_mean = np.mean(kkt_vals)
    kkt_variance = np.var(kkt_vals, ddof=1)

    print("\n=== Summary Statistics ===")
    print(f"NN        - Mean: {nn_mean:.10f}, Variance: {nn_variance:.10f}")
    print(f"KKThPINN  - Mean: {kkt_mean:.10f}, Variance: {kkt_variance:.10f}")

    t_stat, p_value = ttest_rel(nn_vals, kkt_vals)

    print("\n=== Paired T-Test Results ===")
    print(f"T-statistic: {t_stat}")
    print(f"P-value:     {p_value}")

    alpha = 0.05
    if p_value < alpha:
        print(f"Reject H₀: NN and KKThPINN are significantly different for {metric_name} (p = {p_value:.6f})")
        if kkt_mean < nn_mean:
            print(f"KKThPINN performs significantly better in terms of {metric_name}.")
        else:
            print(f"NN performs significantly better in terms of {metric_name}.")
    else:
        print(f"Fail to reject H₀: No significant difference for {metric_name} (p = {p_value:.6f})")


def main():
    csv_file_path = "./experiment_epoch_errors.csv"

    df = pd.read_csv(csv_file_path)

    # RMSE test
    run_paired_test(
        df,
        "NN_Experiment_RMSE",
        "KKThPINN_Experiment_RMSE",
        "RMSE"
    )

    # VIOL test
    run_paired_test(
        df,
        "NN_Experiment_VIOL",
        "KKThPINN_Experiment_VIOL",
        "VIOL"
    )

    # VIOL_NL test
    run_paired_test(
        df,
        "NN_Experiment_VIOL_NL",
        "KKThPINN_Experiment_VIOL_NL",
        "VIOL_NL"
    )


if __name__ == "__main__":
    main()