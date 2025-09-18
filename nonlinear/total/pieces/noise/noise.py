# noise.py — add Gaussian noise to Ca, Cb, Cc ONLY, based on existing data.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATASET_PATH = "./data.csv"   # produced by generate_data.py
SEED = 42                     # reproducible noise
SAVE_PLOTS = True             # set False on headless servers

def gaussian_noise_only_last_three(df, noise_std, output_path, multiplier=3.0, seed=SEED):
    """
    Adds Gaussian noise to the LAST THREE numeric columns ONLY (expected: Ca, Cb, Cc).
    Noise per col ~ N(0, multiplier * noise_std * std(col)).
    Writes to `output_path` and returns the noisy DataFrame.
    """
    rng = np.random.default_rng(seed)
    noisy = df.copy()

    # Identify last three numeric columns (should be Ca, Cb, Cc for your data.csv)
    num_cols = noisy.select_dtypes(include=[np.number]).columns
    if len(num_cols) < 3:
        raise ValueError("Expected at least 3 numeric columns (Temperature, Ca, Cb, Cc).")
    last_three = num_cols[-3:]  # typically ['Ca', 'Cb', 'Cc']
    print("Last three numeric columns to modify:", list(last_three))

    # Add column-wise Gaussian noise (float64-safe)
    for col in last_three:
        col_std = float(noisy[col].to_numpy(dtype=np.float64).std(ddof=0))
        sigma = float(multiplier * noise_std * col_std)
        noise = rng.normal(0.0, sigma, size=len(noisy)).astype(np.float64)
        noisy[col] = noisy[col].to_numpy(dtype=np.float64) + noise
        print(f"[{col}] sigma={sigma:.6e}, noise_mean={noise.mean():.6e}, noise_std={noise.std(ddof=0):.6e}")

    noisy.to_csv(output_path, index=False)
    print(f"Saved noisy dataset -> {output_path}")
    return noisy

def plot_original_vs_noisy_ca(df_clean, df_noisy, title, fname=None):
    plt.figure(figsize=(9, 5))
    plt.plot(df_clean['Temperature (T)'], df_clean['Ca'], label='Original Ca', alpha=0.9)
    plt.plot(df_noisy['Temperature (T)'], df_noisy['Ca'], label='Noisy Ca', alpha=0.9)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Ca (mol/L)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if SAVE_PLOTS and fname:
        plt.savefig(fname, dpi=150)
    plt.show()

def main():
    # 1) Load clean data.csv created by generate_data.py (do NOT mutate it)
    df = pd.read_csv(DATASET_PATH)
    # Cast numeric to float64 to be safe
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].astype(np.float64)

    # 2) Create three noisy variants (only Ca, Cb, Cc get noise)
    small = gaussian_noise_only_last_three(df, noise_std=0.01, output_path="small_noisy_data.csv", multiplier=3.0, seed=SEED)
    moderate = gaussian_noise_only_last_three(df, noise_std=0.05, output_path="moderate_noisy_data.csv", multiplier=3.0, seed=SEED)
    large = gaussian_noise_only_last_three(df, noise_std=0.10, output_path="large_noisy_data.csv", multiplier=3.0, seed=SEED)

    # 3) (Optional) Quick visual checks: Ca only
    plot_original_vs_noisy_ca(df, small,    "Small Noise - Ca",    "small_noise_ca.png")
    plot_original_vs_noisy_ca(df, moderate, "Moderate Noise - Ca", "moderate_noise_ca.png")
    plot_original_vs_noisy_ca(df, large,    "Large Noise - Ca",    "large_noise_ca.png")

if __name__ == "__main__":
    np.set_printoptions(precision=12, suppress=False)
    main()
