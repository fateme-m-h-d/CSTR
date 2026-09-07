import argparse

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = ["Ca", "Cb", "Cc"]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--noise_level", type=float, default=0.05)
    parser.add_argument("--clean_csv", default="data_fixed.csv")
    parser.add_argument("--out_csv", default="data.csv")

    args = parser.parse_args()

    clean = pd.read_csv(args.clean_csv)
    split = np.load("split_indices.npz")

    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]
    center_idx = split["center_idx"]

    clean_outputs = clean[OUTPUT_COLUMNS].to_numpy(dtype=float)

    # Standard deviation calculated only from clean training outputs.
    output_std = clean_outputs[train_idx].std(
        axis=0,
        ddof=0,
    )

    rng = np.random.default_rng(args.seed)

    standard_noise = rng.normal(
        size=clean_outputs.shape,
    )

    noisy_outputs = clean_outputs.copy()

    # Training and validation receive noise.
    noisy_idx = np.concatenate([
        train_idx,
        val_idx,
    ])

    noisy_outputs[noisy_idx] += (
        args.noise_level
        * output_std
        * standard_noise[noisy_idx]
    )

    # Test rows are not changed.
    noisy = clean.copy()
    noisy.loc[:, OUTPUT_COLUMNS] = noisy_outputs
    noisy.to_csv(args.out_csv, index=False)

    print(
        f"Noise seed={args.seed}; "
        f"noise level={args.noise_level:.1%}; "
        f"noisy training={len(train_idx)}; "
        f"noisy validation={len(val_idx)}; "
        f"clean test={len(test_idx)}; "
        f"noisy centers={len(center_idx)}"
    )


if __name__ == "__main__":
    main()