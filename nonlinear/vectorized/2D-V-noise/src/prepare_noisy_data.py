import argparse
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_COLUMNS = ["Temperature (T)", "Cao"]
OUTPUT_COLUMNS = ["Ca", "Cb", "Cc"]
# Same physical domain as src/linearization.py.
TMIN, TMAX = 280.0, 460.0
CMIN, CMAX = 0.8, 1.2


def make_split(clean, nT_regions, nC_regions, split_seed=42):
    """Fixed 60/20/20 split, reserving only the active PL centers for training."""
    if nT_regions < 1 or nC_regions < 1:
        raise ValueError("Region counts must be positive integers.")

    T_edges = np.linspace(TMIN, TMAX, nT_regions + 1)
    C_edges = np.linspace(CMIN, CMAX, nC_regions + 1)
    points = clean[INPUT_COLUMNS].to_numpy(dtype=float)
    centers = []
    for T in 0.5 * (T_edges[:-1] + T_edges[1:]):
        for C in 0.5 * (C_edges[:-1] + C_edges[1:]):
            matches = np.flatnonzero(
                np.isclose(points[:, 0], T, rtol=0.0, atol=1e-10)
                & np.isclose(points[:, 1], C, rtol=0.0, atol=1e-10)
            )
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one existing center at T={T}, Cao={C}; "
                    f"found {len(matches)}. Use region counts whose centers "
                    "are present in the clean input CSV."
                )
            centers.append(int(matches[0]))
    center_idx = np.asarray(centers, dtype=int)
    if len(np.unique(center_idx)) != len(center_idx):
        raise ValueError("Each PL region must have its own center row.")

    n = len(clean)
    n_val = n_test = int(0.2 * n)
    n_train = n - n_val - n_test
    if n_val == 0 or n_test == 0 or len(center_idx) > n_train:
        raise ValueError("Not enough observations for this center-reserved split.")

    # Independent of measurement-noise seed and model initialization.
    rng = np.random.RandomState(split_seed)
    shuffled = rng.permutation(n)
    remaining = shuffled[~np.isin(shuffled, center_idx)]
    extra = n_train - len(center_idx)
    train_idx = np.concatenate([center_idx, remaining[:extra]])
    rng.shuffle(train_idx)
    return {
        "train_idx": train_idx,
        "val_idx": remaining[extra:extra + n_val],
        "test_idx": remaining[extra + n_val:],
        "center_idx": center_idx,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--noise_level", type=float, default=0.05)
    parser.add_argument("--nT_regions", type=int, required=True)
    parser.add_argument("--nC_regions", type=int, default=3)
    parser.add_argument("--clean_csv", default="data_clean.csv")
    parser.add_argument("--out_csv", default="data.csv")
    parser.add_argument("--split_path", default="split_indices.npz")
    args = parser.parse_args()
    if not np.isfinite(args.noise_level) or args.noise_level < 0:
        raise ValueError("noise_level must be finite and nonnegative.")
    if Path(args.clean_csv).resolve() == Path(args.out_csv).resolve():
        raise ValueError("The noisy output must not overwrite the clean source.")

    clean = pd.read_csv(args.clean_csv, float_precision="round_trip")
    if list(clean.columns) != INPUT_COLUMNS + OUTPUT_COLUMNS:
        raise ValueError("Expected columns: Temperature (T), Cao, Ca, Cb, Cc.")
    if not np.isfinite(clean.to_numpy(dtype=float)).all():
        raise ValueError("The clean dataset contains nonfinite values.")
    split = make_split(clean, args.nT_regions, args.nC_regions)
    outputs = clean[OUTPUT_COLUMNS].to_numpy(dtype=float)
    output_std = outputs[split["train_idx"]].std(axis=0, ddof=0)
    rng = np.random.default_rng(args.seed)
    draws = rng.standard_normal(size=outputs.shape)
    noisy_outputs = outputs.copy()
    noisy_idx = np.concatenate([split["train_idx"], split["val_idx"]])
    noisy_outputs[noisy_idx] += args.noise_level * output_std * draws[noisy_idx]

    noisy = clean.copy()
    noisy.loc[:, OUTPUT_COLUMNS] = noisy_outputs
    noisy.to_csv(args.out_csv, index=False)
    np.savez(
        args.split_path,
        **split,
        nT_regions=args.nT_regions,
        nC_regions=args.nC_regions,
        split_seed=42,
        clean_output_std=output_std,
    )
    print(
        f"Regions={args.nT_regions} x {args.nC_regions}; "
        f"train={len(split['train_idx'])}, validation={len(split['val_idx'])}, "
        f"test={len(split['test_idx'])}; "
        f"centers in training={len(split['center_idx'])}; "
        f"noise={args.noise_level:.1%}, noise seed={args.seed}; test stays clean."
    )


if __name__ == "__main__":
    main()
