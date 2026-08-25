# Offline adaptive PL patch for `2D/`

This patch replaces the fixed Cartesian `nT_regions x nC_regions` linearization
with an offline, data-driven binary hyper-rectangle partition.

It uses only:

- the available rows in `data.csv`; and
- the known analytical constraints used by PL-KKT-hPINN.

It does **not** call the CSTR simulator, train a neural network, or use neural
network predictions when building the partition.

## Files

Copy the patch files over the same paths in the repository:

- `2D/src/linearization.py`
- `2D/src/models.py`
- `2D/src/utils.py`
- `2D/scripts/experiment2.py`
- `2D/scripts/run_adaptive.py` (new)

## Main command

Run from the repository's `2D/` directory:

```bash
python3 -m src.linearization \
  --data_csv data.csv \
  --epsilon 0.01 \
  --criterion projection \
  --error_quantile 1.0 \
  --max_regions 64 \
  --max_depth 12 \
  --min_samples_leaf 8 \
  --split_mode quantile \
  --split_quantiles 0.25,0.5,0.75
```

Then train/evaluate with the existing workflow through the new driver:

```bash
python3 -m scripts.run_adaptive \
  --epsilon 0.01 \
  --criterion projection \
  --error_quantile 1.0 \
  --max_regions 64 \
  --min_samples_leaf 8 \
  --repeats 50 \
  --epochs 1000
```

For the repository's synthetic demonstration, append `--generate_data`. For a
real dataset, do not use that flag; place your available observations in
`data.csv` with columns:

```text
Temperature (T), Cao, Ca, Cb, Cc
```

## Error criteria

### `projection` (recommended for the current paper question)

For each observed point in a candidate region:

1. build the local Taylor affine constraint;
2. apply the same closed-form affine KKT projection used by `NNOPT`;
3. evaluate the original known nonlinear constraint at the projected point.

The regional score is the maximum (or selected quantile) nonlinear residual.
This directly estimates the residual caused by using that local PL projection
on the available data.

### `remainder`

Compute the Taylor discrepancy on the observed points:

```text
|c(z_i) - ell_r(z_i)|
```

This is useful when the measured rows do not satisfy the known constraint
exactly, because it separates nonlinear approximation error from the raw data
constraint residual.

### `hybrid`

Use the larger of the `projection` and `remainder` scores.

## Adaptive splitting

The algorithm starts with one input box. If its score is larger than `epsilon`,
it tests candidate cuts in every input dimension. With the default quantile
mode, it tests the 25th, 50th, and 75th percentiles of the data in the region.
For each candidate cut, it constructs two child Taylor models and scores the
candidate by:

```text
max(left_child_error, right_child_error)
```

The cut with the smallest value is accepted. The currently worst leaf is split
first. Refinement stops when the tolerance is met or a data/resource stopping
condition is reached.

## Generated preprocessing artifacts

- `region_bounds.npz`: arbitrary adaptive box bounds used by the hard masks
- `ABb_matrices.csv`: local affine constraints in the repository's format
- `adaptive_regions.csv`: bounds, anchor points, data counts, errors, stop reasons
- `lin_params.csv`: detailed Taylor coefficients
- `partition_summary.json`: final region count and tolerance status

## Important interpretation

This is an **empirical offline estimate** based on the available observations.
It is not an unconditional guarantee for arbitrary future neural-network
outputs. The estimate is defensible under the condition that future inputs and
projected outputs remain in the data-supported regime represented by the rows
used during preprocessing.

For a clean test protocol, build the partition using only the training or a
separate preprocessing subset, not the held-out test outputs.
