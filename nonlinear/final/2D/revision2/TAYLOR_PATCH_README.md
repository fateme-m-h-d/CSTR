# 2D axiswise Taylor partition patch for PL-KKT-hPINN

Copy:

- `2D/src/adaptive_partition.py` (new)
- `2D/src/linearization.py` (replace)
- `2D/scripts/linearization_accuracy.py` (new, optional diagnostic)

The existing model and utility code already load arbitrary monotone `T_edges`
and `C_edges`, so they do not need changes.

## Run one adaptive case

From the repository's `2D/` directory:

```bash
python -m src.generate_data --n_total_points 170 --seed 0 --out_csv data.csv
python -m src.linearization \
  --partition taylor_axiswise \
  --nT_regions 5 \
  --nC_regions 3 \
  --reference_T_points 181 \
  --reference_C_points 81 \
  --safety_factor 1.10
python -m scripts.linearization_accuracy
```

Then train/evaluate with the existing commands.

## What the method equalizes

For each axis it estimates

`mu_j = max_over_other_input ||H_f||_2 * ||dz*/dx_j||_2^2`

and uses the sampled bound

`R_i_axis = safety_factor * max(mu_j in segment i) * h_i^2 / 8`.

The edges are found by a root/shooting solve so all axiswise bounds are nearly
equal. This is a tensor-grid heuristic, not a certified full-cell 2D bound.

## Fair comparison

For every `(nT_regions, nC_regions)`, compare `uniform` and
`taylor_axiswise` using the same `data.csv`, seeds, architecture, epochs, and
repeats. Report:

- maximum/mean sampled linearization error,
- original nonlinear constraint violation,
- RMSE,
- preprocessing and inference time.
