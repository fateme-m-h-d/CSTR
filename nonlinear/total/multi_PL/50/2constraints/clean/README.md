# 2D Non-Isothermal CSTR Data-Efficiency Study

This project studies training-data efficiency for a fixed 21-region, two-constraint KKT-hPINN. The inputs are temperature `T` and feed concentration `Cao`; the outputs are `Ca`, `Cb`, and `Cc`.

The numerical model, sampling order, nonlinear solves, linearizations, data split, architecture, optimizer, and float32 training settings follow the original experiment. The region indicator is intentionally changed from a smooth sigmoid to the same hard step-function convention used by the 1D implementation. In two dimensions, the active mask is the product of the temperature and concentration step masks.

## Run

From this directory:

```bash
python3 -m scripts.run_all
python3 -m scripts.confidence_interval
```

The default study uses `n_inner_per_region = [0, 1, 2, 5, 10, 15, 20, 25]` and 50 repetitions per model.

## Outputs

- `metric_summary_by_samples.csv`
- `rmse_vs_samples_2d.png`

Temporary datasets, matrices, models, learning curves, and per-run tables are removed after aggregation.
