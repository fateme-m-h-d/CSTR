# 1D Isothermal CSTR Data-Efficiency Study

This project studies the effect of training-data size for a fixed 30-region, two-constraint KKT-hPINN. The input is `Cao`; the outputs are `Ca`, `Cb`, and `Cc`.

## Run

From this directory:

```bash
python3 -m scripts.run_all
python3 -m scripts.confidence_interval
```

The default scenarios use `n_inner_per_region = [0, 1, 2, 5, 10, 15, 20, 25]`, 50 repetitions per model, float64 training, and hard step-function region masks.

## Outputs

- `metric_summary_by_samples.csv`
- `rmse_vs_samples_1d_data_efficiency.png`

Temporary models, learning curves, per-run tables, logs, scenario archives, and duplicate scenario files are removed after aggregation.
