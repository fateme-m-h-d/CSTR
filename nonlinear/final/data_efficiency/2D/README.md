# 2D Non-Isothermal CSTR Data-Efficiency Study

This project studies training-data efficiency for a fixed 21-region, two-constraint KKT-hPINN. The inputs are temperature `T` and feed concentration `Cao`; the outputs are `Ca`, `Cb`, and `Cc`.

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
