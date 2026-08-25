# 2D Isothermal CSTR With Two Constraints

This project reproduces the CSTR experiments using temperature and inlet concentration as inputs. Each local projection enforces the linearized reaction balance and the exact total mass balance.

## Run

From this directory:

```bash
python3 -m scripts.run_all
python3 -m scripts.confidence_interval
```

The default study uses 170 fixed data points, 3 concentration regions, temperature segment counts `[1, 2, 3, 5, 7, 9, 11]`, and 50 repetitions per model and scenario.

## Outputs

- `metric_summary_by_segments.csv`
- `time_vs_regions_2d.png`
- `rmse_vs_regions_2d.png`
- `nonlinear_violation_vs_regions_2d.png`
