# 2D CSTR PINN setup

This folder extends the supplied two-input CSTR code with a soft-constrained
PINN. Inputs are `T` and `Cao`; outputs are `Ca`, `Cb`, and `Cc`.

The PINN minimizes

`MSE + mu_rxn * mean(r_rxn^2) + mu_mb * mean(r_mb^2)`

using the exact temperature-dependent nonlinear reaction residual and the
combined mass-balance residual in physical units.

## Smoke test

From this directory:

```bash
python src/main.py \
  --job train \
  --model PINN \
  --model_id PINN2D_smoke \
  --mu_rxn 0.05 \
  --mu_mb 0.01 \
  --epochs 10 \
  --run 0

python src/main.py \
  --job experiment \
  --eval_split val \
  --model PINN \
  --model_id PINN2D_smoke \
  --mu_rxn 0.05 \
  --mu_mb 0.01 \
  --run 0
```

## Small Pareto check

```bash
python scripts/mu_sweep_2d.py \
  --mu_rxn_values 0 0.01 0.05 \
  --mu_mb_values 0 0.01 0.05 \
  --repeats 1 \
  --epochs 10
```

## Coarse validation sweep

```bash
python scripts/mu_sweep_2d.py --repeats 3 --epochs 1000
```

The sweep creates:

- `pinn_2d_mu_sweep_raw.csv`
- `pinn_2d_mu_sweep_summary.csv`
- `pinn_2d_pareto_validation.png`

Only validation metrics are used for weight selection. After choosing one
`(mu_rxn, mu_mb)` pair, evaluate that fixed pair on the test split and use it
for the final 50-repeat NN/PINN/PL-KKT-hPINN comparison.

## Final comparison after selecting the weights

Replace the example weights below with the pair chosen from the validation
Pareto front:

```bash
python scripts/compare_models_2d.py \
  --mu_rxn 0.05 \
  --mu_mb 0.01 \
  --repeats 50 \
  --epochs 1000
```

This creates `nn_pinn_kkt_2d_compare_raw.csv` and
`nn_pinn_kkt_2d_compare_summary.csv` using the untouched test split.
