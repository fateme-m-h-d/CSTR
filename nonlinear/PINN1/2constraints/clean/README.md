# Isothermal CSTR PINN Study

This project compares a neural network, a physics-informed neural network (PINN), and a KKT-hPINN for an isothermal CSTR with two constraints. The input is `Cao`; the outputs are `Ca`, `Cb`, and `Cc`.

The clean implementation preserves the original nonlinear equations, data sampling, float64 training, network architecture, optimizer, shuffled split with seed 42, best-validation checkpointing, PINN loss, and evaluation metrics.

## Generate Data

```bash
python3 -m src.generate_data --n_total_points 150 --seed 0
python3 -m src.linearization --nC_regions 30
```

## Compare Methods

```bash
python3 -m scripts.compare_models
```

This runs the NN, PINN, and KKT-hPINN comparison. Use `--repeats`, `--epochs`, `--mu_rxn`, and `--mu_mb` to override the published defaults.

## PINN Weight Sweep

```bash
python3 -m scripts.mu_sweep --mode rxn_only
python3 -m scripts.mu_sweep --mode mb_only
python3 -m scripts.mu_sweep --mode scale_both
```

The sweep maintains one combined raw table and one combined summary table across modes.
