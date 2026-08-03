# Isothermal CSTR PINN Study

This project compares a neural network, a physics-informed neural network (PINN), and a KKT-hPINN for an isothermal CSTR with two constraints. The input is `Cao`; the outputs are `Ca`, `Cb`, and `Cc`.

## Generate Data

```bash
python3 -m src.generate_data --n_total_points 150 --seed 0
python3 -m src.linearization --nC_regions 30
```

## PINN Weight Sweep

```bash
python3 -m scripts.mu_sweep --mode rxn_only
python3 -m scripts.mu_sweep --mode mb_only
python3 -m scripts.mu_sweep --mode scale_both
```

## Compare Methods

```bash
python3 -m scripts.compare_models
```
