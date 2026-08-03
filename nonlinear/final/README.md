# PL-KKT-hPINN

This repository contains the implementation and experiment scripts for **PL-KKT-hPINN**, a piecewise-linear KKT-based hard-constrained physics-informed neural network framework for surrogate modeling of constrained chemical process systems.

The code is provided to support the reproducibility of the main results reported in the associated paper.

The experiments focus on CSTR case studies and evaluate model accuracy, constraint violation, and data efficiency.

## Repository Structure

```text
PL-KKT-hPINN/
│
├── 1D/
│   └── One-dimensional isothermal CSTR experiments
│
├── 2D/
│   └── Two-dimensional CSTR experiments
│
├── PINN/
│   └── Soft-constrained PINN baseline experiments
│
├── data_efficiency/
│   └── Data-efficiency experiments
│
└── README.md

```

## Citation

If you use this repository, please cite the [paper](https://arxiv.org/abs/2606.10682).



