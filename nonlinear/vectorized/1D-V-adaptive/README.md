# Adaptive 1D PL-KKT-hPINN

Place this folder at `CSTR/nonlinear/vectorized/adaptive_1D`. It uses the uploaded
1D-V code and its original 150-point data.csv: input Cao, outputs [Ca, Cb, Cc],
temperature 350 K, and Cao domain [0.5, 1.5]. The supplied artifacts contain
90 adaptive intervals, matching the 90 uniform segments in the original archive.

## Changes

- Added src/adaptive_partition.py: the one-input reduction of the 2D sampled
  Hessian/Taylor partition indicator.
- Replaced src/linearization.py: creates adaptive edges and solves the physical
  steady state at each new interval center to form the linearized constraint.
- Updated scripts/run_all.py to reuse data.csv and call adaptive partitioning.
- models.py, utils.py, main.py, train.py, config.py, generate_data.py, diagnostics,
  and scripts/experiment2.py are unchanged copies of the uploaded 1D-V files.
  The vectorized bucketize calls already support sorted nonuniform interval edges.
  Validation and test DataLoaders retain shuffle=False; training retains shuffling.

## Generate intervals and run

Run these commands from inside adaptive_1D. Generate 90 intervals, or change 90
to any desired positive integer:

```bash
python -m src.linearization --n_regions 90
```

The old flag --nC_regions is an alias. Optional offline sampling controls are:

```bash
python -m src.linearization --n_regions 90 --reference_C_points 1025 --safety_factor 1.10
```

Outputs: region_edges.npz (C_edges), ABb_matrices.csv, lin_params.csv, and
region_partition_summary.csv. data.csv is never modified by this generator.

Test one training/evaluation repetition, then run 50:

```bash
SCENARIO_ID=adaptive_1d_90 NUM_ITERATIONS=1 python -m scripts.experiment2
SCENARIO_ID=adaptive_1d_90 NUM_ITERATIONS=50 python -m scripts.experiment2
```

Each repetition trains and evaluates both NN and KKThPINN. The uploaded runner
uses 1000 epochs, batch size 16, Adam with learning rate 1e-4, two hidden layers
of width 32, and explicitly passes --dtype 64. These existing settings are retained.
In particular, this uploaded 1D-V runner uses float64, not float32. Its EPOCHS
setting comes from src/main.py; it does not read an EPOCHS environment variable.
Prediction-only timing is retained. Repeated invocations replace the summary
CSVs and use the original run-indexed checkpoint paths. Use separate folder
copies when retaining results for different interval counts.

The optional inherited sweep is `python -m scripts.run_all`. It runs every count
in config.py, uses the fixed original dataset, and retains the existing summary
and cleanup behavior. For one 90-region comparison, use experiment2 instead.

## One-input methodology

Let z(Cao) = [Cao, Ca(Cao), Cb(Cao), Cc(Cao)] and

    f = Cao - Ca - tau*kf*Ca*Cb^2 + tau*kr*Cc.

For interval Q with full width h, compute on the physical reference curve:

    q_CC = (dz/dCao)^T H_f (dz/dCao)
    M_CC = maximum sampled |q_CC| inside Q
    indicator(Q) = safety_factor * M_CC * h^2 / 8

With coordinate order [Cao, Ca, Cb, Cc], the nonzero Hessian entries are
H[Ca,Cb] = H[Cb,Ca] = -2*tau*kf*Cb and H[Cb,Cb] = -2*tau*kf*Ca.
Finite differences estimate dz/dCao, matching the supplied 2D method. Bisect the
interval with the largest indicator at its midpoint until the target count is
reached. In 1D there is only one direction to split. Sort the final intervals so
the edge intervals and matrix groups have matching region IDs. The mass-balance
constraint is already linear and retains its exact coefficients.

This is a sampled curvature indicator, not a certified Taylor bound for all
points or NN predictions. The code uses the supplied CSTR equations to solve
1025 reference points by default and each interval center. These points are used
only offline and do not enlarge the training, validation, or test datasets.

## Validation and inherited evaluation behavior

The offline generator was executed and its intervals, coefficient ordering,
center residuals, analytic Hessian, and Python syntax were checked. The original
vectorized model and utility files are copied byte for byte. PyTorch is unavailable
in the preparation environment, so first run the one-repetition command on your
server before starting 50 repetitions.

The original train.py averages batch means equally. For 30 test samples and batch
size 16, the 16-sample and 14-sample batches get equal weight. This existing
behavior is retained to match 1D-V. Apply sample weighting consistently in both
uniform and adaptive versions before reporting a sample-weighted mean or RMSE.
