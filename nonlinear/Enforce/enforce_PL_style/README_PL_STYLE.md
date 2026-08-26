# ENFORCE — PL-KKT-hPINN-style CSTR benchmark

This clean folder revises the earlier ENFORCE CSTR test so the experimental
protocol is aligned as closely as practical with the **1D PL-KKT-hPINN** code.

## Fixed comparison settings

- Dataset: 150 rows from `data.csv`
- Split: **90 train / 30 validation / 30 test**
- Split permutation: fixed RandomState seed 42
- Scaling: **MaxAbs scaling fitted on all 150 rows before the split**
- Network: 1 input → 32 → 32 → 3 outputs
  - ENFORCE uses `hidden_layers=1` because its implementation already creates the first input→hidden layer; one additional hidden layer gives **two actual hidden layers**, matching PL-KKT-hPINN.
- Epochs: 1000
- Batch size: 16
- Learning rate: 1e-4
- Precision: float64
- Hardware: CPU
- ENFORCE projection tolerance: 1e-6
- ENFORCE max projection iterations: 100
- ENFORCE displacement weight: 0.5 (method default)

The split is fixed across repetitions. Run 1 uses model seed 42, run 2 uses 43,
etc.; only model/training randomness changes.

## Important interpretation

The three methods do **not** have identical projection algorithms, so setting
`tolerance=1e-6` and `max iterations=100` gives the same numerical requirements,
not an identical stopping norm. ENFORCE uses its own AdaNP stopping rule.

The 30 validation rows are reserved exactly as in PL-KKT-hPINN, but ENFORCE's
native `Trainer.fit()` API trains on the 90 training tensors only and does not use a
validation set. For the cross-method comparison, the checkpoint is the **final epoch**
model. The 30 validation rows and 30 test rows are never used for ENFORCE gradient
updates.

For the agreed final protocol, training is shuffled, while validation/test ordering is
deterministic. ENFORCE has no native validation loader, so validation shuffling is not
applicable; its test set is fixed and evaluated as the same 16 + 14 rows every run.
PL-KKT-hPINN should likewise be rerun with `train shuffle=True`, `val shuffle=False`,
and `test shuffle=False`.

## Timing definitions

### Training time

`experiment2_pl_style.py` starts `time.perf_counter()` immediately before the
full training subprocess and stops when that subprocess exits. Thus Python
startup/imports, data preparation, model construction, 1000 epochs, and
checkpoint writing are included, matching the timing boundary used by the
PL-KKT-hPINN experiment wrapper.

### Evaluation time

`run_enforce_cstr.py --job experiment` loads/prepares the data **before** the
timer. The timer starts before model construction/checkpoint loading and ends
after the 30 test predictions, metric calculations, and report/CSV writing.
The test set is processed in batches of 16 (16 + 14), like PL-KKT-hPINN.

Do **not** use ENFORCE's old built-in `Inference time` number for the final
cross-method timing table; use the `Evaluation time` produced here.

ENFORCE evaluation is intentionally **not** wrapped in `torch.no_grad()` because
AdaNP needs autograd to compute the constraint Jacobian at inference.

## Run

Activate the environment containing ENFORCE:

```bash
source ~/CSTR/myenv/bin/activate
cd /path/to/Enforce_PL_Style
```

One complete training + test run:

```bash
python experiment2_pl_style.py --runs 1
```

Fifty independent training repetitions using the same data split:

```bash
python experiment2_pl_style.py --runs 50
```

Or run jobs separately:

```bash
python run_enforce_cstr.py --job train --run 1
python run_enforce_cstr.py --job experiment --run 1
```

## Output

- `models/enforce_run_<n>.pt` — trained network checkpoint
- `results/pl_split_indices.csv` — exact source-row membership of train/val/test
- `results/enforce_run_<n>_test_predictions.csv` — physical predictions/residuals
- `results/enforce_run_<n>_metrics.json` — test metrics
- `results/enforce_run_<n>_report.txt` — human-readable report
- `results/enforce_pl_style_benchmark.csv` — per-run training/evaluation times + metrics

The scripts report both:

1. **PL-style scaled RMSE** and **PL-style original nonlinear violation**, using
   PL's unweighted averaging over test batches; and
2. physical-unit RMSE plus global mean/max residuals for `g1` and `g2`.
