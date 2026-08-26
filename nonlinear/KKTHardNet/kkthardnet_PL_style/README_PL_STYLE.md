# KKT-HardNet — PL-KKT-hPINN-style CSTR benchmark

This clean folder revises the earlier KKT-HardNet CSTR test so the experimental
protocol is aligned as closely as practical with the **1D PL-KKT-hPINN** code.

## Fixed comparison settings

- Dataset: 150 rows from `data.csv`
- Split: **90 train / 30 validation / 30 test**
- Split permutation: fixed RandomState seed 42
- Scaling: **MaxAbs scaling fitted on all 150 rows before the split**
- Network: 1 input → 32 → 32 → 3 outputs
- Epochs: 1000
- Batch size: 16
- Learning rate: 1e-4
- Precision: float64
- Hardware/backend: CPU/JAX
- KKT projection tolerance: 1e-6
- KKT max Gauss-Newton projection iterations: 100

KKT-HardNet-specific settings without a direct PL-KKT-hPINN analogue are kept
from the previous working script: `epoch_mlp=100` and `cons_alpha=0.5`.

The split is fixed across repetitions. Run 1 uses model seed 42, run 2 uses 43,
etc.; only model/training randomness changes.

## How the exact PL train/validation/test rows are preserved

KKT-HardNet normally performs its own train/validation split using NumPy
`default_rng`, which is not the same permutation routine used in PL-KKT-hPINN.
The revised script therefore:

1. creates PL's exact 90 training, 30 validation, and 30 test row indices;
2. keeps all 30 test rows completely outside KKT-HardNet's training CSVs;
3. places the 90 training + 30 validation rows into a 120-row package CSV in an
   arrangement that inverts KKT-HardNet's internal `default_rng` split; and
4. uses `train_frac=0.75`, causing the package to recover exactly the intended
   90 training and 30 validation rows.

`kkthn_pl_inputs/run_<n>/kkt_internal_row_mapping.csv` records that mapping.

Because KKT-HardNet receives scaled variables, its symbolic constraints multiply
the scaled `Cao, Ca, Cb, Cc` back by their MaxAbs scales before evaluating the
original physical CSTR equations.

## Important interpretation

The same `1e-6` tolerance and `100` maximum iterations are given to ENFORCE and
KKT-HardNet, but their projection algorithms/stopping definitions are not
identical. This is a matched numerical requirement, not an assertion that the
algorithms use the same residual norm internally.

KKT-HardNet reports validation metrics during training and saves the **final epoch**
parameters. The agreed cross-method protocol is to use the final epoch model for all
three methods; PL-KKT-hPINN should therefore be rerun with its checkpointing changed
to save/use the final epoch.

For the agreed final protocol, training minibatches remain shuffled, while validation
and test ordering are deterministic. KKT-HardNet internally applies a fixed validation
permutation and does not expose a `shuffle=False` switch; this wrapper pre-inverts that
permutation so the **effective validation batches** are PL's unshuffled first 16 and
remaining 14 validation samples. Test prediction is likewise fixed at 16 + 14.

For test prediction this benchmark explicitly uses:

```python
projection_backend="jax"
```

so the result is not mixed with KKT-HardNet's optional compiled native-C
projection backend.

## Timing definitions

### Training time

`experiment2_pl_style.py` times the complete training subprocess externally with
`time.perf_counter()`, matching PL-KKT-hPINN's experiment wrapper. Python
startup/imports, data preparation, model construction, 1000 epochs, and package
artifact saving are therefore included.

### Evaluation time

The experiment script prepares data before the timer. The timer starts before
`KKTHardNet().load(...)` and stops after the 30 test predictions, metrics, and
report/CSV writing. Test prediction is performed in batches of 16 (16 + 14), as
in PL-KKT-hPINN.

Do **not** use KKT-HardNet's package-reported estimated JAX single/batch
microbenchmark times for the final cross-method table; use the `Evaluation time`
produced by this script.

## Run

Activate the dedicated KKT-HardNet environment:

```bash
source ~/CSTR/kkt_env/bin/activate
cd /path/to/KKTHardNet_PL_Style
```

One complete training + test run:

```bash
python experiment2_pl_style.py --runs 1
```

Fifty repetitions using the same fixed data split:

```bash
python experiment2_pl_style.py --runs 50
```

Or run jobs separately:

```bash
python run_kkthardnet_cstr.py --job train --run 1
python run_kkthardnet_cstr.py --job experiment --run 1
```

## Output

- KKT-HardNet's normal timestamped run folder
- `models/kkt_run_<n>_metadata_path.txt` — pointer used to reload the trained run
- `results/pl_split_indices.csv` — exact PL row membership
- `kkthn_pl_inputs/run_<n>/kkt_internal_row_mapping.csv` — proof of exact package split mapping
- `results/kkt_run_<n>_test_predictions.csv` — physical test predictions/residuals
- `results/kkt_run_<n>_metrics.json` — test metrics
- `results/kkt_run_<n>_report.txt` — human-readable report
- `results/kkt_pl_style_benchmark.csv` — per-run training/evaluation times + metrics

The scripts report both PL-style scaled metrics and physical-unit RMSE plus
mean/max residuals for `g1` and `g2`.
