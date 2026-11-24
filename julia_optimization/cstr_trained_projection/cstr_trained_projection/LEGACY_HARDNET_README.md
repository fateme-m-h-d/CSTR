# Older-package training → PyTorch backbone → direct projection KKT

This workflow uses the older **KKT-HardNet 1.0.1 API**, pinned to commit
`bc9ba8908e650659436f531724386508e6cf30e0` of
https://github.com/SOULS-TAMU/kkt-hardnet (package subdirectory `kkthn`).
Training is performed by the package. There is no custom training loop or
reimplementation of its projection solver.

The model passed to MathOptAI is a `.pt` copy of the trained **backbone only**.
The nonlinear projection is added separately as direct JuMP KKT equations.
No Newton/Gauss–Newton iterations are expanded in the outer optimization.

## 1. Install the older API in a separate environment

Run these commands from this folder (Python 3.12 was tested):

```bash
python -m venv .venv-legacy-hardnet
source .venv-legacy-hardnet/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements_legacy_hardnet.txt
```

The package pins its historical JAX/NumPy/Optax dependencies. The requirements
also pin an Orbax version compatible with that stack. Do not upgrade to the
current main branch: it has a different API and numerical implementation.

## 2. Train using the package and export `.pt`

```bash
python python/train_legacy_hardnet.py --output artifacts_legacy_new/hardnet
```

A tested checkpoint is already included in `artifacts_legacy/hardnet`. The command above trains a new one; use that new path in Julia to embed it. Use another new directory when rerunning:

```bash
python python/train_legacy_hardnet.py --output artifacts_legacy_run2/hardnet --seed 43
```

The defaults match the attached older benchmark's main training choices:
1,000 epochs, first 100 raw-network warm-up epochs, two 32-neuron ReLU hidden
layers, batch size 16, Adam learning rate 1e-4, consistency coefficient 0.5,
float64, final-epoch checkpoint. The data split uses RandomState(42): 102
training / 34 validation / 34 test points. The training/validation CSV is
rearranged to invert the package's internal split. Test rows never enter the
training CSV. Full-data MaxAbs scaling is retained from the attached workflow.

The package's projected-phase loss is scaled-output supervised MSE plus
`0.5 * mean((raw_scaled - projected_scaled)^2)`. The backbone activation is
ReLU because that is what this historical package implements; exporting as
`tanh` would change the trained function.

Physical parameters are fixed to the EXISTING optimization case:

- T in [280,460], Cao in [0.8,1.2], tau=10, CB0=2, CC0=0.
- kf=1e13 exp(-90000/(8.314 T)).
- kr=1e11 exp(-100000/(8.314 T)).

The script verifies the data against the same reference equations and refuses
to train if the labels belong to the former Ear=80000 case. It preserves the
inherited A and summed-concentration equations; the extra B balance remains a
separate diagnostic, as in the existing optimization scripts.

Projection changes relative to the attached benchmark: the stopping tolerance
is tightened from 1e-6 to **1e-8**, and the iteration CAP is increased from 100
to **1000**. This is a tolerance-stopped solve, not a fixed-1000-step model.
These settings are exposed as `--projection-tolerance` and
`--projection-max-iterations`. The included successful run needed at most 42
iterations at the 173 checked inputs. The package stops on the **L2 norm of
the full KKT residual**, not only physical constraint feasibility.

The old package differentiates using a custom VJP/regularized implicit KKT
adjoint (`backward_reg=1e-8`), not by backpropagating through all forward
iterations. Its old backward implementation does NOT reject unconverged solves
automatically. This wrapper therefore audits the final trained projection on
all data inputs plus three off-grid inputs and rejects the export for use in
embedding if any checked projection misses tolerance or becomes nonfinite.
That audit is not a claim that every intermediate training solve or every
possible point in the continuous domain converges.

### Files produced

Under the selected output directory (`artifacts_legacy_new/hardnet/` above):

- `backbone.pt`: plain float64 `torch.nn.Sequential` saved with
  `torch.save(model, ...)`, as required by `MathOptAI.PytorchModel`.
- `backbone_state_dict.pt`: optional weights-only copy; this is NOT the file
  passed to MathOptAI.
- `layers.json`: the same weights in readable form for initialization/checks.
- `config.json`: physics, input/output scaling, package revision, training and
  projection settings, and a relative pointer to the package bundle.
- `parity.csv`: physical inputs, raw physical predictions and corrected outputs.
- `projection_diagnostics.csv`: final multipliers, iteration counts, the
  package-reported KKT L2 residual and an independent KKT residual check.
- `export_checks.json`: whether all checked projection solves passed.
- `check_inputs.csv`, `split.json`, package training CSVs, and the package's
  complete timestamped model bundle with `model_weights.npz` and metadata.

The JAX affine matrices have shape `(in,out)`. Export copies their TRANSPOSE
into PyTorch's `(out,in)` weights and copies each bias, inserting the same ReLU
activations. There is no PyTorch retraining and no numerical projection hidden
inside `backbone.pt`. Its inputs and raw outputs are SCALED coordinates.

Version 1.0.1 leaves its public prediction cache unset immediately after
training. The wrapper reloads the saved bundle through `KKTHardNet().load(...)`
before export verification; it does not modify the package algorithm.

To re-export a trained bundle WITHOUT retraining:

```bash
python python/export_legacy_hardnet.py artifacts_legacy/hardnet
python python/check_legacy_export.py artifacts_legacy/hardnet
```

Keep the entire timestamped package bundle when copying the checkpoint.

## 3. Embed the backbone and add the projection's KKT equations

Use your existing Julia project setup. Set the Python environment BEFORE
starting Julia:

```bash
export JULIA_PYTHONCALL_EXE="$(command -v python)"
export JULIA_CONDAPKG_BACKEND=Null
export CSTR_LEGACY_PYTHON="$(command -v python)"

julia --project=julia julia/check_legacy_kkt.jl artifacts_legacy/hardnet full
julia --project=julia julia/run_legacy_kkt.jl artifacts_legacy/hardnet full
```

The builder uses:

```julia
predictor = MathOptAI.PytorchModel(joinpath(directory, "backbone.pt"))
raw_scaled, formulation = MathOptAI.add_predictor(
    model, predictor, [T / input_scale[1], Cao / input_scale[2]],
)
```

It then adds THREE scaled projection outputs `w` and TWO equality multipliers
`lambda`. Physical concentrations are `y[i] = output_scale[i] * w[i]`.
Let `D = Diagonal(output_scale)` and `h` be the raw scaled MLP output. The exact
projection problem being represented is:

    minimize_w  0.5 * ||w - h||^2
    subject to  g(x, D*w) = 0.

The five direct KKT equations are:

    g(x, D*w) = 0
    w - h + D * J_g(x, D*w)' * lambda = 0.

The equivalent physical-coordinate objective is
`0.5 * ||D^(-1)*(y - raw_physical)||^2`; its stationarity uses `D^(-2)`.
Using unweighted physical distance would give a DIFFERENT projection.

No inner nonnegativity constraints are introduced because the original package
model declares only the two equalities. The free-input outer model retains
nonnegative concentrations, separately from the inner projection KKT system.
The full output derivative information comes from JuMP's KKT equations and the
MathOptAI backbone embedding. There are no Python projection derivative
callbacks in the Julia optimization model.

### Optional backbone embedding experiments

```bash
julia --project=julia julia/run_legacy_kkt.jl artifacts_legacy/hardnet reduced
julia --project=julia julia/run_legacy_kkt.jl artifacts_legacy/hardnet gray
```

**Observed limitation:** the supplied reduced- and gray-backbone runs reached a KKT-feasible
point with approximately 0.476 output discrepancy from the package and was
correctly REJECTED by the forward check. It must not be reported as a successful
optimization of the trained predictor. The default full-space run passed. Thus
convergence of the package's projection does not, by itself, make a direct KKT
outer model branch-equivalent. A branch-selection condition or a different
projection formulation would require additional analysis; none is silently
added here.

These choices apply to the BACKBONE only. The projection remains five explicit
KKT equations with five variables for every option. In particular, `reduced`
here is a hybrid (reduced MLP + explicit KKT projection), not a two-variable
reduced embedding of the complete implicitly defined predictor.

The old package's ReLU backbone is nonsmooth. MathOptAI's standard ReLU
formulation uses `max(0,x)`; Ipopt is being used locally on this piecewise
smooth model. Changing the activation after training is not an acceptable fix.

## 4. Validation and interpretation

`check_legacy_kkt.jl` fixes each saved input, warm-starts with its package output
and checks the direct KKT solution against that forward prediction. The runner
also checks package prediction at the FREE optimization solution, since a
nonlinear KKT system may have multiple roots. It rejects a forward discrepancy
above 1e-6 rather than silently treating a different KKT root as the trained
predictor. Root matching at checked points is evidence, not a uniqueness or
global-projection-minimum proof.

The runner first solves with two inputs held fixed; the SECOND Ipopt solve is
the free-input optimization time to report. `direct_kkt_<mode>.json` records the
selected inputs, corrected concentrations, objective and forward discrepancy.

The included checkpoint's test physical RMSE is approximately 0.355. Its
full-space optimized extra B-balance residual is approximately -0.465; that
balance is not enforced by this projection. Consequently, passing KKT and
export checks is not a claim of accurate predictions or satisfaction of the
entire original physical model. Hyperparameter tuning/accuracy assessment is
separate from verifying this training-to-embedding workflow.

This is a new package-trained model for the Ear=100000 case. It is not the
previous custom fixed-iteration HardNet checkpoint, and its results should be
labeled separately in comparisons. See `LEGACY_HARDNET_VALIDATION.txt` for the
actual checks completed in the supplied environment.

## Updating your existing folder selectively

Add these NEW files (the previous numerical source files are unchanged):

- `python/legacy_hardnet.py`
- `python/train_legacy_hardnet.py`
- `python/export_legacy_hardnet.py`
- `python/check_legacy_export.py`
- `python/legacy_forward.py`
- `julia/legacy_hardnet_kkt.jl`
- `julia/check_legacy_kkt.jl`
- `julia/run_legacy_kkt.jl`
- `requirements_legacy_hardnet.txt`
- `LEGACY_HARDNET_README.md`, `LEGACY_HARDNET_VALIDATION.txt`

Only the root `README.md` receives a pointer to this workflow. Copy the entire
new `artifacts_legacy/` directory if using the included tested checkpoint.
