# Symbolic reduced-space update — 18 November 2026

This archive replaces the old `reduced` callback implementation. The old timings
labelled reduced must not be used for the symbolic reduced-space comparison.

- `full`: unchanged explicit MLP and projection-stage equations.
- `reduced`: MathOptAI `reduced_space=true` on the backbone, then native symbolic
  arithmetic for the saved projection. No custom nonlinear operators or Python
  value/gradient/Hessian callbacks are registered for this branch.
- `gray`: unchanged MathOptAI gray-box wrapper around `complete.pt`.

Python is still required for importing checkpoints, full-space initialization,
and independent parity reports. It is not used for reduced solver evaluations.
No retraining is required. All saved weights, projection depths, scaling,
constraint equations, and operating objectives are retained.

## Run these checks first

From the directory containing `julia`, `python`, and `artifacts`:

```bash
julia --project=julia julia/check_symbolic.jl mass
julia --project=julia julia/check_embedding.jl mass reduced
julia --project=julia julia/run.jl mass reduced

julia --project=julia julia/check_symbolic.jl pl
julia --project=julia julia/check_embedding.jl pl reduced
julia --project=julia julia/run.jl pl reduced
```

For iterative projections, first check numerical algebra independently of
symbolic expression expansion:

```bash
julia --project=julia julia/check_symbolic.jl enforce numeric
julia --project=julia julia/check_symbolic.jl hardnet numeric
julia --project=julia julia/check_symbolic.jl enforce
julia --project=julia julia/check_symbolic.jl hardnet
```

Then run the same embedding parity and optimization commands if construction
succeeds. `check_symbolic` checks that reduced construction adds no variables or
linking constraints. `check_embedding` checks output parity at five saved inputs.
Only report solve results after those checks pass.

## Important feasibility limit

Fully substituting repeated nonlinear corrections duplicates expressions when
JuMP/MOI expands expression trees. The implementation counts expanded nodes and
stops above 2,000,000 (configurable with `CSTR_SYMBOLIC_MAX_NODES`). In particular,
40-stage ENFORCE/HardNet may exceed this limit early. This is an unsupported-size
result for this direct symbolic implementation, not a failed trained model.
There is no silent callback fallback, no early stopping of projection iterations,
and no shortened checkpoint. Increasing the limit can require excessive memory.
Do not reduce saved iteration counts just to obtain a solver result: that changes
the forward predictor. Handling large iterative models efficiently may require
an expression-DAG-capable backend or additional formulation work.

The small linear systems are expanded using fixed-order Gaussian elimination
on positive-definite Gram/regularized normal matrices. Unlike PyTorch's pivoted
solve, its floating-point behavior can differ near singularity. Numerical parity
checks must pass; this does not add regularization to the ENFORCE or PL systems.

An independent NumPy translation matched all 173 saved parity points for mass,
PL, and ENFORCE to about 2e-15. HardNet differed by up to 6.3e-5, exceeding the
1e-6 parity tolerance: its algebraically equivalent linear-system evaluation is
numerically sensitive. Treat HardNet symbolic mode as unvalidated; do not loosen
the parity tolerance just to accept it. See NUMERICAL_ALGEBRA_CHECK.txt.

Julia was unavailable in the editing environment. Julia syntax was checked when
possible, but these Julia scripts must be executed in your configured server
environment before treating this update as validated.

---

The earlier README below records the training/model design. Any earlier claims
that reduced uses callbacks are superseded by the update above.

# CSTR training with projection and Julia embedding

This folder trains through each projection and embeds the resulting corrected
predictor in the operating optimization. The main reproducibility check compares
the Julia outputs with the exact Python forward pass at fixed operating inputs.

The implementation is one portable PyTorch version. The uploaded native-package
projects were inspected to identify their equations, split, scaling, and projection
settings. This folder contains explicit variants of their projection algorithms,
with fixed settings that can be reproduced in JuMP. The uploaded package models
are separate experiments with different kinetics; their saved weights are not
substituted into this study.

## Start here

From this folder, activate your Python environment and run:

```bash
python -m pip install -r requirements.txt
python python/train.py --method all
python python/verify.py

export JULIA_PYTHONCALL_EXE="$(command -v python)"
export JULIA_CONDAPKG_BACKEND=Null
julia julia/setup.jl

julia --project=julia julia/exact.jl
julia --project=julia julia/check_embedding.jl mass full
julia --project=julia julia/run.jl mass full
julia --project=julia julia/run.jl mass reduced
julia --project=julia julia/run.jl mass gray
julia --project=julia julia/run.jl mass bilevel
```

Use the same Python environment for training and PythonCall. Run the export
commands in the shell before starting Julia. The setup script creates the Julia
project and resolves package versions by package name.

For the other three methods, replace mass with pl, hardnet, or enforce:

```bash
julia --project=julia julia/check_embedding.jl enforce full
julia --project=julia julia/run.jl enforce full
julia --project=julia julia/run.jl enforce reduced
julia --project=julia julia/run.jl enforce gray
```

The same commands work for baseline. Bilevel is an additional formulation of the
trained mass model. Its leader replays the saved affine/tanh layers using the
BilevelJuMP legacy nonlinear interface and its follower solves the mass-only QP.

Training results in artifacts can be used immediately if included in the archive.
They are new models trained by this folder. Running train.py again replaces the
selected method's artifacts using the shared split.json. A separate output root
can be supplied with --output; pass that root as the third Julia argument when
using it. The smoke-test checkpoints used during development are excluded.

## What full, reduced, and gray mean here

The comparison applies to the **complete trained MLP plus projection**.

| CLI mode | Optimization representation |
| --- | --- |
| full | MathOptAI expands the MLP and the custom adapter adds every projection solve and update as explicit variables and equations. |
| reduced | Only T and Cao are variables. Three scalar nonlinear expressions evaluate the complete corrected predictor. PyTorch supplies exact automatic derivatives through the MLP and all projection updates. |
| gray | MathOptAI wraps complete.pt, including the projection, as its vector nonlinear oracle. The model has T, Cao, and three corrected-output variables. |

The adapter in julia/common.jl is a MathOptAI.AbstractPredictor for the complete
saved predictor. The usual add_predictor flags choose the formulation:

    y, form = MathOptAI.add_predictor(model, predictor, [T,Cao])
    y, form = MathOptAI.add_predictor(model, predictor, [T,Cao]; reduced_space=true)
    y, form = MathOptAI.add_predictor(model, predictor, [T,Cao]; gray_box=true)

MathOptAI's built-in PyTorch parser recognizes standard MLP layers. The projection
requires this custom extension. In reduced space the extension uses JuMP scalar
nonlinear operators with derivative callbacks instead of expanding a massive nested
expression for every matrix solve. There are no projection or MLP stage variables.
This is a complete reduced-space formulation with opaque differentiable functions;
its derivative evaluation cost should be reported alongside solve time. Gray space
uses MathOptAI's complete-model vector oracle, retaining output variables.

All three formulations use the SAME trained artifact and finite projection
algorithm. The driver prints the variable count so the scope can be verified.
Bilevel is an additional formulation of the mass-trained affine model; it expands
the leader MLP and represents the same correction as a follower QP.

The complete PyTorch model supplies the reference forward pass and owns the trained
backbone and projection configuration. Full space loads backbone.pt through
MathOptAI and rebuilds the correction stages from config.json. Reduced and gray
space evaluate complete.pt. A fixed-input check precedes optimization, because
similarity between equations is insufficient to establish forward equivalence.

## Which model is trained

Inputs are temperature and A feed concentration. Outputs are the three corrected
outlet concentrations. Separate backbones are trained for:

| Training name | Projection in training and inference | Julia embedding |
| --- | --- | --- |
| baseline | Identity | MathOptAI backbone |
| mass | Orthogonal Euclidean projection onto Ca+Cb+Cc=Cao+2 | Explicit linear projection or equivalent BilevelJuMP follower QP |
| pl | Orthogonal projection onto a sigmoid-weighted blend of regional kinetic Taylor rows, plus the exact summed balance | The same weights and affine rows in one NLP |
| hardnet | Fixed-depth regularized Gauss-Newton iterations on the nonlinear distance-projection KKT residual | The same initial raw output, zero initial multipliers, and every update expanded as equations |
| enforce | Fixed-depth tangent-plane orthogonal projections of the current iterate | The same Jacobians and tangent projections expanded as equations |

The HardNet implementation uses physical coordinates and analytic derivatives.
Its regularized normal equation is
(K'K + regularization I) delta = -K' F,
with 0.5 step length and a fixed number of updates. Here F contains stationarity
anchored to the original raw prediction plus the two nonlinear/linear balances.
The paper/package also supports transformed variables, inequality handling,
adaptive stopping, and other solver options. This equality-only version must be
reported as a physical-coordinate, fixed-depth HardNet-style variant.

The ENFORCE implementation uses the same tangent-plane projection formula as
AdaNP. It uses a fixed 100-epoch unconstrained warmup, followed by projected training
and a displacement-loss coefficient of 0.1. Its fixed depth replaces the package's
adaptive depth and automatic activation logic. It should be reported as a
fixed-depth ENFORCE-style variant.

The PL implementation is a smooth-gating adaptation of the joint Taylor variant.
There are 11 temperature intervals and 3 feed intervals. Sigmoid width is one
eighth of the interval width. It projects onto the **blended constraints**; it
does not blend separately projected outputs. This preserves the exact summed
balance, while individual regional kinetic rows and the original nonlinear kinetic
equation can have residuals. Training and embedding both use this same smooth
variant. It is separate from a hard-indicator PL-KKT-hPINN and from a learned-lifting
implementation. No solve-per-region loop is used.

The mass case is the affine KKT-hPINN training case. BilevelJuMP is its optimization
representation. The lower-level objective is half the squared physical Euclidean
distance from the raw output, with one affine equality and no lower-level bounds.
Upper-level nonnegativity matches the explicit mass embedding. General nonlinear
kinetics are outside this affine QP.

## Training protocol

The default is 1000 epochs, two 32-unit tanh layers, float64, Adam at 1e-4, batches
of 16, seed 42, and a saved 60/20/20 split. MaxAbs scales are fitted to the complete
dataset to match the original supplied protocol. Every method starts from the
same random backbone initialization. Its final-layer bias starts at the mean
scaled training target; its final-layer weights are reduced by a factor 100.

Mass and PL use the projected output in every training epoch. HardNet and ENFORCE
use 100 warmup epochs and projected loss afterward. Validation always evaluates
the complete projected model. Best checkpoint selection considers projected
validation MSE after warmup. The data loss is MSE in scaled output coordinates.
Projection distance is physical Euclidean distance, the same metric used in Julia.
ENFORCE additionally minimizes normalized displacement between raw and projected
outputs. Gradients flow through the projection linear solves into the backbone.

All methods share the data and split. Projection-specific warmup, regularization,
and displacement penalties are recorded in config.json and should be disclosed
when comparing training times. Warmup is part of training time. Timing of compilation,
loading, and forward passes should be reported separately from solver time.

The number of projection steps is saved during training. Julia reads it from the
artifact. Changing the depth, damping, gates, or metric changes the deployed
predictor; retrain/export and rerun parity checks for that experiment. Additional
CLI arguments are available with python python/train.py --help.

## Case equations and the uploaded-package mismatch

The supplied data/cstr.csv comes from the original optimization.zip and contains
170 rows. Its input locations exactly match the uploaded 2D package datasets.
Those package scripts use reverse activation energy 80000 J/mol, while the
optimization example and this folder use 100000 J/mol. The maximum difference
between their concentration labels is about 1.0009 mol/L. The original package
checkpoints therefore belong to a different kinetic model.

The retained reference model is the one used in the original Julia optimization:

gA = Cao - Ca - 10 kf(T) Ca Cb^2 + 10 kr(T) Cc = 0
gSum = Cao + 2 - Ca - Cb - Cc = 0
gB = 2 - Cb - 2(Cao - Ca) = 0

kf(T) = 1e13 exp(-90000/(8.314 T))
kr(T) = 1e11 exp(-100000/(8.314 T))

The exact benchmark and generated targets use all three equations. The
projections target gSum alone for mass and gA plus gSum for the nonlinear cases.
The B residual is always reported. These source equations imply Cc=3(Cao-Ca).
If Cc represents the molar concentration for A+2B reversible to C, this convention
requires revision before treating the model as stoichiometrically faithful.
The current benchmark is kept consistent with the original data and study.

The operating bounds are 280<=T<=460 and 0.8<=Cao<=1.2. Nonnegativity is imposed
on corrected outputs in optimization, with no intermediate clamping. The shared
interior-test objective is

J = 0.7 Cao - Ca - 0.5 ((T-370)/90)^2.

The exact reference is approximately T=391.41 K, Cao=1.0111 mol/L, J=0.53578.

## Checks and interpreting results

verify.py checks complete-forward derivatives against finite differences, gradient
flow to backbone weights, serialization, and the projection callbacks' first and
second derivatives. Nonlinear projections can switch roots or become ill-conditioned;
a derivative failure near such a point needs investigation.

check_embedding.jl fixes operating inputs, solves the embedded algebra, and compares
the concentrations against Python. It allows negative outputs during this check
so forward equivalence can be tested even outside the optimization-feasible subset.
run.jl uses corrected-output nonnegativity, first solves at fixed inputs, then
releases them with warm starts.

The final report includes the original-model objective at selected inputs, all
original residuals, and a fresh Python forward-match check. The nonlinear methods
must pass an absolute physical residual threshold of 1e-6 as well as forward-match
and solver checks. Fixed-depth projection does not itself guarantee that tolerance
everywhere. The test point may fail because the saved forward predictor is negative,
because its iterations have not converged sufficiently, or because Ipopt failed.
Keep the raw exit status when diagnosing this.

Parity is a forward-equivalence test. It does not establish global optimality.
A KKT root in a nonlinear projection may be a nonminimal stationary point.
The unrolled method preserves the saved algorithm's root selection and finite
depth; an implicit KKT replacement can select a different root. This folder keeps
the actual saved recurrence for HardNet.

## Files

python/cstr_models.py defines physics, the shared MLP, and differentiable projections.
python/train.py trains and exports all methods.
python/bridge.py loads exports, supplies exact stage starts, and differentiates the complete forward map for reduced-space callbacks.
python/verify.py checks derivatives and gradient flow.
julia/common.jl defines the complete-model MathOptAI adapter and its three embeddings.
julia/bilevel.jl defines the mass-only affine follower.
julia/check_embedding.jl compares Python and Julia at fixed inputs.
julia/run.jl performs operating optimization.
julia/exact.jl solves the original three-equation reference.
julia/setup.jl creates the Julia environment.

Every method exports complete.pt, backbone.pt, config.json, layers.json,
history.csv, metrics.json, and parity.csv. artifacts/split.json records row indices
and the input CSV hash.

## Sources

MathOptAI PyTorch embedding definitions and custom layers
https://lanl-ansi.github.io/MathOptAI.jl/stable/manual/PyTorch/

BilevelJuMP lower-level restrictions
https://joaquimg.github.io/BilevelJuMP.jl/dev/tutorials/non_linear/

KKT-hPINN paper and code
https://arxiv.org/abs/2402.07251
https://github.com/li-group/KKThPINN

KKT-HardNet paper and code
https://arxiv.org/abs/2507.08124
https://github.com/SOULS-TAMU/kkt-hardnet

ENFORCE paper and code
https://arxiv.org/abs/2502.06774
https://github.com/process-intelligence-research/ENFORCE

PL-KKT-hPINN
https://www.sciencedirect.com/science/article/pii/S0098135426004266
