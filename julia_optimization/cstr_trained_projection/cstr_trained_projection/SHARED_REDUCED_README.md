# Shared nonlinear subexpressions for reduced ENFORCE

This adds a separate `reduced_shared` mode. It retains two decision variables,
uses no external value/derivative operators, and preserves all 40 saved ENFORCE
steps. No retraining or changes to Python/artifacts/environment files are needed.

## Files to copy into an existing cstr_trained_projection folder

Replace:
- julia/common.jl
- julia/run.jl

Add:
- julia/shared_reduced.jl
- julia/check_shared.jl
- julia/run_shared.jl

Retain the existing julia/symbolic_reduced.jl. The original full/reduced/gray and
bilevel implementations are unchanged. README files are optional documentation.

## Run

From the project root:

```bash
julia --project=julia julia/check_shared.jl
julia --project=julia julia/check_embedding.jl enforce reduced_shared
julia --project=julia julia/run.jl enforce reduced_shared
```

`check_shared.jl` is standalone and requires only JuMP, MathOptAI, Ipopt, JSON3,
and standard libraries. It tests 1 and 40 stages at every saved CSV parity point,
plus objective gradient and Hessian centered-difference checks at the reference
input. It adds three output nonnegativity constraints, but no linking constraints.

`run_shared.jl` is an optional standalone optimization driver without Python:

```bash
julia --project=julia julia/run_shared.jl
```

Use the normal `run.jl` command for the independent Python and original-model
report. That command still imports Python for reporting and other modes, not for
shared-mode solver evaluation.

## What changes mathematically?

Nothing in the saved forward model. ENFORCE uses y_next = y - J'*(J*J')^(-1)*g.
The 2-by-2 system is represented using elimination arithmetic, as in the previous
symbolic implementation. No extra regularization or damping is introduced.

What changes is expression storage. Legacy nonlinear subexpressions hold MLP
operations, residuals, Jacobian entries, solve quantities and corrected stages.
They are expression references, not optimization variables. JuMP/MOI performs
AD on these expressions rather than calling custom Python derivative callbacks.

The backbone is reconstructed from layers.json using MathOptAI native
Affine/Tanh/etc predictors with reduced_space=true. Its expressions are then
translated into stored legacy subexpressions, preserving shared references.
This is an explicit subexpression-storage strategy, not automatic detection of
all algebraically identical subexpressions. The exported weights and scaling
must remain paired with the same checkpoint.

The mode is ENFORCE-only: HardNet's forward-parity issue is not resolved here.
Shared storage addresses duplication; it does not guarantee fast derivatives,
good conditioning, convergence, or feasibility of omitted reactor equations.
The formulation uses the same inherited case equations as the existing project.

## Validation performed in this update

Executed with Julia 1.10.10 and locally installed JuMP/MathOptAI/Ipopt:
- One-stage projection: all 173 CSV points matched within 6.56e-15.
- Full saved 40-stage projection: all 173 CSV points matched within 3.50e-15.
- Objective gradient and Hessian matched centered differences at (391.41,1.0111).
- Two decision variables; no MLP/projection linking constraints.
- 2,901 stored nonlinear subexpressions for the complete 40-stage model.
  This is stored graph size, not directly comparable to the old expanded-tree
  node counter (different counting units).
- Standalone free-input optimization: LOCALLY_SOLVED in 9 iterations,
  T=393.76886133747803, CA0=0.9109003758567394, objective=0.5112377764505208.
- The standalone free-input Ipopt time here was 0.023 s; this is not a controlled
  cross-machine comparison against earlier results on your server.

The Python-reporting run.jl path still needs execution in your configured
PythonCall environment; no Python/PyTorch was installed in this test runtime.
See SHARED_VALIDATION.txt for the actual Julia check and solver output.
