Complete KKT-HardNet gray-box patch

Copy these files into your existing cstr_trained_projection folder:
  ADD python/legacy_complete_gray.py
  ADD julia/legacy_complete_gray.jl
  REPLACE julia/run_legacy_kkt.jl

Activate the same Python environment used for legacy-package training and PythonCall.
Run from the project root:
  julia --project=julia julia/run_legacy_kkt.jl artifacts_legacy_new/hardnet gray
Use your actual trained artifact directory if different.

No retraining. Full and reduced branches remain unchanged.
Gray now wraps the complete trained MLP plus original package projection.
Outer variables: T, Cao and three concentrations (5 total).
The package solves its projection internally to the saved tolerance.
Jacobian and weighted Hessian are supplied through MathOptAI's vector gray-box oracle.
Derivatives use implicit differentiation of the local KKT system, including
backbone scaling and physical output scaling; no differentiation of Newton iterations.
No multipliers or projection KKT constraints are added to the outer optimization.
Values use the original package initialization, tolerance and weights.

Verification: Python wrapper values matched the public package predictor exactly
at the reference input; Jacobian and Hessian evaluations returned finite values.
Julia integration and full optimization have not yet been run for this patch.
Derivatives assume a nonsingular KKT system and a locally unchanged solution branch.
