# Single CSTR projection embeddings

## Run the comparison

Use the Julia environment from your existing full-range MathOptAI project. It
needs JuMP, Ipopt, MathOptAI and PythonCall with PyTorch available to PythonCall.
The bilevel run also needs BilevelJuMP in that environment. The checkpoints
are the supplied full-range PyTorch Sequential models.

```sh
julia run.jl exact
julia run.jl baseline tanh
julia run.jl bilevel tanh 4 3
julia run.jl pl tanh 4 3
julia run.jl hardnet tanh 4 3 6
julia run.jl enforce tanh 4 3 6
julia run.jl pl-smooth tanh 4 3
```

Arguments are method, activation, number of temperature regions, number of feed
regions, and number of projection steps. Region counts apply to the two
regional methods and the sigmoid experiment. The iteration count applies to the
two iterative methods. Run the commands from this folder with your usual active
Julia project. If you need to add dependencies, run
`import Pkg; Pkg.add(["JuMP","Ipopt","MathOptAI","PythonCall","BilevelJuMP"])`
in the intended Julia environment before running the scripts.

Start with the supplied tanh checkpoint and Ipopt across all methods. Tanh is a
smooth activation. The plain JuMP routes also accept softplus, sigmoid and ReLU
checkpoints. ReLU with Ipopt is a nonsmooth comparison. Sigmoid gates in
`pl-smooth` affect region selection independently of the NN activation.

## Changes from julia_optimization

- `common.jl` keeps the full-range input scaling, output unscaling and the
  three concentration outputs. The JuMP variants use MathOptAI's full-space
  embedding of the frozen PyTorch network.
- `run.jl` shares the operating bounds and the interior-test objective
  `0.7 * CA0 - CA - 0.5 * ((T - 370)/90)^2`. The exact reference should give
  about 391.41 K, 1.0111 mol/L and 0.53578 with the supplied equations.
- `methods/` inserts the corrections between the raw output and the outer
  objective. Nonnegativity applies to the concentration vector used by that
  objective. The projection distance uses concentrations in physical units.
- The original code's two projection targets are the summed concentration
  balance and the A kinetic balance. The exact benchmark additionally uses the
  B species balance, which generated the training targets. The output reports
  its residual for every case. The supplied equations imply C formation equal
  to three times A depletion on the reference branch. Changing that convention
  requires rebuilding the data and checkpoints as well as the benchmark.

## How the four projection methods differ

| Command | Correction embedded in the optimization |
|---|---|
| bilevel | The leader evaluates the NN and maximizes the operating objective. A separate follower minimizes half the squared distance from the raw outputs subject to two affine equalities for a fixed region. BilevelJuMP generates the follower KKT conditions. |
| hardnet | A fixed number of dampable Newton steps solve the KKT system of the nonlinear equality projection. Every stationarity equation is anchored to the original NN output. The implementation uses analytic derivatives in physical coordinates and unrolls the steps into JuMP. |
| pl | Each region uses a joint first-order Taylor approximation of the kinetic balance. JuMP evaluates the explicit affine orthogonal projector. The summed balance stays exact. |
| enforce | Each step linearizes the original residual at the current iterate, then projects that iterate onto the resulting tangent plane. The implementation unrolls a fixed number of ENFORCE/AdaNP steps. |

BilevelJuMP supports the affine follower QP. The original kinetic equality
contains a cubic term in the concentrations, with Arrhenius coefficients
depending on the leader's temperature, so it falls outside the supported
lower-level formulation. A mass-only QP is possible. The supplied experiment
uses the same two affine rows as PL, which lets you compare automatic KKT
reformulation against explicit elimination region by region. The leader uses
the very same tanh checkpoint via a compact affine/tanh embedding in legacy
JuMP nonlinear syntax. Only this route reads individual layer parameters
directly. Lower-level nonnegativity is deliberately absent, to match the
equality-only projection; the leader requires the corrected output to be
nonnegative in both variants.

The regional comparison solves one continuous problem per cell. At shared
cell boundaries, either adjacent map is allowed, and the reported candidate
is the best accepted regional solve. If you need a unique prediction at ties,
define a region ownership rule and check it in the reported boundary solutions.
The sigmoid experiment solves a single smooth NLP. It blends the affine
**kinetic equations** using separable sigmoid weights, then projects onto
that blended equation and the original summed balance. Those are the rows
it enforces. Individual cell kinetic equations may have nonzero residuals.

The HardNet-style route implements the Newton-KKT idea. It uses the original
physical variables, equality constraints, and direct Newton updates, without
the paper's log-exponential lifting, regularized Gauss-Newton implementation,
or JAX training. Its terminal KKT check establishes numerical stationarity
and feasibility within the set tolerance. It does not certify the nearest
point globally. ENFORCE uses fixed depth for the smooth JuMP embedding. Increase
the step count for either iterative route if its terminal residual is too large.
These comparisons use frozen backbones with projection at solve time, without
end-to-end retraining. The PL route demonstrates the joint Taylor variant,
without learned lifted variables or adaptive partition generation.

## Read the results

Each run prints the raw and corrected outlets, solver status, surrogate
objective, both nonlinear-model projection residuals, and the extra B balance
residual. The method-specific check is the affine residual for bilevel and PL,
the blended affine residual for smooth PL, the terminal nonlinear residual
for ENFORCE, and the terminal KKT residual for HardNet. The selector excludes
failed local solves, materially negative corrected outlets and method residuals
above 1e-6. PL and bilevel can pass with a nonzero original kinetic residual.

`original_objective` is recomputed at the candidate's T and CA0 by solving
the original three-equation reference branch. Compare it against the exact
case's objective. Compare bilevel and PL in the same cells with the same tanh
checkpoint. Their affine projection is mathematically identical for any fixed
input; different local operating optima or failures can still occur.

## Method references

- BilevelJuMP nonlinear models: https://joaquimg.github.io/BilevelJuMP.jl/dev/tutorials/non_linear/
- KKT-hPINN: https://www.sciencedirect.com/science/article/pii/S0098135424001820
- KKT-HardNet: https://arxiv.org/abs/2507.08124
- PL-KKT-hPINN: https://www.sciencedirect.com/science/article/pii/S0098135426004266
- ENFORCE: https://arxiv.org/abs/2502.06774
