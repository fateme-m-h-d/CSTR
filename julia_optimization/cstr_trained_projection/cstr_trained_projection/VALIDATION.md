# Validation record

All five architectures completed 1000 training epochs using the saved common split.
Validation checkpoint selection used the projected output after any warmup.
These are trial checkpoints and include the metrics below.

| Method | Test scaled RMSE | Max original A residual on test data | Minimum test concentration |
| --- | ---: | ---: | ---: |
| baseline | 0.120116 | 44.3433 | -0.338206 |
| mass | 0.119263 | 35.7272 | -0.339711 |
| pl | 0.0811986 | 75.2552 | -0.194557 |
| hardnet | 0.121735 | 10.5586 | -0.28559 |
| enforce | 0.066798 | 3.55271e-15 | -0.469624 |

Python checks passed for gradient flow through the projection, serialized forward evaluation,
finite-difference comparisons of complete-model and projection derivatives, Hessian symmetry,
and torch.vmap compatibility at the recorded check inputs.

Some trial predictions are negative. The HardNet fixed-depth checkpoint also misses the
physical residual tolerance at some inputs. The optimization driver imposes nonnegativity
and rejects candidates failing physical residual and forward-equivalence checks.

All Julia sources passed a tree-sitter syntax parse. Julia was unavailable in the development
environment, so package loading, MathOptAI custom-adapter compatibility, and numerical Ipopt
or BilevelJuMP runs have not been executed here. Run check_embedding.jl on your server first.
Successful Python checks do not establish successful Julia embedding.
