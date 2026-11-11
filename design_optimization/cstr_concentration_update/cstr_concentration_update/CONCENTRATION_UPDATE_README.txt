Concentration-formulation update
================================

Original files are preserved. New files added:

1-reactor concentration versions
--------------------------------
1reactor/concentration/cstr_1reactor_concentration_exact.jl
    Exact one-reactor optimization using C0, CR and C1 instead of component
    molar-flow variables F, FR, P and R.

1reactor/concentration/train_concentration_extent_nn.py
    Trains a ReLU NN with mapping
        (C1[A], C1[B], V) -> xi
    where xi = k*C1[A]*C1[B]*V is the positive A->B reaction source magnitude.
    Saves concentration_extent_nn.pt in the same directory.

1reactor/concentration/cstr_1reactor_concentration_nn.jl
    One-reactor concentration optimization with the trained NN embedded.
    Uses -xi in the A balance and +xi in the B balance.

5-reactor concentration versions
--------------------------------
cstr_concentration_exact.jl
    Exact five-reactor GDP model using outlet concentrations C[i,n] and recycle
    concentration CR instead of F, FR, P and R.

cstr_concentration_nn.jl
    Five-reactor GDP model using the same concentration_extent_nn.pt at every
    candidate reactor. The NN prediction is used only when that unit is an
    active CSTR; bypassed units use xi=0.

Files intentionally NOT overwritten
-----------------------------------
1reactor/cstr_1reactor_exact.jl
1reactor/newNN/train_extent_nn.py
1reactor/newNN/cstr_1reactor_extent_nn.jl
cstr_exact.jl
cstr_nonlinear_embedding.jl

Suggested run order
-------------------
1. julia 1reactor/concentration/cstr_1reactor_concentration_exact.jl
2. python 1reactor/concentration/train_concentration_extent_nn.py
3. julia 1reactor/concentration/cstr_1reactor_concentration_nn.jl
4. julia cstr_concentration_exact.jl
5. julia cstr_concentration_nn.jl
