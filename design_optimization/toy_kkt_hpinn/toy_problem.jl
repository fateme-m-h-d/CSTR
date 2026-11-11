using JuMP
using MathOptAI
using Ipopt
using LinearAlgebra

function solve_toy(; reduced::Bool)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "tol", 1e-10)

    @variable(model, 0 <= x[1:1] <= 1, start = 0.5)

    # Ordinary NN: Linear(1,1) -> tanh -> Linear(1,2).
    # Example weights stand in for trained weights.
    backbone = MathOptAI.Pipeline(
        MathOptAI.AbstractPredictor[
            MathOptAI.Affine(reshape([2.0], 1, 1), [0.0]),
            MathOptAI.Tanh(),
            MathOptAI.Affine(
                reshape([0.3, -0.1], 2, 1),
                [0.2, 0.4],
            ),
        ],
    )

    yraw, _ = MathOptAI.add_predictor(
        model, backbone, x;
        reduced_space = reduced,
    )

    # Constraint: -x + y₁ + y₂ = 1.
    A = reshape([-1.0], 1, 1)
    B = [1.0 1.0]
    b = [1.0]

    # Compute the fixed projection using linear solves.
    BBt = B * B'
    Astar = -B' * (BBt \ A)
    Bstar = Matrix{Float64}(I, 2, 2) - B' * (BBt \ B)
    bstar = B' * (BBt \ b)

    # Projection takes [x; yraw] as its input.
    projection = MathOptAI.Affine(
        hcat(Astar, Bstar), bstar,
    )

    y, _ = MathOptAI.add_predictor(
        model, projection, vcat(x, yraw);
        reduced_space = reduced,
    )

    # A target with a known, unique optimum at x = 0.4.
    target = 0.4 + 0.5 * 0.4 + 0.2 * tanh(2 * 0.4)
    @objective(model, Min, (y[1] - target)^2)

    optimize!(model)
    @assert is_solved_and_feasible(model)

    xv = value(x[1])
    yv = value.(y)

    # Compare optimization outputs with a direct forward pass.
    raw_direct = [
        0.3 * tanh(2 * xv) + 0.2,
        -0.1 * tanh(2 * xv) + 0.4,
    ]
    projected_direct =
        Astar * [xv] + Bstar * raw_direct + bstar

    residual = abs(sum(yv) - (1 + xv))

    @assert norm(yv - projected_direct, Inf) < 1e-7
    @assert residual < 1e-7
    @assert abs(xv - 0.4) < 1e-5

    println(reduced ? "Reduced-space" : "Full-space")
    println("  status = ", termination_status(model))
    println("  x = ", xv)
    println("  y = ", yv)
    println("  objective = ", objective_value(model))
    println("  constraint residual = ", residual)
    println("  variables = ", num_variables(model))
    println(
        "  constraints excluding variable bounds = ",
        num_constraints(model; count_variable_in_set_constraints = false),
    )

    return xv, yv
end

full = solve_toy(reduced = false)
reduced = solve_toy(reduced = true)

@assert abs(full[1] - reduced[1]) < 1e-5
@assert norm(full[2] - reduced[2], Inf) < 1e-5