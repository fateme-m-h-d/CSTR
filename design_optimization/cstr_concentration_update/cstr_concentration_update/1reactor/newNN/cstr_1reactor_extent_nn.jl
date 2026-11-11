"""
Single-CSTR optimization using a neural-network
reaction-extent surrogate.

NN mapping:

    (FAin, FBin, Qin, V) -> xi

where:

    FAout = FAin - xi
    FBout = FBin + xi

The NN replaces:
    - rate[A]
    - rate[B]
    - kinetic equation
    - rate[A]*V
    - rate[B]*V

The same xi is used in both species balances,
so A -> B stoichiometry is preserved exactly.
"""

using JuMP
using Gurobi

import PythonCall
import MathOptAI


function build_cstr_1reactor_extent_nn(
    optimizer
)

    I = [:A, :B]

    # Used only for diagnostics later.
    # It is NOT imposed as an optimization constraint.
    k = 2.0

    QF0 = 1.0

    C0 = Dict(
        :A => 0.99,
        :B => 0.01,
    )

    F0 = Dict(
        i => C0[i] * QF0
        for i in I
    )

    model = Model(
        optimizer
    )


    # ============================================================
    # Original process variables
    # ============================================================

    @variable(
        model,
        0 <= Q <= 10
    )

    @variable(
        model,
        0 <= QFR <= 10
    )

    @variable(
        model,
        0 <= F[I] <= 10
    )

    @variable(
        model,
        0 <= FR[I] <= 10
    )

    # NOTE:
    # rate[A] and rate[B] have been removed.

    @variable(
        model,
        0 <= V <= 10
    )

    @variable(
        model,
        0 <= c <= 10
    )

    @variable(
        model,
        0 <= QR <= 10
    )

    @variable(
        model,
        0 <= QP <= 10
    )

    @variable(
        model,
        0 <= R[I] <= 10
    )

    @variable(
        model,
        0 <= P[I] <= 10
    )


    # ============================================================
    # Reactor inlet variables
    # ============================================================

    @variable(
        model,
        0 <= FAin <= 10
    )

    @variable(
        model,
        0 <= FBin <= 10
    )

    @variable(
        model,
        1 <= Qin <= 10
    )


    # Fresh feed + recycle = actual reactor inlet

    @constraint(
        model,
        FAin ==
        F0[:A] + FR[:A]
    )

    @constraint(
        model,
        FBin ==
        F0[:B] + FR[:B]
    )

    @constraint(
        model,
        Qin ==
        QF0 + QFR
    )


    # ============================================================
    # Optional but useful exact invariant
    #
    # Total concentration A+B = 1 mol/L
    #
    # This keeps NN inputs on the same physical manifold
    # used to generate its training data.
    # ============================================================

    @constraint(
        model,
        FAin + FBin == Qin
    )


    # ============================================================
    # NN
    #
    # Inputs:
    #   FAin / 10
    #   FBin / 10
    #   Qin  / 10
    #   V    / 10
    #
    # Output:
    #   xi / 10
    # ============================================================

    predictor = MathOptAI.PytorchModel(
        joinpath(
            @__DIR__,
            "extent_nn.pt",
        )
    )


    @variable(
        model,
        0 <= FAin_scaled <= 1
    )

    @variable(
        model,
        0 <= FBin_scaled <= 1
    )

    @variable(
        model,
        0.1 <= Qin_scaled <= 1
    )

    @variable(
        model,
        0 <= V_scaled <= 1
    )


    @constraint(
        model,
        FAin_scaled == FAin / 10.0
    )

    @constraint(
        model,
        FBin_scaled == FBin / 10.0
    )

    @constraint(
        model,
        Qin_scaled == Qin / 10.0
    )

    @constraint(
        model,
        V_scaled == V / 10.0
    )


    nn_output, _ = MathOptAI.add_predictor(
        model,
        predictor,
        [
            FAin_scaled,
            FBin_scaled,
            Qin_scaled,
            V_scaled,
        ];
        config = Dict(
            :ReLU =>
            MathOptAI.ReLUSOS1,
        ),
    )


    # NN output in physical units [mol/s]

    @variable(
        model,
        0 <= xi_NN <= 10
    )

    @constraint(
        model,
        xi_NN ==
        10.0 * only(nn_output)
    )


    # ============================================================
    # CSTR balances using ONE common reaction extent
    # ============================================================

    # A is consumed
    @constraint(
        model,
        FAin
        - F[:A]
        - xi_NN
        == 0
    )

    # B is produced
    @constraint(
        model,
        FBin
        - F[:B]
        + xi_NN
        == 0
    )

    # Reaction does not change volumetric flow
    @constraint(
        model,
        Qin - Q == 0
    )


    # ============================================================
    # Splitter
    # ============================================================

    @constraint(
        model,
        [i in I],
        F[i]
        - P[i]
        - R[i]
        == 0
    )

    @constraint(
        model,
        Q
        - QP
        - QR
        == 0
    )

    @constraint(
        model,
        [i in I],
        P[i] * Q
        - F[i] * QP
        == 0
    )

    # Product specification
    @constraint(
        model,
        0.95 * QP
        - P[:B]
        == 0
    )


    # ============================================================
    # Reactor volume
    # ============================================================

    @constraint(
        model,
        c - V == 0
    )


    # ============================================================
    # Recycle
    # ============================================================

    @constraint(
        model,
        [i in I],
        FR[i] - R[i] == 0
    )

    @constraint(
        model,
        QFR - QR == 0
    )


    # ============================================================
    # Objective
    # ============================================================

    @objective(
        model,
        Min,
        c
    )


    return model
end


# ============================================================
# Solve
# ============================================================

model = build_cstr_1reactor_extent_nn(
    Gurobi.Optimizer
)

set_optimizer_attribute(
    model,
    "NonConvex",
    2,
)

set_optimizer_attribute(
    model,
    "TimeLimit",
    120.0,
)

optimize!(model)


# ============================================================
# Results
# ============================================================

println(
    "\n========== EXTENT-NN RESULTS =========="
)

println(
    "Status = ",
    termination_status(model),
)


if has_values(model)

    println(
        "Objective V = ",
        objective_value(model),
    )

    println(
        "\n--- Reactor inlet ---"
    )

    println(
        "FAin = ",
        value(model[:FAin]),
    )

    println(
        "FBin = ",
        value(model[:FBin]),
    )

    println(
        "Qin = ",
        value(model[:Qin]),
    )

    println(
        "V = ",
        value(model[:V]),
    )


    println(
        "\n--- NN reaction extent ---"
    )

    xi = value(
        model[:xi_NN]
    )

    println(
        "xi_NN = ",
        xi,
    )


    println(
        "\n--- Reactor outlet ---"
    )

    FAout = value(
        model[:F][:A]
    )

    FBout = value(
        model[:F][:B]
    )

    Qout = value(
        model[:Q]
    )

    println(
        "F[A] = ",
        FAout,
    )

    println(
        "F[B] = ",
        FBout,
    )

    println(
        "Q = ",
        Qout,
    )


    println(
        "\n--- Product ---"
    )

    PA = value(
        model[:P][:A]
    )

    PB = value(
        model[:P][:B]
    )

    println(
        "P[A] = ",
        PA,
    )

    println(
        "P[B] = ",
        PB,
    )

    println(
        "P[A] + P[B] = ",
        PA + PB,
    )

    println(
        "QP = ",
        value(model[:QP]),
    )

    println(
        "QR = ",
        value(model[:QR]),
    )


    # ========================================================
    # Post-optimization physical audit
    #
    # Evaluate the ORIGINAL kinetic equation at the
    # NN-selected solution.
    # ========================================================

    Vval = value(
        model[:V]
    )

    exact_xi_at_solution =
        2 *
        Vval *
        FAout *
        FBout /
        (Qout^2)

    println(
        "\n--- ORIGINAL-PHYSICS AUDIT ---"
    )

    println(
        "NN xi = ",
        xi,
    )

    println(
        "Exact kinetic xi at same point = ",
        exact_xi_at_solution,
    )

    println(
        "Surrogate error at optimum = ",
        xi
        - exact_xi_at_solution,
    )

end