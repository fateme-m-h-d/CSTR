"""
Single-CSTR optimization in concentration variables with an embedded ReLU NN.

NN mapping:
    (C1[A], C1[B], V) -> xi

where xi >= 0 is the reaction-source magnitude [mol/s]:
    xi = -rate_A*V = rate_B*V.

The same xi is used with opposite signs in the A and B balances, so the
A -> B stoichiometry is preserved exactly.

Compared with the exact concentration model, this surrogate model removes:
    rate[A], rate[B]
    rate[A] = -k*C1[A]*C1[B]
    rate[B] = -rate[A]

and replaces both reaction source terms with one common NN output xi.
"""

using JuMP
using Gurobi
import PythonCall
import MathOptAI

function build_cstr_1reactor_concentration_nn(optimizer)

    I = [:A, :B]

    QF0 = 1.0
    C0 = Dict(:A => 0.99, :B => 0.01)

    model = Model(optimizer)

    # ============================================================
    # Variables
    # ============================================================

    @variable(model, 0 <= Q1 <= 10)
    @variable(model, 0 <= QR <= 10)
    @variable(model, 0 <= QP <= 10)

    @variable(model, 0 <= C1[I] <= 1)
    @variable(model, 0 <= CR[I] <= 1)

    @variable(model, 0 <= V <= 10)
    @variable(model, 0 <= c <= 10)

    # Reaction source magnitude used in both species balances.
    @variable(model, 0 <= xi_NN <= 10)

    # ============================================================
    # Flow and splitter equations
    # ============================================================

    @constraint(model, QF0 + QR - Q1 == 0)
    @constraint(model, Q1 - QP - QR == 0)

    # Simple splitter: recycle concentration equals reactor outlet concentration.
    @constraint(model, [i in I], CR[i] - C1[i] == 0)

    # Product concentration equals reactor outlet concentration.
    @constraint(model, C1[:B] == 0.95)

    # ============================================================
    # Embedded concentration NN
    # ============================================================

    predictor = MathOptAI.PytorchModel(
        joinpath(@__DIR__, "concentration_extent_nn.pt")
    )

    @variable(model, 0 <= V_scaled <= 1)
    @constraint(model, V_scaled == V / 10.0)

    nn_output, _ = MathOptAI.add_predictor(
        model,
        predictor,
        [C1[:A], C1[:B], V_scaled];
        config = Dict(
            :ReLU => MathOptAI.ReLUSOS1,
        ),
    )

    # Training used xi_scaled = xi/10.
    @constraint(model, xi_NN == 10.0 * only(nn_output))

    # ============================================================
    # Component balances
    # ============================================================

    # A is consumed: source term = -xi_NN.
    @constraint(
        model,
        QF0 * C0[:A] + QR * CR[:A] - Q1 * C1[:A] - xi_NN == 0
    )

    # B is produced: source term = +xi_NN.
    @constraint(
        model,
        QF0 * C0[:B] + QR * CR[:B] - Q1 * C1[:B] + xi_NN == 0
    )

    @constraint(model, c - V == 0)

    # @constraint(model, QR == 0.0)

    # ============================================================
    # Objective
    # ============================================================

    @objective(model, Min, c)

    return model
end


model = build_cstr_1reactor_concentration_nn(Gurobi.Optimizer)

# Q* C terms remain bilinear in the concentration formulation.
set_optimizer_attribute(model, "NonConvex", 2)
set_optimizer_attribute(model, "TimeLimit", 120.0)

optimize!(model)

println("\n========== ONE-REACTOR CONCENTRATION NN ==========")
println("Status = ", termination_status(model))

if has_values(model)
    Vv = value(model[:V])
    CA = value(model[:C1][:A])
    CB = value(model[:C1][:B])
    xi = value(model[:xi_NN])

    println("Objective V = ", objective_value(model))
    println("Q1 = ", value(model[:Q1]))
    println("QP = ", value(model[:QP]))
    println("QR = ", value(model[:QR]))
    println("C1[A] = ", CA)
    println("C1[B] = ", CB)
    println("xi_NN = ", xi)

    # Audit against original exact kinetics.
    xi_exact = 2.0 * CA * CB * Vv

    println("\n--- ORIGINAL-PHYSICS AUDIT ---")
    println("NN xi = ", xi)
    println("Exact xi at same point = ", xi_exact)
    println("Surrogate error = ", xi - xi_exact)
end
