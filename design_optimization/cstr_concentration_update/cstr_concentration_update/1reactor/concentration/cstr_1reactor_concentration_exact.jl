"""
Single-CSTR exact optimization in concentration variables.

This is equivalent to the original one-reactor molar-flow formulation, but
component molar-flow variables are replaced by concentrations.

Notation:
    C0[i] : fixed fresh-feed concentration [mol/L]
    CR[i] : recycle concentration [mol/L]
    C1[i] : reactor outlet concentration [mol/L]

Because the downstream unit is a simple splitter (not a separator), the
recycle and product have the same composition as the reactor outlet:
    CR[i] = C1[i].

The exact reaction model is retained:
    rate[A] = -k*C1[A]*C1[B]
    rate[B] = -rate[A]
"""

using JuMP
using Gurobi

function build_cstr_1reactor_concentration_exact(optimizer)

    I = [:A, :B]

    k = 2.0
    QF0 = 1.0
    C0 = Dict(:A => 0.99, :B => 0.01)

    model = Model(optimizer)

    # ============================================================
    # Variables
    # ============================================================

    @variable(model, 0 <= Q1 <= 10)            # reactor outlet flow [L/s]
    @variable(model, 0 <= QR <= 10)            # recycle flow [L/s]
    @variable(model, 0 <= QP <= 10)            # product flow [L/s]

    @variable(model, 0 <= C1[I] <= 1)          # reactor outlet concentrations [mol/L]
    @variable(model, 0 <= CR[I] <= 1)          # recycle concentrations [mol/L]

    @variable(model, -10 <= rate[I] <= 10)     # species rates [mol/(L*s)]

    @variable(model, 0 <= V <= 10)             # reactor volume [L]
    @variable(model, 0 <= c <= 10)             # active volume used in objective [L]

    # ============================================================
    # Volumetric-flow balances
    # ============================================================

    # Fresh feed + recycle enter the reactor.
    @constraint(model, QF0 + QR - Q1 == 0)

    # Reactor outlet is split into product + recycle.
    @constraint(model, Q1 - QP - QR == 0)

    # ============================================================
    # Splitter composition
    # ============================================================

    # Simple splitter: recycle has the same composition as reactor outlet.
    @constraint(model, [i in I], CR[i] - C1[i] == 0)

    # Product has the same composition as reactor outlet, so the original
    # P[B]/QP = 0.95 specification becomes C1[B] = 0.95.
    @constraint(model, C1[:B] == 0.95)

    # ============================================================
    # Component balances in concentration form
    # ============================================================

    # Original balance:
    #   F0[i] + R[i] - F1[i] + rate[i]*V = 0
    # with
    #   F0[i] = QF0*C0[i]
    #   R[i]  = QR*CR[i]
    #   F1[i] = Q1*C1[i]
    @constraint(
        model,
        [i in I],
        QF0 * C0[i] + QR * CR[i] - Q1 * C1[i] + rate[i] * V == 0
    )

    # ============================================================
    # Exact CSTR kinetics
    # ============================================================

    # Equivalent to the original
    # rate[A]*Q1^2 + k*F[A]*F[B] = 0,
    # after using F[i] = Q1*C1[i].
    @constraint(model, rate[:A] + k * C1[:A] * C1[:B] == 0)
    @constraint(model, rate[:B] + rate[:A] == 0)

    @constraint(model, c - V == 0)

    # @constraint(model, QR == 0.0)

    # ============================================================
    # Objective
    # ============================================================

    @objective(model, Min, c)

    return model
end


model = build_cstr_1reactor_concentration_exact(Gurobi.Optimizer)

set_optimizer_attribute(model, "NonConvex", 2)
set_optimizer_attribute(model, "TimeLimit", 120.0)

optimize!(model)

println("\n========== ONE-REACTOR CONCENTRATION EXACT ==========")
println("Status = ", termination_status(model))

if has_values(model)
    println("Objective V = ", objective_value(model))
    println("Q1 = ", value(model[:Q1]))
    println("QP = ", value(model[:QP]))
    println("QR = ", value(model[:QR]))
    println("C1[A] = ", value(model[:C1][:A]))
    println("C1[B] = ", value(model[:C1][:B]))
    println("CR[A] = ", value(model[:CR][:A]))
    println("CR[B] = ", value(model[:CR][:B]))
    println("rate[A] = ", value(model[:rate][:A]))
    println("rate[B] = ", value(model[:rate][:B]))

    rA = value(model[:rate][:A])
    rB = value(model[:rate][:B])
    Vv = value(model[:V])

    println("rate[A]*V = ", rA * Vv)
    println("rate[B]*V = ", rB * Vv)
    println("source sum = ", (rA + rB) * Vv)
end
