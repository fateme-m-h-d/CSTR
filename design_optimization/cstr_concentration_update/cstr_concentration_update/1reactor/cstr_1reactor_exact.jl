"""
Single-CSTR version of the CSTR superstructure.

Purpose:
Establish an exact one-reactor baseline before replacing any
bilinear terms with a neural network.
"""

using JuMP
using Gurobi

function build_cstr_1reactor(optimizer)

    I = [:A, :B]

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

    model = Model(optimizer)

    # ============================================================
    # Variables
    # ============================================================

    @variable(model, 0 <= Q <= 10)

    @variable(model, 0 <= QFR <= 10)

    @variable(model, 0 <= F[I] <= 10)

    @variable(model, 0 <= FR[I] <= 10)

    @variable(model, -10 <= rate[I] <= 10)

    @variable(model, 0 <= V <= 10)

    @variable(model, 0 <= c <= 10)

    @variable(model, 0 <= QR <= 10)

    @variable(model, 0 <= QP <= 10)

    @variable(model, 0 <= R[I] <= 10)

    @variable(model, 0 <= P[I] <= 10)


    # ============================================================
    # Mole balances
    # ============================================================

    @constraint(
        model,
        [i in I],
        F0[i] + FR[i] - F[i] + rate[i] * V == 0
    )

    @constraint(
        model,
        QF0 + QFR - Q == 0
    )


    # ============================================================
    # Splitter
    # ============================================================

    @constraint(
        model,
        [i in I],
        F[i] - P[i] - R[i] == 0
    )

    @constraint(
        model,
        Q - QP - QR == 0
    )

    @constraint(
        model,
        [i in I],
        P[i] * Q - F[i] * QP == 0
    )

    @constraint(
        model,
        0.95 * QP - P[:B] == 0
    )


    # ============================================================
    # CSTR equations
    # ============================================================

    @constraint(
        model,
        rate[:A] * Q^2
        + k * F[:A] * F[:B] == 0
    )

    @constraint(
        model,
        rate[:B] + rate[:A] == 0
    )

    @constraint(
        model,
        c - V == 0
    )


    # ============================================================
    # Recycle
    #
    # With one reactor, there is no recycle-location decision.
    # The recycle is simply associated with this reactor.
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
# Solve exact one-reactor model
# ============================================================

model = build_cstr_1reactor(Gurobi.Optimizer)

set_optimizer_attribute(
    model,
    "NonConvex",
    2
)

optimize!(model)

# JuMP provides access to MathOptInterface through JuMP.MOI
println("\n========== INFEASIBILITY DIAGNOSTICS ==========")
println("Termination status: ", termination_status(model))
println("Primal status: ", primal_status(model))

if termination_status(model) == JuMP.MOI.INFEASIBLE
    compute_conflict!(model)

    conflict_status = get_attribute(
        model,
        JuMP.MOI.ConflictStatus(),
    )
    println("Conflict status: ", conflict_status)

    if conflict_status == JuMP.MOI.CONFLICT_FOUND
        println("\nConstraints and bounds in the conflict:")

        for (Ftype, Stype) in list_of_constraint_types(model)
            for con in all_constraints(model, Ftype, Stype)
                status = get_attribute(
                    con,
                    JuMP.MOI.ConstraintConflictStatus(),
                )

                if status == JuMP.MOI.IN_CONFLICT
                    println(con)
                end
            end
        end
    end
end
# ============================================================
# Results
# ============================================================

println("\n========== ONE-REACTOR EXACT RESULTS ==========")

println(
    "Status = ",
    termination_status(model)
)

if has_values(model)

    println(
        "Objective = ",
        objective_value(model)
    )

    println(
        "V = ",
        value(model[:V])
    )

    println(
        "Q = ",
        value(model[:Q])
    )

    println(
        "rate[A] = ",
        value(model[:rate][:A])
    )

    println(
        "rate[B] = ",
        value(model[:rate][:B])
    )

    println(
        "F[A] = ",
        value(model[:F][:A])
    )

    println(
        "F[B] = ",
        value(model[:F][:B])
    )

    println(
        "QR = ",
        value(model[:QR])
    )

    println(
        "QP = ",
        value(model[:QP])
    )


    rA = value(model[:rate][:A])
    rB = value(model[:rate][:B])
    Vv = value(model[:V])

    PA = value(model[:P][:A])
    PB = value(model[:P][:B])

    println("\n--- Exact-model consistency check ---")
    println("P[A] = ", PA)
    println("P[B] = ", PB)
    println("P[A] + P[B] = ", PA + PB)
    println("Fresh A+B = ", 0.99 + 0.01)

    println("rate[A]*V = ", rA * Vv)
    println("rate[B]*V = ", rB * Vv)
    println("Sum of reaction source terms = ",
            rA * Vv + rB * Vv) 
end