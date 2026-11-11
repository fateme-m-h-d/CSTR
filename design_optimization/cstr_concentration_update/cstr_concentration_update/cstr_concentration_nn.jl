"""
Five-reactor CSTR superstructure in concentration variables with the same
embedded concentration-based ReLU NN reused at every candidate reactor.

NN mapping at reactor n:
    (C[A,n], C[B,n], V[n]) -> xi_pred[n]

where xi is the positive reaction-source magnitude [mol/s]:
    xi = -rate_A*V = rate_B*V.

For an active CSTR:
    xi_used[n] = xi_pred[n]
For a bypass:
    xi_used[n] = 0

The global component balances use -xi_used for A and +xi_used for B, so the
A -> B stoichiometry is preserved exactly.

The GDP design logic is otherwise the same as the exact superstructure.
"""

using DisjunctiveProgramming
using JuMP
using Gurobi
import PythonCall
import MathOptAI

function build_cstr_concentration_nn(optimizer; NT::Int = 5)

    I = [:A, :B]
    N = 1:NT

    QF0 = 1.0
    C0 = Dict(:A => 0.99, :B => 0.01)

    model = GDPModel(optimizer)

    # ============================================================
    # Continuous variables
    # ============================================================

    @variable(model, 0 <= Q[N] <= 10)
    @variable(model, 0 <= QFR[N] <= 10)

    @variable(model, 0 <= C[I, N] <= 1)
    @variable(model, 0 <= CR[I] <= 1)

    @variable(model, 0 <= V[N] <= 10)
    @variable(model, 0 <= c[N] <= 10)

    @variable(model, 0 <= QR <= 10)
    @variable(model, 0 <= QP <= 10)

    # Raw NN predictions are allowed a small negative range so an irrelevant
    # prediction at a bypassed unit cannot by itself make the model infeasible.
    @variable(model, -10 <= xi_pred[N] <= 10)

    # xi_used is the physical reaction source used by the balances.
    @variable(model, 0 <= xi_used[N] <= 10)

    # ============================================================
    # Splitter
    # ============================================================

    @constraint(model, Q[1] - QP - QR == 0)
    @constraint(model, [i in I], CR[i] - C[i, 1] == 0)
    @constraint(model, C[:B, 1] == 0.95)

    # ============================================================
    # Embed one copy of the same NN for each candidate reactor
    # ============================================================

    predictor = MathOptAI.PytorchModel(
        joinpath(@__DIR__, "concentration_extent_nn_5reactor.pt")
    )

    @variable(model, 0 <= V_scaled[N] <= 1)
    @constraint(model, [n in N], V_scaled[n] == V[n] / 10.0)

    for n in N
        nn_output, _ = MathOptAI.add_predictor(
            model,
            predictor,
            [C[:A, n], C[:B, n], V_scaled[n]];
            config = Dict(
                :ReLU => MathOptAI.ReLUSOS1,
            ),
        )

        @constraint(model, xi_pred[n] == 10.0 * only(nn_output))
    end

    # ============================================================
    # Global component and flow balances
    # ============================================================

    # Feed-side unit NT.
    @constraint(
        model,
        QF0 * C0[:A] + QFR[NT] * CR[:A] - Q[NT] * C[:A, NT]
        - xi_used[NT] == 0
    )

    @constraint(
        model,
        QF0 * C0[:B] + QFR[NT] * CR[:B] - Q[NT] * C[:B, NT]
        + xi_used[NT] == 0
    )

    @constraint(model, QF0 + QFR[NT] - Q[NT] == 0)

    # Units n < NT.
    @constraint(
        model,
        [n in 1:(NT - 1)],
        Q[n + 1] * C[:A, n + 1] + QFR[n] * CR[:A]
        - Q[n] * C[:A, n] - xi_used[n] == 0
    )

    @constraint(
        model,
        [n in 1:(NT - 1)],
        Q[n + 1] * C[:B, n + 1] + QFR[n] * CR[:B]
        - Q[n] * C[:B, n] + xi_used[n] == 0
    )

    @constraint(
        model,
        [n in 1:(NT - 1)],
        Q[n + 1] + QFR[n] - Q[n] == 0
    )

    # Same equal-volume design restriction as original model.
    @constraint(model, [n in 2:NT], V[n] - V[n - 1] == 0)

    # ============================================================
    # Reactor 1: always active
    # ============================================================

    @constraint(model, xi_used[1] - xi_pred[1] == 0)
    @constraint(model, c[1] - V[1] == 0)

    # ============================================================
    # CSTR / bypass decisions for n = 2..NT
    # ============================================================

    Yp = Dict{Int, Any}()

    for n in 2:NT
        Y = @variable(model, [1:2], Logical, base_name = "Yp_$(n)")

        # Active CSTR: use the NN prediction and count reactor volume.
        @constraint(model, xi_used[n] - xi_pred[n] == 0, Disjunct(Y[1]))
        @constraint(model, c[n] - V[n] == 0, Disjunct(Y[1]))

        # Bypass: zero reaction source, zero recycle injected locally,
        # and zero active volume. Global balances enforce pass-through.
        @constraint(model, xi_used[n] == 0, Disjunct(Y[2]))
        @constraint(model, QFR[n] == 0, Disjunct(Y[2]))
        @constraint(model, c[n] == 0, Disjunct(Y[2]))

        @disjunction(model, Y)
        Yp[n] = Y
    end

    # Active-prefix logic.
    for n in 3:NT
        @constraint(model, Yp[n][1] ⟹ Yp[n - 1][1] := true)
    end

    # ============================================================
    # Recycle location: exactly one location
    # ============================================================

    Yr = @variable(model, [N], Logical, base_name = "Yr")

    for kk in N
        @constraint(model, QFR[kk] - QR == 0, Disjunct(Yr[kk]))

        for n in N
            n == kk && continue
            @constraint(model, QFR[n] == 0, Disjunct(Yr[kk]))
        end
    end

    @disjunction(model, Yr)

    # Recycle can only enter an active CSTR.
    for kk in 2:NT
        @constraint(model, Yr[kk] ⟹ Yp[kk][1] := true)
    end

    # ============================================================
    # Objective
    # ============================================================

    @objective(model, Min, sum(c[n] for n in N))

    return model
end


model = build_cstr_concentration_nn(Gurobi.Optimizer)

# The concentration network balances still contain Q*C products.
set_optimizer_attribute(model, "NonConvex", 2)
set_optimizer_attribute(model, "TimeLimit", 120.0)

elapsed = @elapsed begin
    optimize!(model, gdp_method = BigM())
end

println("\n========== FIVE-REACTOR CONCENTRATION NN ==========")
println("Status = ", termination_status(model))
println("Total optimize! time = ", elapsed, " s")

if has_values(model)
    println("Objective = ", objective_value(model))

    println("\nQ = ")
    println(value.(model[:Q]))

    println("\nQFR = ")
    println(value.(model[:QFR]))

    println("\nV = ")
    println(value.(model[:V]))

    println("\nActive volume c = ")
    println(value.(model[:c]))

    println("\nC[A,:] = ")
    println([value(model[:C][:A, n]) for n in 1:5])

    println("\nC[B,:] = ")
    println([value(model[:C][:B, n]) for n in 1:5])

    println("\nxi_used = ")
    println(value.(model[:xi_used]))

    println("\nxi_pred = ")
    println(value.(model[:xi_pred]))

    # Original-physics audit at every candidate reactor.
    println("\n--- ORIGINAL-PHYSICS AUDIT ---")
    for n in 1:5
        CA = value(model[:C][:A, n])
        CB = value(model[:C][:B, n])
        Vv = value(model[:V][n])
        xi_nn = value(model[:xi_used][n])
        xi_exact = K_RXN * CA * CB * Vv

        println(
            "n=", n,
            ": xi_used=", xi_nn,
            ", exact xi=", xi_exact,
            ", error=", xi_nn - xi_exact,
        )
    end

    println("\nQR = ", value(model[:QR]))
    println("QP = ", value(model[:QP]))
end
