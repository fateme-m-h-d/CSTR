"""
Five-reactor CSTR superstructure in MOLAR-FLOW variables with one shared
reaction-extent ReLU NN reused at every candidate reactor.

NN mapping at reactor n:
    (FAin[n], FBin[n], Qin[n], V[n]) -> xi_pred[n]

where xi is the positive reaction extent/source [mol/s]:
    FAout = FAin - xi
    FBout = FBin + xi

For an active CSTR:
    xi_used[n] = xi_pred[n]
For a bypass:
    xi_used[n] = 0

This removes the explicit reaction-rate variables and the original kinetic
constraints from the optimization. The original splitter equations are kept,
so the model still contains bilinear splitter constraints.
"""

using DisjunctiveProgramming
using JuMP
using Gurobi
import PythonCall
import MathOptAI


const K_RXN = 2.0

function build_cstr_flow_extent_nn(optimizer; NT::Int = 5)

    I = [:A, :B]
    N = 1:NT

    QF0 = 1.0
    C0 = Dict(:A => 0.99, :B => 0.01)
    F0 = Dict(i => C0[i] * QF0 for i in I)

    # Used only for the post-solution physics audit.
    k = 2.0

    model = GDPModel(optimizer)

    # ============================================================
    # Original flow variables
    # ============================================================

    @variable(model, 0 <= Q[N] <= 10)
    @variable(model, 0 <= QFR[N] <= 10)

    @variable(model, 0 <= F[I, N] <= 10)
    @variable(model, 0 <= FR[I, N] <= 10)

    # rate[A,n] and rate[B,n] are intentionally removed.

    @variable(model, 0 <= V[N] <= 10)
    @variable(model, 0 <= c[N] <= 10)

    @variable(model, 0 <= QR <= 10)
    @variable(model, 0 <= QP <= 10)
    @variable(model, 0 <= R[I] <= 10)
    @variable(model, 0 <= P[I] <= 10)

    # ============================================================
    # Reactor inlet variables used by the NN
    # ============================================================

    @variable(model, 0 <= FAin[N] <= 10)
    @variable(model, 0 <= FBin[N] <= 10)
    @variable(model, 0 <= Qin[N] <= 10)

    # Feed-side unit NT: fresh feed + local recycle.
    @constraint(model, FAin[NT] == F0[:A] + FR[:A, NT])
    @constraint(model, FBin[NT] == F0[:B] + FR[:B, NT])
    @constraint(model, Qin[NT]  == QF0 + QFR[NT])

    # Units n < NT: upstream outlet + local recycle.
    @constraint(model, [n in 1:(NT - 1)],
        FAin[n] == F[:A, n + 1] + FR[:A, n])
    @constraint(model, [n in 1:(NT - 1)],
        FBin[n] == F[:B, n + 1] + FR[:B, n])
    @constraint(model, [n in 1:(NT - 1)],
        Qin[n] == Q[n + 1] + QFR[n])

    # Exact invariant for this A -> B system with C_A + C_B = 1 mol/L.
    # It is linear and helps keep NN inputs on the physical training manifold.
    @constraint(model, [n in N], FAin[n] + FBin[n] == Qin[n])

    # ============================================================
    # Embed one copy of the SAME extent NN at every candidate reactor
    # ============================================================

    predictor = MathOptAI.PytorchModel(
        joinpath(@__DIR__, "extent_nn_5reactor.pt")
    )

    @variable(model, 0 <= FAin_scaled[N] <= 1)
    @variable(model, 0 <= FBin_scaled[N] <= 1)
    @variable(model, 0 <= Qin_scaled[N] <= 1)
    @variable(model, 0 <= V_scaled[N] <= 1)

    @constraint(model, [n in N], FAin_scaled[n] == FAin[n] / 10.0)
    @constraint(model, [n in N], FBin_scaled[n] == FBin[n] / 10.0)
    @constraint(model, [n in N], Qin_scaled[n]  == Qin[n]  / 10.0)
    @constraint(model, [n in N], V_scaled[n]    == V[n]    / 10.0)

    # Raw NN predictions may be slightly negative away from the relevant
    # manifold. xi_used is the physical nonnegative source used in balances.
    @variable(model, -10 <= xi_pred[N] <= 10)
    @variable(model, 0 <= xi_used[N] <= 10)

    for n in N
        nn_output, _ = MathOptAI.add_predictor(
            model,
            predictor,
            [
                FAin_scaled[n],
                FBin_scaled[n],
                Qin_scaled[n],
                V_scaled[n],
            ];
            config = Dict(
                :ReLU => MathOptAI.ReLUSOS1,
            ),
        )

        @constraint(model, xi_pred[n] == 10.0 * only(nn_output))
    end

    # ============================================================
    # Reactor material balances using ONE common reaction extent
    # ============================================================

    @constraint(model, [n in N],
        FAin[n] - F[:A, n] - xi_used[n] == 0)

    @constraint(model, [n in N],
        FBin[n] - F[:B, n] + xi_used[n] == 0)

    @constraint(model, [n in N],
        Qin[n] - Q[n] == 0)

    # ============================================================
    # Original splitter formulation (still bilinear)
    # ============================================================

    @constraint(model, [i in I], F[i, 1] - P[i] - R[i] == 0)
    @constraint(model, Q[1] - QP - QR == 0)

    # Equal composition across the splitter.
    # These are the remaining continuous nonlinear constraints.
    @constraint(model, [i in I],
        P[i] * Q[1] - F[i, 1] * QP == 0)

    @constraint(model, 0.95 * QP - P[:B] == 0)

    # Same equal-volume restriction as the original model.
    @constraint(model, [n in 2:NT], V[n] - V[n - 1] == 0)

    # ============================================================
    # Reactor 1 is always active
    # ============================================================

    @constraint(model, xi_used[1] - xi_pred[1] == 0)
    @constraint(model, c[1] - V[1] == 0)

    # ============================================================
    # CSTR / bypass decisions for reactors 2..NT
    # ============================================================

    Yp = Dict{Int, Any}()

    for n in 2:NT
        Y = @variable(model, [1:2], Logical, base_name = "Yp_$(n)")

        # Active CSTR: use the NN reaction source and count volume.
        @constraint(model, xi_used[n] - xi_pred[n] == 0, Disjunct(Y[1]))
        @constraint(model, c[n] - V[n] == 0, Disjunct(Y[1]))

        # Bypass: no reaction and no recycle can enter this bypassed unit.
        # The global balances then enforce pass-through of F and Q.
        @constraint(model, xi_used[n] == 0, Disjunct(Y[2]))
        @constraint(model, [i in I], FR[i, n] == 0, Disjunct(Y[2]))
        @constraint(model, QFR[n] == 0, Disjunct(Y[2]))
        @constraint(model, c[n] == 0, Disjunct(Y[2]))

        @disjunction(model, Y)
        Yp[n] = Y
    end

    # Active-prefix logic: if reactor n is active, n-1 must be active.
    for n in 3:NT
        @constraint(model, Yp[n][1] ⟹ Yp[n - 1][1] := true)
    end

    # ============================================================
    # Recycle location: exactly one location
    # ============================================================

    Yr = @variable(model, [N], Logical, base_name = "Yr")

    for kk in N
        @constraint(model, [i in I], FR[i, kk] - R[i] == 0, Disjunct(Yr[kk]))
        @constraint(model, QFR[kk] - QR == 0, Disjunct(Yr[kk]))

        for n in N
            n == kk && continue
            @constraint(model, [i in I], FR[i, n] == 0, Disjunct(Yr[kk]))
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


# ============================================================
# Build and solve
# ============================================================

model = build_cstr_flow_extent_nn(Gurobi.Optimizer)

# The reactor kinetics have been replaced by the ReLU NN, but the original
# splitter still has bilinear continuous constraints.
set_optimizer_attribute(model, "NonConvex", 2)
set_optimizer_attribute(model, "TimeLimit", 120.0)

elapsed = @elapsed begin
    optimize!(model, gdp_method = BigM())
end

println("\n========== FIVE-REACTOR FLOW EXTENT-NN ==========")
println("Status = ", termination_status(model))
println("Gurobi solve time = ", solve_time(model), " s")
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

    println("\nFAin = ")
    println(value.(model[:FAin]))

    println("\nFBin = ")
    println(value.(model[:FBin]))

    println("\nQin = ")
    println(value.(model[:Qin]))

    println("\nF[A,:] = ")
    println([value(model[:F][:A, n]) for n in 1:5])

    println("\nF[B,:] = ")
    println([value(model[:F][:B, n]) for n in 1:5])

    println("\nxi_used = ")
    println(value.(model[:xi_used]))

    println("\nxi_pred = ")
    println(value.(model[:xi_pred]))

    # ------------------------------------------------------------
    # Original-physics audit at every candidate reactor.
    # For a CSTR:
    #   xi_exact = k*V/Q^2 * FAout*FBout
    # ------------------------------------------------------------
    println("\n--- ORIGINAL-PHYSICS AUDIT ---")
    for n in 1:5
        FAout = value(model[:F][:A, n])
        FBout = value(model[:F][:B, n])
        Qout = value(model[:Q][n])
        Vv = value(model[:V][n])
        xi_nn = value(model[:xi_used][n])

        xi_exact = K_RXN * Vv * FAout * FBout / (Qout^2)

        println(
            "n=", n,
            ": xi_used=", xi_nn,
            ", exact xi=", xi_exact,
            ", error=", xi_nn - xi_exact,
        )
    end

    println("\nP[A] = ", value(model[:P][:A]))
    println("P[B] = ", value(model[:P][:B]))
    println("QR = ", value(model[:QR]))
    println("QP = ", value(model[:QP]))
end
