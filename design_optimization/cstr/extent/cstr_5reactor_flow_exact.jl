"""
Exact five-reactor CSTR superstructure in molar-flow variables.
This is the reference model for comparison with cstr_5reactor_extent_nn.jl.
"""

using DisjunctiveProgramming
using JuMP
using Gurobi

const K_RXN = 2.0
const CSTR_KNOWN_OPT = 3.0620145766


NT = 5 
function build_cstr_flow_exact(optimizer; NT::Int = 5)
    I = [:A, :B]
    N = 1:NT
    QF0 = 1.0
    C0 = Dict(:A => 0.99, :B => 0.01)
    F0 = Dict(i => C0[i] * QF0 for i in I)

    model = GDPModel(optimizer)

    @variable(model, 0 <= Q[N] <= 10)
    @variable(model, 0 <= QFR[N] <= 10)
    @variable(model, 0 <= F[I,N] <= 10)
    @variable(model, 0 <= FR[I,N] <= 10)
    @variable(model, -10 <= rate[I,N] <= 10)
    @variable(model, 0 <= V[N] <= 10)
    @variable(model, 0 <= c[N] <= 10)
    @variable(model, 0 <= QR <= 10)
    @variable(model, 0 <= QP <= 10)
    @variable(model, 0 <= R[I] <= 10)
    @variable(model, 0 <= P[I] <= 10)

    # Feed-side reactor n = NT
    @constraint(model, [i in I],
        F0[i] + FR[i,NT] - F[i,NT] + rate[i,NT] * V[NT] == 0)
    @constraint(model, QF0 + QFR[NT] - Q[NT] == 0)

    # Remaining reactor balances
    @constraint(model, [i in I, n in 1:(NT-1)],
        F[i,n+1] + FR[i,n] - F[i,n] + rate[i,n] * V[n] == 0)
    @constraint(model, [n in 1:(NT-1)],
        Q[n+1] + QFR[n] - Q[n] == 0)

    # Splitter
    @constraint(model, [i in I], F[i,1] - P[i] - R[i] == 0)
    @constraint(model, Q[1] - QP - QR == 0)
    @constraint(model, [i in I], P[i] * Q[1] - F[i,1] * QP == 0)
    @constraint(model, 0.95 * QP - P[:B] == 0)

    # Equal reactor volumes
    @constraint(model, [n in 2:NT], V[n] - V[n-1] == 0)

    # Reactor 1 always active
    @constraint(model, rate[:A,1] * Q[1]^2 + K_RXN * F[:A,1] * F[:B,1] == 0)
    @constraint(model, rate[:B,1] + rate[:A,1] == 0)
    @constraint(model, c[1] - V[1] == 0)

    # CSTR/bypass decisions for reactors 2..NT
    Yp = Dict{Int,Any}()
    for n in 2:NT
        Y = @variable(model, [1:2], Logical, base_name="Yp_$(n)")

        # Active CSTR
        @constraint(model,
            rate[:A,n] * Q[n]^2 + K_RXN * F[:A,n] * F[:B,n] == 0,
            Disjunct(Y[1]))
        @constraint(model, rate[:B,n] + rate[:A,n] == 0, Disjunct(Y[1]))
        @constraint(model, c[n] - V[n] == 0, Disjunct(Y[1]))

        # Bypass
        @constraint(model, [i in I], rate[i,n] == 0, Disjunct(Y[2]))
        @constraint(model, [i in I], FR[i,n] == 0, Disjunct(Y[2]))
        @constraint(model, QFR[n] == 0, Disjunct(Y[2]))
        @constraint(model, c[n] == 0, Disjunct(Y[2]))

        @disjunction(model, Y)
        Yp[n] = Y
    end

    # Active-prefix logic
    for n in 3:NT
        @constraint(model, Yp[n][1] ⟹ Yp[n-1][1] := true)
    end

    # Exactly one recycle location
    Yr = @variable(model, [N], Logical, base_name="Yr")
    for kk in N
        @constraint(model, [i in I], FR[i,kk] - R[i] == 0, Disjunct(Yr[kk]))
        @constraint(model, QFR[kk] - QR == 0, Disjunct(Yr[kk]))

        for n in N
            n == kk && continue
            @constraint(model, [i in I], FR[i,n] == 0, Disjunct(Yr[kk]))
            @constraint(model, QFR[n] == 0, Disjunct(Yr[kk]))
        end
    end
    @disjunction(model, Yr)

    # Recycle can only enter an active CSTR
    for kk in 2:NT
        @constraint(model, Yr[kk] ⟹ Yp[kk][1] := true)
    end

    @objective(model, Min, sum(c[n] for n in N))
    return model
end

model = build_cstr_flow_exact(Gurobi.Optimizer)
set_optimizer_attribute(model, "NonConvex", 2)
set_optimizer_attribute(model, "TimeLimit", 120.0)

elapsed = @elapsed begin
    optimize!(model, gdp_method = BigM())
end

println("\n========== FIVE-REACTOR FLOW EXACT ==========")
println("Status = ", termination_status(model))
println("Gurobi solve time = ", solve_time(model), " s")
println("Total optimize! time = ", elapsed, " s")

if has_values(model)
    println("Objective = ", objective_value(model))
    println("Known original optimum = ", CSTR_KNOWN_OPT)
    println("\nQ = ")
    println(value.(model[:Q]))
    println("\nQFR = ")
    println(value.(model[:QFR]))
    println("\nV = ")
    println(value.(model[:V]))
    println("\nActive volume c = ")
    println(value.(model[:c]))
    println("\nF[A,:] = ")
    println([value(model[:F][:A,n]) for n in 1:NT])
    println("\nF[B,:] = ")
    println([value(model[:F][:B,n]) for n in 1:NT])
    println("\nrate[A,:] = ")
    println([value(model[:rate][:A,n]) for n in 1:NT])
    println("\nrate[B,:] = ")
    println([value(model[:rate][:B,n]) for n in 1:NT])
    println("\nP[A] = ", value(model[:P][:A]))
    println("P[B] = ", value(model[:P][:B]))
    println("QR = ", value(model[:QR]))
    println("QP = ", value(model[:QP]))
end
