"""
Five-reactor CSTR superstructure written in concentration variables.

This preserves the original GDP design decisions:
  * reactor 1 is always an active CSTR;
  * reactors 2..NT choose CSTR or bypass;
  * active reactors form a prefix;
  * exactly one recycle location is selected;
  * recycle may only enter an active CSTR;
  * all candidate reactor volumes are equal;
  * objective = minimum total active reactor volume.

Concentration variables:
    C[i,n] : concentration of component i leaving unit n [mol/L]
    CR[i]  : recycle concentration [mol/L]

Because the downstream unit is a simple splitter,
    CR[i] = C[i,1]
and product concentration also equals C[i,1].

The original molar-flow variables F, FR, P and R are not needed.
"""

using DisjunctiveProgramming
using JuMP
using Gurobi

const CSTR_KNOWN_OPT = 3.0620145766

function build_cstr_concentration_exact(optimizer; NT::Int = 5)

    I = [:A, :B]
    N = 1:NT

    k = 2.0
    QF0 = 1.0
    C0 = Dict(:A => 0.99, :B => 0.01)

    model = GDPModel(optimizer)

    # ============================================================
    # Continuous variables
    # ============================================================

    @variable(model, 0 <= Q[N] <= 10)           # outlet flow of unit n [L/s]
    @variable(model, 0 <= QFR[N] <= 10)         # recycle flow injected at n [L/s]

    @variable(model, 0 <= C[I, N] <= 1)         # outlet concentrations [mol/L]
    @variable(model, 0 <= CR[I] <= 1)           # recycle concentration [mol/L]

    @variable(model, -10 <= rate[I, N] <= 10)  # species rates [mol/(L*s)]

    @variable(model, 0 <= V[N] <= 10)           # candidate reactor volume [L]
    @variable(model, 0 <= c[N] <= 10)           # active volume counted in objective [L]

    @variable(model, 0 <= QR <= 10)             # total recycle flow [L/s]
    @variable(model, 0 <= QP <= 10)             # product flow [L/s]

    # ============================================================
    # Splitter
    # ============================================================

    @constraint(model, Q[1] - QP - QR == 0)

    # Simple splitter: recycle/product composition = reactor-1 outlet composition.
    @constraint(model, [i in I], CR[i] - C[i, 1] == 0)

    # Original P[B]/QP = 0.95 becomes product/outlet concentration = 0.95.
    @constraint(model, C[:B, 1] == 0.95)

    # ============================================================
    # Global reactor balances in concentration form
    # ============================================================

    # Feed-side unit n = NT:
    # QF0*C0 + QFR*CR - Q*C + rate*V = 0
    @constraint(
        model,
        [i in I],
        QF0 * C0[i] + QFR[NT] * CR[i] - Q[NT] * C[i, NT]
        + rate[i, NT] * V[NT] == 0
    )

    @constraint(model, QF0 + QFR[NT] - Q[NT] == 0)

    # Other units n < NT:
    # upstream outlet + recycle injection - local outlet + reaction = 0
    @constraint(
        model,
        [i in I, n in 1:(NT - 1)],
        Q[n + 1] * C[i, n + 1] + QFR[n] * CR[i]
        - Q[n] * C[i, n] + rate[i, n] * V[n] == 0
    )

    @constraint(
        model,
        [n in 1:(NT - 1)],
        Q[n + 1] + QFR[n] - Q[n] == 0
    )

    # Equal candidate reactor volumes, same as the original model.
    @constraint(model, [n in 2:NT], V[n] - V[n - 1] == 0)

    # ============================================================
    # Reactor 1: always active CSTR
    # ============================================================

    @constraint(model, rate[:A, 1] + k * C[:A, 1] * C[:B, 1] == 0)
    @constraint(model, rate[:B, 1] + rate[:A, 1] == 0)
    @constraint(model, c[1] - V[1] == 0)

    # ============================================================
    # CSTR / bypass decisions for n = 2..NT
    # ============================================================

    Yp = Dict{Int, Any}()

    for n in 2:NT
        Y = @variable(model, [1:2], Logical, base_name = "Yp_$(n)")

        # Active CSTR equations.
        @constraint(
            model,
            rate[:A, n] + k * C[:A, n] * C[:B, n] == 0,
            Disjunct(Y[1]),
        )
        @constraint(model, rate[:B, n] + rate[:A, n] == 0, Disjunct(Y[1]))
        @constraint(model, c[n] - V[n] == 0, Disjunct(Y[1]))

        # Bypass: no reaction, no recycle injected at that location,
        # and no active volume counted. The global balances then enforce
        # unchanged composition through the bypass.
        @constraint(model, [i in I], rate[i, n] == 0, Disjunct(Y[2]))
        @constraint(model, QFR[n] == 0, Disjunct(Y[2]))
        @constraint(model, c[n] == 0, Disjunct(Y[2]))

        @disjunction(model, Y)
        Yp[n] = Y
    end

    # Active-prefix structure.
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


model = build_cstr_concentration_exact(Gurobi.Optimizer)

set_optimizer_attribute(model, "NonConvex", 2)

elapsed = @elapsed begin
    optimize!(model, gdp_method = BigM())
end

println("\n========== FIVE-REACTOR CONCENTRATION EXACT ==========")
println("Status = ", termination_status(model))
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

    println("\nC[A,:] = ")
    println([value(model[:C][:A, n]) for n in 1:5])

    println("\nC[B,:] = ")
    println([value(model[:C][:B, n]) for n in 1:5])

    println("\nrate[A,:] = ")
    println([value(model[:rate][:A, n]) for n in 1:5])

    println("\nrate[B,:] = ")
    println([value(model[:rate][:B, n]) for n in 1:5])

    println("\nQR = ", value(model[:QR]))
    println("QP = ", value(model[:QP]))
end
