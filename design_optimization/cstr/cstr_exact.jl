"""
cstr — CSTR-series superstructure (Liñán et al. 2020; Ovalle et al. 2025),
as packaged in SECQUOIA/gdplib (`gdplib/cstr`, `gdp_reactor.py`). NT = 5
reactors in series with a single 1st-order autocatalytic reaction A → B;
minimize total reactor volume. Optimal layout: NT CSTRs with a recycle
before reactor NT.

**Nonconvex** MINLP: the mole balances carry `rate·V`, the splitter a
bilinear `P·Q`, and the CSTR disjunct the autocatalytic rate law
`rate_A·Q² + 2·F_A·F_B = 0`. So LOA is heuristic here (its OA cuts are not
valid underestimators), and the direct BigM/MBM solve needs Gurobi's
spatial branch-and-bound (`NonConvex=2`).

Encoding note — the gdplib model mixes cardinality and single-literal
indicator logic that DisjunctiveProgramming.jl does not express directly
(it has ∧ ∨ ¬ ⟹ ⇔ but no `exactly`, and rejects single-literal
propositions). Equivalent restructuring used here, same feasible set and
optimum:
  * reactor 1 is **always** a CSTR (its equations are global) — this forces
    `cstr[1]=true` without a single-literal constraint;
  * the cstr/bypass choice for reactors 2..NT is a 2-term disjunction with
    an active-prefix rule `cstr[n] ⟹ cstr[n-1]`;
  * the recycle location is a single **NT-way disjunction** (exactly-one
    for free), with `recycle@k ⟹ cstr[k]`.

Documented optimum: 3.0620145766.
"""

using DisjunctiveProgramming
using JuMP
using Gurobi

const CSTR_KNOWN_OPT = 3.0620145766

function build_cstr(optimizer; NT::Int = 5)
    I = [:A, :B]
    N = 1:NT
    k = 2.0
    QF0 = 1.0
    C0 = Dict(:A => 0.99, :B => 0.01)
    F0 = Dict(i => C0[i] * QF0 for i in I)

    model = GDPModel(optimizer)

    @variable(model, 0 <= Q[N] <= 10)
    @variable(model, 0 <= QFR[N] <= 10)
    @variable(model, 0 <= F[I, N] <= 10)
    @variable(model, 0 <= FR[I, N] <= 10)
    @variable(model, -10 <= rate[I, N] <= 10)
    @variable(model, 0 <= V[N] <= 10)
    @variable(model, 0 <= c[N] <= 10)
    @variable(model, 0 <= QR <= 10)
    @variable(model, 0 <= QP <= 10)
    @variable(model, 0 <= R[I] <= 10)
    @variable(model, 0 <= P[I] <= 10)

    # --- Global balances ---
    # Unreacted feed (n = NT): mole balance (nonlinear: rate·V) + continuity
    @constraint(model, [i in I],
        F0[i] + FR[i, NT] - F[i, NT] + rate[i, NT] * V[NT] == 0)
    @constraint(model, QF0 + QFR[NT] - Q[NT] == 0)
    # Reactor balances (n < NT)
    @constraint(model, [i in I, n in 1:(NT - 1)],
        F[i, n + 1] + FR[i, n] - F[i, n] + rate[i, n] * V[n] == 0)
    @constraint(model, [n in 1:(NT - 1)], Q[n + 1] + QFR[n] - Q[n] == 0)
    # Splitter (bilinear P·Q, F·QP)
    @constraint(model, [i in I], F[i, 1] - P[i] - R[i] == 0)
    @constraint(model, Q[1] - QP - QR == 0)
    @constraint(model, [i in I], P[i] * Q[1] - F[i, 1] * QP == 0)
    @constraint(model, QP * 0.95 - P[:B] == 0)
    # Equal reactor volumes
    @constraint(model, [n in 2:NT], V[n] - V[n - 1] == 0)

    # --- Reactor 1: always a CSTR (forces the active prefix to start at 1) ---
    @constraint(model, rate[:A, 1] * Q[1]^2 + k * F[:A, 1] * F[:B, 1] == 0)
    @constraint(model, rate[:B, 1] + rate[:A, 1] == 0)
    @constraint(model, c[1] - V[1] == 0)

    # --- CSTR / bypass disjunction, reactors 2..NT (1 = CSTR, 2 = bypass) ---
    Yp = Dict{Int, Any}()
    for n in 2:NT
        Y = @variable(model, [1:2], Logical, base_name = "Yp_$(n)")
        @constraint(model,
            rate[:A, n] * Q[n]^2 + k * F[:A, n] * F[:B, n] == 0,
            Disjunct(Y[1]))
        @constraint(model, rate[:B, n] + rate[:A, n] == 0, Disjunct(Y[1]))
        @constraint(model, c[n] - V[n] == 0, Disjunct(Y[1]))
        @constraint(model, [i in I], FR[i, n] == 0, Disjunct(Y[2]))
        @constraint(model, [i in I], rate[i, n] == 0, Disjunct(Y[2]))
        @constraint(model, QFR[n] == 0, Disjunct(Y[2]))
        @constraint(model, c[n] == 0, Disjunct(Y[2]))
        @disjunction(model, Y)
        Yp[n] = Y
    end
    for n in 3:NT
        @constraint(model, Yp[n][1] ⟹ Yp[n - 1][1] := true)
    end

    # --- Recycle location: exactly one, as a single NT-way disjunction ---
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
    for kk in 2:NT
        @constraint(model, Yr[kk] ⟹ Yp[kk][1] := true)
    end

    @objective(model, Min, sum(c[n] for n in N))

    return model
end





# ============================================================
# Build and solve the model
# ============================================================

model = build_cstr(Gurobi.Optimizer)

# The model still contains the other original nonlinear terms
set_optimizer_attribute(model, "NonConvex", 2)

# ============================================================
# Solve and record timing
# ============================================================

total_time = @elapsed begin
    optimize!(model, gdp_method = BigM())
end

solver_time = solve_time(model)

println("\n================ TIMING ================")
println("Gurobi solve time       = ", solver_time, " s")
println("Total optimize! time    = ", total_time, " s")

# ============================================================
# Report results
# ============================================================

println("\n================ RESULTS ================")
println("Termination status = ", termination_status(model))

if has_values(model)

    println("Objective = ", objective_value(model))

    # NN-replaced product
    r_val  = value(model[:rate][:A, 5])
    V_val  = value(model[:V][5])
    

    exact_product = r_val * V_val
    

    println("\n--- Bilinear NN check ---")
    println("rate[A,5] = ", r_val)
    println("V[5]      = ", V_val)
    println("Exact r*V = ", exact_product)
    

    println("\n--- Reactor volumes ---")
    println(value.(model[:V]))

    println("\n--- Reaction rates ---")
    println(value.(model[:rate]))

else
    println("No feasible solution values are available.")
end
