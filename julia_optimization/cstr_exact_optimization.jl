using JuMP
import Ipopt

const MOI = JuMP.MOI

function solve_cstr_conversion_optimization(;
    alpha = 0.2,
    beta = 0.2,
    T_start = 385.0,
    CA0_start = 0.88,
)

    # ============================================================
    # 1. Parameters
    # ============================================================

    tau = 10.0

    Af = 1.0e13
    Eaf = 90_000.0

    Ar = 1.0e11
    Ear = 80_000.0

    R = 8.314

    CB0 = 2.0
    CC0 = 0.0

    T_min = 280.0
    T_max = 460.0

    CA0_min = 0.8
    CA0_max = 1.2

    # ============================================================
    # 2. Optimization model
    # ============================================================

    model = Model(Ipopt.Optimizer)

    set_optimizer_attribute(model, "tol", 1.0e-10)
    set_optimizer_attribute(model, "max_iter", 3000)
    set_optimizer_attribute(model, "print_level", 5)

    # ============================================================
    # 3. Variables
    # ============================================================

    @variable(
        model,
        T_min <= T <= T_max,
        start = T_start
    )

    @variable(
        model,
        CA0_min <= CA0 <= CA0_max,
        start = CA0_start
    )

    @variable(model, CA >= 0.0, start = 0.37)
    @variable(model, CB >= 0.0, start = 0.98)
    @variable(model, CC >= 0.0, start = 1.52)

    # ============================================================
    # 4. Derived expressions
    # ============================================================

    @expression(
        model,
        x,
        CA0 - CA
    )

    @expression(
        model,
        conversion,
        (CA0 - CA) / CA0
    )

    @expression(
        model,
        kf,
        Af * exp(-Eaf / (R * T))
    )

    @expression(
        model,
        kr,
        Ar * exp(-Ear / (R * T))
    )

    # ============================================================
    # 5. Reactor constraints
    # ============================================================

    @constraint(
        model,
        reaction_balance,
        CA0 - CA ==
        tau * (
            kf * CA * CB^2 -
            kr * CC
        )
    )

    @constraint(
        model,
        B_balance,
        CB0 - CB ==
        2.0 * (CA0 - CA)
    )

    @constraint(
        model,
        mass_balance,
        CA0 - CA +
        CB0 - CB +
        CC0 - CC ==
        0.0
    )

    # ============================================================
    # 6. Normalized operating variables
    # ============================================================

    @expression(
        model,
        T_hat,
        (T - T_min) / (T_max - T_min)
    )

    @expression(
        model,
        CA0_hat,
        (CA0 - CA0_min) / (CA0_max - CA0_min)
    )

    # ============================================================
    # 7. Objective
    # ============================================================

    @objective(
        model,
        Max,
        conversion -
        alpha * T_hat^2 -
        beta * (CA0_hat - 0.5)^2
    )

    # ============================================================
    # 8. Solve
    # ============================================================

    optimize!(model)

    status = termination_status(model)

    if status ∉ (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
        error("Optimization failed. Status: $status")
    end

    # ============================================================
    # 9. Results
    # ============================================================

    T_opt = value(T)
    CA0_opt = value(CA0)

    CA_opt = value(CA)
    CB_opt = value(CB)
    CC_opt = value(CC)

    x_opt = value(x)
    conversion_opt = value(conversion)

    T_hat_opt = value(T_hat)
    CA0_hat_opt = value(CA0_hat)

    J_opt = objective_value(model)

    # ============================================================
    # 10. Constraint residuals
    # ============================================================

    kf_opt = Af * exp(-Eaf / (R * T_opt))
    kr_opt = Ar * exp(-Ear / (R * T_opt))

    reaction_residual =
        CA0_opt - CA_opt -
        tau * (
            kf_opt * CA_opt * CB_opt^2 -
            kr_opt * CC_opt
        )

    B_balance_residual =
        CB0 - CB_opt -
        2.0 * (CA0_opt - CA_opt)

    mass_residual =
        CA0_opt - CA_opt +
        CB0 - CB_opt +
        CC0 - CC_opt

    println()
    println("==================================================")
    println("CSTR conversion optimization result")
    println("==================================================")

    println("Termination status   = ", status)
    println("Primal status        = ", primal_status(model))

    println()
    println("Decision variables:")
    println("T                    = ", T_opt, " K")
    println("CA0                  = ", CA0_opt)

    println()
    println("Reactor outputs:")
    println("CA                   = ", CA_opt)
    println("CB                   = ", CB_opt)
    println("CC                   = ", CC_opt)

    println()
    println("Objective quantities:")
    println("x                    = ", x_opt)
    println("A conversion         = ", conversion_opt)
    println("T_hat                = ", T_hat_opt)
    println("CA0_hat              = ", CA0_hat_opt)
    println("Objective J          = ", J_opt)

    println()
    println("Constraint residuals:")
    println("reaction residual    = ", reaction_residual)
    println("B-balance residual   = ", B_balance_residual)
    println("mass residual        = ", mass_residual)

    println("==================================================")

    return (
        status = status,
        J = J_opt,
        T = T_opt,
        CA0 = CA0_opt,
        CA = CA_opt,
        CB = CB_opt,
        CC = CC_opt,
        x = x_opt,
        conversion = conversion_opt,
        reaction_residual = reaction_residual,
        B_balance_residual = B_balance_residual,
        mass_residual = mass_residual,
    )
end

result = solve_cstr_conversion_optimization(
    alpha = 0.2,
    beta = 0.2,
)