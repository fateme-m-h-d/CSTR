"""
Single-CSTR version of the CSTR superstructure.

Purpose:
Establish a one-reactor baseline before replacing any
bilinear terms with a neural network.
"""

using JuMP
using Gurobi
import PythonCall
import MathOptAI

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
    # Bilinear NN: approximate rate[A] * V
    # ============================================================

    predictor = MathOptAI.PytorchModel(
        joinpath(@__DIR__, "bilinear_nn_focused.pt")
    )

    # Training scaling:
    # x_scaled = x / 10
    # y_scaled = y / 10
    # z_scaled = xy / 100

    @variable(model, -1 <= rateA_scaled <= 1)
    @variable(model,  0 <= V_scaled <= 1)

    @constraint(model,
        rateA_scaled == rate[:A] / 10.0
    )

    @constraint(model,
        V_scaled == V / 10.0
    )

    nn_output, _ = MathOptAI.add_predictor(
        model,
        predictor,
        [rateA_scaled, V_scaled];
        config = Dict(
            :ReLU => MathOptAI.ReLUSOS1,
        ),
    )

    # Convert NN output back to physical units
    @variable(model, -100 <= rateA_V_NN <= 100)

    @constraint(model,
        rateA_V_NN == 100.0 * only(nn_output)
    )
    # ============================================================
    # Mole balances
    # ============================================================

    # @constraint(
    #     model,
    #     [i in I],
    #     F0[i] + FR[i] - F[i] + rate[i] * V == 0
    # )
    @constraint(
    model,
    F0[:A] + FR[:A] - F[:A] + rateA_V_NN == 0
    )


    @constraint(
    model,
    F0[:B] + FR[:B] - F[:B] + rate[:B] * V == 0
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

set_optimizer_attribute(model, "TimeLimit", 120.0)

optimize!(model)

# if termination_status(model) == JuMP.MOI.INFEASIBLE
#     compute_conflict!(model)

#     iis_path = joinpath(@__DIR__, "cstr_conflict.ilp")
#     return_code = Gurobi.GRBwrite(
#         unsafe_backend(model),
#         iis_path,
#     )

#     if return_code != 0
#         error("Gurobi export failed with code $return_code")
#     end

#     println("Conflict saved to: ", iis_path)
# end

# ============================================================
# Results
# ============================================================


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

end


println("\n========== ONE-REACTOR ONE-NN RESULTS ==========")

println("Status = ", termination_status(model))

if has_values(model)

    println("Objective = ", objective_value(model))

    rA_val = value(model[:rate][:A])
    V_val  = value(model[:V])
    nn_val = value(model[:rateA_V_NN])

    exact_val = rA_val * V_val

    println("\n--- Bilinear NN check ---")
    println("rate[A]       = ", rA_val)
    println("V             = ", V_val)
    println("Exact rateA*V = ", exact_val)
    println("NN rateA*V    = ", nn_val)
    println("NN error      = ", nn_val - exact_val)

    println("\n--- Process solution ---")
    println("rate[B] = ", value(model[:rate][:B]))
    println("Q       = ", value(model[:Q]))
    println("F[A]    = ", value(model[:F][:A]))
    println("F[B]    = ", value(model[:F][:B]))
    println("QR      = ", value(model[:QR]))
    println("QP      = ", value(model[:QP]))



    PA = value(model[:P][:A])
    PB = value(model[:P][:B])

    rA = value(model[:rate][:A])
    rB = value(model[:rate][:B])
    Vv = value(model[:V])

    nnA = value(model[:rateA_V_NN])

    exact_A_source = rA * Vv
    exact_B_source = rB * Vv

    println("\n--- PHYSICAL CONSISTENCY AUDIT ---")

    println("P[A] = ", PA)
    println("P[B] = ", PB)

    println("Fresh total molar flow = ", 1.0)
    println("Product total molar flow = ", PA + PB)
    println("Product excess/deficit = ", PA + PB - 1.0)

    println("\nExact rate[A]*V = ", exact_A_source)
    println("NN A source      = ", nnA)
    println("Exact rate[B]*V = ", exact_B_source)

    println("NN multiplication error = ",
            nnA - exact_A_source)

    println("A-source + B-source used in balances = ",
            nnA + exact_B_source)

end