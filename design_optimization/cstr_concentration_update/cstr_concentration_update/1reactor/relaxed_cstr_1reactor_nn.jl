"""
Single-CSTR version of the CSTR superstructure.

Purpose:
Establish an exact one-reactor baseline before replacing any
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
    # @constraint(
    # model,
    # F0[:A] + FR[:A] - F[:A] + rateA_V_NN == 0
    # )


    @variable(model, slack_A_pos >= 0)
    @variable(model, slack_A_neg >= 0)

    @constraint(model, balance_A_relaxed,
        F0[:A] + FR[:A] - F[:A] + rateA_V_NN ==
        slack_A_pos - slack_A_neg
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

    # @objective(
    #     model,
    #     Min,
    #     c
    # )
    @objective(model, Min, slack_A_pos + slack_A_neg)

    return model
end

# Execute the function and store the newly built relaxed model
model = build_cstr_1reactor(Gurobi.Optimizer)

set_optimizer_attribute(model, "NonConvex", 2)
set_optimizer_attribute(model, "TimeLimit", 120.0)

# Verify that the new model contains the slack variables
println("Positive slack: ", model[:slack_A_pos])
println("Negative slack: ", model[:slack_A_neg])
println("Objective: ", objective_function(model))

optimize!(model)

println("Status: ", termination_status(model))

if has_values(model)
    println("Minimum balance relaxation found: ",
        objective_value(model))
    println("Lower bound on required relaxation: ",
        objective_bound(model))

    rA = value(model[:rate][:A])
    vol = value(model[:V])
    prediction = value(model[:rateA_V_NN])

    println("V: ", vol)
    println("Exact rate[A]*V: ", rA * vol)
    println("NN prediction: ", prediction)
    println("NN error: ", prediction - rA * vol)
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

end