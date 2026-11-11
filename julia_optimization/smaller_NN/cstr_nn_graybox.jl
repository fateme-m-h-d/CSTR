using JuMP
import Ipopt
import MathOptAI
import PythonCall

const MOI = JuMP.MOI

function solve_cstr_nn_graybox(;
    activation = "relu")

    # ------------------------------------------------------
    # Optimization model
    # ------------------------------------------------------

    model = Model(Ipopt.Optimizer)

    set_optimizer_attribute(model, "tol", 1.0e-8)
    set_optimizer_attribute(model, "max_iter", 3000)
    set_optimizer_attribute(model, "print_level", 5)

    # ------------------------------------------------------
    # Decision variables = NN inputs
    # ------------------------------------------------------

    @variable(
        model,
        280.0 <= T <= 350.0,
        start = 385.0
    )

    @variable(
        model,
        0.8 <= CA0 <= 1.2,
        start = 0.88
    )


    # Scale physical optimization variables
    @expression(model, T_scaled, T / 350.0)
    @expression(model, CA0_scaled, CA0 / 1.2)
    # ------------------------------------------------------
    # Load PyTorch NN
    # ------------------------------------------------------

    # predictor = MathOptAI.PytorchModel(
    #     joinpath(@__DIR__, "cstr_nn.pt")
    # )

    model_path = joinpath(
        @__DIR__,
        "cstr_nn_$(activation).pt"
    )

    predictor = MathOptAI.PytorchModel(
        model_path
    )

    # ------------------------------------------------------
    # Embed NN as GRAY BOX
    #
    # inputs  = [T, CA0]
    # outputs = [CA, CB, CC]
    # ------------------------------------------------------

    y_scaled, formulation = MathOptAI.add_predictor(
        model,
        predictor,
        [T_scaled, CA0_scaled];
        gray_box = true
        
    )

    CA_scaled = y_scaled[1]
    CB_scaled = y_scaled[2]
    CC_scaled = y_scaled[3]

    
    # Convert outputs back to physical units
    # @expression(model, CA, CA_scaled * 1.19241398)
    # @expression(model, CB, CB_scaled * 1.98983454)
    # @expression(model, CC, CC_scaled * 1.95362953)
    # @expression(model, CA, CA_scaled * 1.1923875)
    # @expression(model, CB, CB_scaled * 1.9897989)
    # @expression(model, CC, CC_scaled * 2.85705345)
    @expression(model, CA, CA_scaled * 1.1923874953)
    @expression(model, CB, CB_scaled * 1.9897988732)
    @expression(model, CC, CC_scaled * 2.0837321561)


    # Optional physical bounds
    @constraint(model, CA >= 0.0)
    @constraint(model, CB >= 0.0)
    @constraint(model, CC >= 0.0)


    # ------------------------------------------------------
    # Objective
    # ------------------------------------------------------

    @expression(
        model,
        conversion,
        (CA0 - CA) / CA0
    )

    # @objective(
    #     model,
    #     Max,
    #     conversion
    # )

    @expression(model, x, CA0 - CA)

    lambda_CA0 = 0.3

    @expression(
        model,
        economic_objective,
        x - lambda_CA0 * CA0
    )

    @objective(
        model,
        Max,
        economic_objective
    )

    # ------------------------------------------------------
    # Solve
    # ------------------------------------------------------

    optimize!(model)

    println()
    println("Status     = ", termination_status(model))
    println("T          = ", value(T))
    println("CA0        = ", value(CA0))
    println("CA         = ", value(CA))
    println("CB         = ", value(CB))
    println("CC         = ", value(CC))
    println("Conversion = ", value(conversion))

    println()
    println("MathOptAI formulation:")
    println(formulation)

    return (
        activation = activation,
        status = termination_status(model),
        T = value(T),
        CA0 = value(CA0),
        CA = value(CA),
        CB = value(CB),
        CC = value(CC),
        conversion = value(conversion),
    )
end


# solve_cstr_nn_graybox(activation="relu")


activations = [
    "relu",
    "tanh",
    "softplus",
    "sigmoid",
]

results = []

for activation in activations

    println()
    println("======================================")
    println("Activation: ", activation)
    println("======================================")

    result = solve_cstr_nn_graybox(
        activation = activation
    )

    push!(results, result)
end


println()
println("==============================================================")
println("Activation comparison")
println("==============================================================")

for r in results
    println(
        r.activation,
        " | T = ", r.T,
        " | CA0 = ", r.CA0,
        " | CA = ", r.CA,
        " | Conversion = ", r.conversion,
        " | Status = ", r.status
    )
end