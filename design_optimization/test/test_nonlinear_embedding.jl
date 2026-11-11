ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] =
    raw"/home/f25moham/CSTR/myenv/bin/python"

using JuMP
using Gurobi
import PythonCall
import MathOptAI

model = Model(Gurobi.Optimizer)

# Physical variables
@variable(model, -10 <= x <= 10)
@variable(model, 0 <= y <= 10)

# Fix a test point: x = 2, y = 3
fix(x, 2.0; force = true)
fix(y, 3.0; force = true)

# Scaling used during Python training
x_scaled = x / 10.0
y_scaled = y / 10.0

# Load PyTorch NN from the same folder as this Julia file
predictor = MathOptAI.PytorchModel(
    joinpath(@__DIR__, "bilinear_nn.pt")
)

# Embed NN
nn_output, formulation = MathOptAI.add_predictor(
    model,
    predictor,
    [x_scaled, y_scaled];
    config = Dict(
        :ReLU => MathOptAI.ReLUSOS1,
    ),
)

# NN output was trained as z_scaled = z / 100
z_nn = 100.0 * only(nn_output)

@objective(model, Min, z_nn)

optimize!(model)

println()
println("Status      = ", termination_status(model))
println("x           = ", value(x))
println("y           = ", value(y))
println("Exact x*y   = ", value(x) * value(y))
println("NN product  = ", value(z_nn))
println("Error       = ", value(z_nn) - value(x) * value(y))