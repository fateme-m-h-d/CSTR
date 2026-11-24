include(joinpath(@__DIR__, "common.jl"))

t = parse(Float64, get(ARGS, 1, "391.41"))
a = parse(Float64, get(ARGS, 2, "1.0111"))
steps = parse(Int, get(ARGS, 3, "6"))
damping = parse(Float64, get(ARGS, 4, "1.0"))

torch = PythonCall.pyimport("torch")
checkpoint = joinpath(@__DIR__, "checkpoints", "cstr_nn_tanh.pt")
network = torch.load(checkpoint; map_location = "cpu", weights_only = false)
network = network.double().eval()

input = torch.tensor([[t / 460, a / 1.2]]; dtype = torch.float64)
scaled = PythonCall.pyconvert(
    Vector{Float64}, network(input).detach().reshape(-1).tolist()
)
raw = YSCALE .* scaled

y = copy(raw)
lambda = zeros(2)
identity3 = Matrix{Float64}(I, 3, 3)

println("Fixed inputs = ", (t, a))
println("Raw NN concentrations = ", raw)

for k in 0:steps
    g = residual(t, a, y)
    J = jac_y(t, y)
    stationarity = y - raw + J' * lambda
    F = [stationarity; g]

    println((step = k, kinetic = g[1], summed_balance = g[2],
             stationarity = norm(stationarity, Inf),
             kkt = norm(F, Inf), min_concentration = minimum(y)))

    k == steps && break

    H = identity3 + lambda[1] *
        [hess_kinetic(t, y, i, j) for i in 1:3, j in 1:3]
    K = [H J'; J zeros(2, 2)]
    delta = K \ (-F)

    y .+= damping * delta[1:3]
    lambda .+= damping * delta[4:5]
end

println("Projected concentrations = ", y)