import BilevelJuMP
const BL = BilevelJuMP

# BilevelJuMP uses legacy JuMP nonlinear syntax in the leader. Read only
# the layers of the existing Sequential checkpoint. No training is performed.
function tanh_layers(path)
    pc = PythonCall
    torch = pc.pyimport("torch")
    network = torch.load(path; map_location = "cpu", weights_only = false)
    layers = []
    for layer in network
        kind = pc.pyconvert(String, layer.__class__.__name__)
        if kind == "Linear"
            W = pc.pyconvert(Vector{Vector{Float64}}, layer.weight.detach().tolist())
            b = pc.pyconvert(Vector{Float64}, layer.bias.detach().tolist())
            push!(layers, (W, b))
        elseif kind == "Tanh"
            push!(layers, :tanh)
        else
            error("Unexpected layer in the provided tanh checkpoint: $kind")
        end
    end
    return layers
end

function upper_network(model, t, a, layers)
    upper_only = BL.UpperOnly(model)
    previous = [t / 460, a / 1.2]
    for layer in layers
        if layer === :tanh
            next = @variable(upper_only, [1:length(previous)])
            @NLconstraint(BL.Upper(model), [i in 1:length(previous)],
                next[i] == tanh(previous[i]))
        else
            W, b = layer
            next = @variable(upper_only, [1:length(b)])
            @constraint(BL.Upper(model), [i in 1:length(b)],
                next[i] == sum(W[i][j] * previous[j] for j in eachindex(previous)) + b[i])
        end
        previous = next
    end
    @variable(BL.Upper(model), raw[1:3])
    @constraint(BL.Upper(model), [i in 1:3], raw[i] == YSCALE[i] * previous[i])
    return raw
end

function solve_bilevel(region, start, path, coefficients)
    model = BL.BilevelModel(Ipopt.Optimizer; mode = BL.ProductMode(1e-8))
    upper, lower = BL.Upper(model), BL.Lower(model)
    t, a = operating_inputs(upper, region, start)
    raw = upper_network(model, t, a, tanh_layers(path))
    initial = reference_outlets(start...)
    @variable(lower, y[i in 1:3], start = initial[i])
    # (; A, B, b) = coefficients
    # @constraint(lower, [j in 1:2],
    #     A[j,1] * t + A[j,2] * a + sum(B[j,i] * y[i] for i in 1:3) == b[j])
    (; A, B, b) = coefficients
    @constraint(lower, [j in eachindex(b)],
        A[j,1] * t + A[j,2] * a +
        sum(B[j,i] * y[i] for i in 1:3) == b[j]
    )
    @objective(lower, Min, sum((y[i] - raw[i])^2 for i in 1:3) / 2)
    set_outer_objective(upper, t, a, y)
    optimize!(model)
    return collect_result(model, t, a, y, raw;
        diagnostic = () -> affine_error(t, a, y, coefficients))
end
