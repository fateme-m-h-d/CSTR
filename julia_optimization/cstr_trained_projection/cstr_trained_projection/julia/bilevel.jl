import BilevelJuMP
const BL = BilevelJuMP

# BilevelJuMP's supported legacy nonlinear leader interface.
# Replay the SAME exported backbone, including trained parameters and exact scaling.
function leader_backbone(model, t, a, directory, cfg, start)
    layer_data = JSON3.read(read(joinpath(directory,"layers.json"),String))
    numeric = PythonCall.pyconvert(Vector{Vector{Float64}},BRIDGE.numeric_layers(directory,collect(start)))
    previous = [t/Float64(cfg.input_scale[1]),a/Float64(cfg.input_scale[2])]
    for (k,layer) in enumerate(layer_data)
        next = @variable(BL.UpperOnly(model),[i in 1:length(numeric[k])],start=numeric[k][i])
        if layer.kind == "linear"
            W,b = layer.weight,layer.bias
            @constraint(BL.Upper(model),[i in eachindex(b)],
                next[i] == sum(Float64(W[i][j])*previous[j] for j in eachindex(previous))+Float64(b[i]))
        elseif layer.kind == "tanh"
            @NLconstraint(BL.Upper(model),[i in eachindex(next)],next[i] == tanh(previous[i]))
        elseif layer.kind == "sigmoid"
            @NLconstraint(BL.Upper(model),[i in eachindex(next)],next[i] == 1/(1+exp(-previous[i])))
        else
            @NLconstraint(BL.Upper(model),[i in eachindex(next)],next[i] == log(1+exp(previous[i])))
        end
        previous = next
    end
    @variable(BL.Upper(model),raw[1:3])
    @constraint(BL.Upper(model),[i in 1:3],raw[i] == Float64(cfg.output_scale[i])*previous[i])
    return raw
end

function build_bilevel(directory;start=(391.41,1.0111),nonnegative=true)
    cfg = load_config(directory)
    cfg.method == "mass" || error("The bilevel follower implements the trained mass-only affine KKT-hPINN.")
    model = BL.BilevelModel(Ipopt.Optimizer;mode=BL.ProductMode(1e-8))
    @variable(BL.Upper(model),280 <= t <= 460,start=start[1])
    @variable(BL.Upper(model),0.8 <= a <= 1.2,start=start[2])
    raw = leader_backbone(model,t,a,directory,cfg,start)
    expected = to_vector(BRIDGE.evaluate(directory,collect(start))["projected"])
    @variable(BL.Lower(model),y[i in 1:3],start=expected[i])
    @constraint(BL.Lower(model),sum(y) == a+2)
    @objective(BL.Lower(model),Min,sum((y[i]-raw[i])^2 for i in 1:3)/2)
    if nonnegative
        @constraint(BL.Upper(model),[i in 1:3],y[i] >= 0)
    end
    @objective(BL.Upper(model),Max,0.7*a-y[1]-0.5*((t-370)/90)^2)
    return (;model,t,a,y,raw,cfg)
end
