# Shared symbolic reduced ENFORCE. Legacy nonlinear subexpressions are computed
# values, not decision variables. No registered operators/Python AD callbacks.

# Replay the exported trained layers using native MathOptAI predictors. This
# uses exactly the saved weights and avoids requiring PyTorch for this builder.
function exported_mlp_predictor(directory)
    layers = JSON3.read(read(joinpath(directory,"layers.json"),String))
    predictors = MathOptAI.AbstractPredictor[]
    for layer in layers
        kind = String(layer.kind)
        if kind == "linear"
            A = [Float64(layer.weight[i][j]) for i in eachindex(layer.weight),
                 j in eachindex(layer.weight[1])]
            push!(predictors,MathOptAI.Affine(A,Float64.(layer.bias)))
        elseif kind == "tanh"
            push!(predictors,MathOptAI.Tanh())
        elseif kind == "sigmoid"
            push!(predictors,MathOptAI.Sigmoid())
        elseif kind == "softplus"
            push!(predictors,MathOptAI.SoftPlus())
        else
            error("Unsupported exported layer: $kind")
        end
    end
    return MathOptAI.Pipeline(predictors)
end

# Explicitly translate modern expressions into a legacy subexpression DAG.
# Memoization preserves shared object references; every nonlinear operation
# becomes a stored node whose arguments reference already stored expressions.
function store_shared_expression(model,x,memo)
    x isa Number && return x
    x isa JuMP.VariableRef && return x
    x isa JuMP.NonlinearExpression && return x
    haskey(memo,x) && return memo[x]
    expression = if x isa JuMP.GenericNonlinearExpr
        Expr(:call,x.head,[store_shared_expression(model,arg,memo) for arg in x.args]...)
    elseif x isa JuMP.GenericAffExpr
        terms = Any[Float64(JuMP.constant(x))]
        append!(terms,[Expr(:call,:*,Float64(c),v) for (c,v) in JuMP.linear_terms(x)])
        Expr(:call,:+,terms...)
    elseif x isa JuMP.GenericQuadExpr
        terms = Any[store_shared_expression(model,x.aff,memo)]
        append!(terms,[Expr(:call,:*,Float64(c),v,w) for (c,v,w) in JuMP.quad_terms(x)])
        Expr(:call,:+,terms...)
    else
        error("Cannot translate symbolic object $(typeof(x))")
    end
    result = JuMP.add_nonlinear_expression(model,expression)
    memo[x] = result
    return result
end

function enforce_shared_projection(model,t,a,raw,cfg;steps=Int(cfg.steps))
    y = collect(raw)
    ear = Float64(cfg.ear)
    kf = @NLexpression(model,1e13*exp(-90000/(8.314*t)))
    kr = @NLexpression(model,1e11*exp(-ear/(8.314*t)))
    for stage in 1:steps
        ca,cb,cc = y
        g1 = @NLexpression(model,a-ca-10*kf*ca*cb^2+10*kr*cc)
        g2 = @NLexpression(model,a+2-ca-cb-cc)
        j1 = @NLexpression(model,-1-10*kf*cb^2)
        j2 = @NLexpression(model,-20*kf*ca*cb)
        j3 = @NLexpression(model,10*kr)
        a11 = @NLexpression(model,j1^2+j2^2+j3^2)
        a12 = @NLexpression(model,-j1-j2-j3)
        # Same fixed-order elimination as symbolic_reduced.jl, now with shared
        # subexpressions. No extra regularization, clipping, or damping.
        ratio = @NLexpression(model,a12/a11)
        pivot = @NLexpression(model,3-ratio*a12)
        d2 = @NLexpression(model,(g2-ratio*g1)/pivot)
        d1 = @NLexpression(model,(g1-a12*d2)/a11)
        next_ca = @NLexpression(model,ca-j1*d1+d2)
        next_cb = @NLexpression(model,cb-j2*d1+d2)
        next_cc = @NLexpression(model,cc-j3*d1+d2)
        y = [next_ca,next_cb,next_cc]
    end
    return y
end

function build_shared_reduced(directory;start=(391.41,1.0111),nonnegative=true,
                              steps_override=nothing)
    cfg = JSON3.read(read(joinpath(directory,"config.json"),String))
    String(cfg.method) == "enforce" || error("reduced_shared currently supports ENFORCE only")
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model,"tol",1e-8)
    set_optimizer_attribute(model,"max_iter",3000)
    @variable(model,280 <= t <= 460,start=start[1])
    @variable(model,0.8 <= a <= 1.2,start=start[2])
    scaled,_ = MathOptAI.add_predictor(model,exported_mlp_predictor(directory),
        [t/Float64(cfg.input_scale[1]),a/Float64(cfg.input_scale[2])];reduced_space=true)
    memo = IdDict{Any,Any}()
    raw = [store_shared_expression(model,Float64(cfg.output_scale[i])*scaled[i],memo) for i in 1:3]
    steps = isnothing(steps_override) ? Int(cfg.steps) : Int(steps_override)
    0 <= steps <= Int(cfg.steps) || error("Invalid diagnostic stage count")
    y = enforce_shared_projection(model,t,a,raw,cfg;steps)
    @assert num_variables(model) == 2
    @assert num_constraints(model;count_variable_in_set_constraints=false) == 0
    if nonnegative
        @NLconstraint(model,[i in 1:3],y[i] >= 0)
    end
    @NLobjective(model,Max,0.7*a-y[1]-0.5*((t-370)/90)^2)
    println("Complete-model embedding = reduced_shared; decision variables = ",num_variables(model),
        "; saved projection stages = ",steps)
    println("Stored nonlinear subexpressions = ",length(JuMP.nonlinear_model(model).expressions))
    return (;model,t,a,y,raw,cfg)
end
