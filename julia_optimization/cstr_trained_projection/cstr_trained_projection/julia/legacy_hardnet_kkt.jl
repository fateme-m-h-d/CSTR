# MathOptAI backbone (.pt) + direct KKT projection in scaled coordinates.
using JuMP, LinearAlgebra, DelimitedFiles
import Ipopt, MathOptAI, JSON3, PythonCall
const LEGACY_ROOT=dirname(@__DIR__)

function legacy_raw(directory,x,cfg)
    h=x ./ Float64.(cfg.input_scale);values=Float64[]
    for layer in JSON3.read(read(joinpath(directory,"layers.json"),String))
        if layer.kind=="linear"
            W=[Float64(layer.weight[i][j]) for i in eachindex(layer.weight),j in eachindex(layer.weight[1])]
            h=W*h+Float64.(layer.bias)
        elseif layer.kind=="relu"
            h=max.(h,0.)
        else
            error("Unknown exported activation")
        end
        append!(values,h)
    end
    return h,values
end

function legacy_physics(t,a,y,cfg)
    f=1e13*exp(-90000/(8.314*t));r=1e11*exp(-Float64(cfg.ear)/(8.314*t))
    g=[a-y[1]-10*f*y[1]*y[2]^2+10*r*y[3],a+2-sum(y)]
    J=[-1-10*f*y[2]^2 -20*f*y[1]*y[2] 10*r; -1. -1. -1.]
    return g,J
end

function build_legacy_kkt(directory;backbone="full",start=(391.41,1.0111),y_start=nothing,nonnegative=true)
    cfg=JSON3.read(read(joinpath(directory,"config.json"),String))
    cfg.method=="hardnet_legacy_package" || error("Use an exported old-package checkpoint")
    checks=JSON3.read(read(joinpath(directory,"export_checks.json"),String))
    checks.all_converged || error("Package projections failed the export checks; do not assume KKT parity")
    Float64(cfg.ear)==100000. || error("Physical parameters differ from the optimization case")
    backbone in ("full","reduced","gray") || error("Choose a backbone embedding: full, reduced, gray")
    model=Model(Ipopt.Optimizer)
    set_optimizer_attribute(model,"tol",1e-8)
    set_optimizer_attribute(model,"constr_viol_tol",1e-8)
    set_optimizer_attribute(model,"max_iter",3000)
    @variable(model,280<=t<=460,start=start[1])
    @variable(model,.8<=a<=1.2,start=start[2])
    raw_start,layer_start=legacy_raw(directory,collect(start),cfg)
    if backbone=="gray"
        # MathOptAI's gray-box probe uses torch's default dtype. Preserve float64.
        torch=PythonCall.pyimport("torch")
        torch.set_default_dtype(torch.float64)
    end
    predictor=MathOptAI.PytorchModel(joinpath(directory,"backbone.pt"))
    before=Set(all_variables(model))
    raw,form=MathOptAI.add_predictor(model,predictor,
        [t/Float64(cfg.input_scale[1]),a/Float64(cfg.input_scale[2])];
        reduced_space=(backbone=="reduced"),gray_box=(backbone=="gray"))
    added=[v for v in all_variables(model) if !(v in before)]
    if length(added)==length(layer_start)
        set_start_value.(added,layer_start)
    elseif backbone=="gray"
        set_start_value.(raw,raw_start)
    end
    s=Float64.(cfg.output_scale)
    guess=y_start===nothing ? raw_start : Float64.(y_start)./s
    _,Jn=legacy_physics(start[1],start[2],guess.*s,cfg)
    Js=Jn*Diagonal(s)
    multiplier_guess=transpose(Js)\(raw_start-guess)
    @variable(model,w[i=1:3],start=guess[i])
    @variable(model,lambda[i=1:2],start=multiplier_guess[i])
    y=[s[i]*w[i] for i in 1:3]
    g,J=legacy_physics(t,a,y,cfg)
    stationary=[w[i]-raw[i]+s[i]*sum(J[j,i]*lambda[j] for j in 1:2) for i in 1:3]
    @constraint(model,[i=1:2],g[i]==0)
    @constraint(model,[i=1:3],stationary[i]==0)
    if nonnegative
        @constraint(model,[i=1:3],w[i]>=0)
    end
    @objective(model,Max,.7*a-y[1]-.5*((t-370)/90)^2)
    println("Backbone = ",backbone,"; projection = direct scaled-coordinate KKT; variables = ",num_variables(model))
    return (;model,t,a,w,y,lambda,raw,g,stationary,cfg)
end

function legacy_report(r)
    (;model,t,a,y,g,stationary)=r
    println("Termination = ",termination_status(model))
    has_values(model) || return false
    residual=norm([value.(stationary);value.(g)])
    println("T = ",value(t),"; Cao = ",value(a),"; concentrations = ",value.(y))
    println("Objective = ",objective_value(model),"; full KKT L2 residual = ",residual)
    println("Extra B balance residual = ",2-value(y[2])-2*(value(a)-value(y[1])))
    return termination_status(model) in (MOI.LOCALLY_SOLVED,MOI.OPTIMAL) && residual<=1e-7
end
