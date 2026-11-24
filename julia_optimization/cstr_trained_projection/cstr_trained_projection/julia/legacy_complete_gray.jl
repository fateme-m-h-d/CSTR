# Complete MLP + PACKAGE projection, using MathOptAI's vector gray-box interface.
# No projection outputs/multipliers/intermediate stages inside the outer model.
if !isdefined(@__MODULE__,:LEGACY_ROOT)
    include("legacy_hardnet_kkt.jl")
end

struct LegacyCompletePredictor <: MathOptAI.AbstractPredictor
    directory::String
end

function legacy_complete_bridge(directory)
    sys=PythonCall.pyimport("sys")
    path=joinpath(LEGACY_ROOT,"python")
    sys.path.insert(0,path)
    return PythonCall.pyimport("legacy_complete_gray").load(abspath(directory))
end

function MOI.VectorNonlinearOracle(p::MathOptAI.GrayBox{LegacyCompletePredictor},n::Int)
    n==2 || error("The complete CSTR predictor takes T and Cao")
    bridge=legacy_complete_bridge(p.predictor.directory)
    vec(v)=PythonCall.pyconvert(Vector{Float64},v)
    function eval_f(ret,x)
        f=vec(bridge.value(collect(x[1:2])))
        for i in 1:3;ret[i]=f[i]-x[2+i];end
        return
    end
    structure=[(i,j) for j in 1:2 for i in 1:3]
    append!(structure,[(i,2+i) for i in 1:3])
    function eval_jacobian(ret,x)
        J=vec(bridge.jacobian_flat(collect(x[1:2])))
        ret[1:6].=J
        ret[7:9].=-1.
        return
    end
    function eval_hessian(ret,x,mu)
        ret.=vec(bridge.weighted_hessian(collect(x[1:2]),collect(mu)))
        return
    end
    return MOI.VectorNonlinearOracle(;
        dimension=5,l=zeros(3),u=zeros(3),eval_f,
        jacobian_structure=structure,eval_jacobian,
        hessian_lagrangian_structure=[(1,1),(2,1),(2,2)],
        eval_hessian_lagrangian=eval_hessian)
end

function MathOptAI.add_predictor(model::JuMP.AbstractModel,p::LegacyCompletePredictor,x::Vector;
        gray_box::Bool=true,reduced_space::Bool=false,kwargs...)
    gray_box && !reduced_space || error("This predictor implements complete gray box only")
    return MathOptAI.add_predictor(model,MathOptAI.GrayBox(p;hessian=true),x)
end

function build_legacy_complete_gray(directory;start=(391.41,1.0111))
    cfg=JSON3.read(read(joinpath(directory,"config.json"),String))
    bridge=legacy_complete_bridge(directory)
    initial=PythonCall.pyconvert(Vector{Float64},bridge.value(collect(start)))
    model=Model(Ipopt.Optimizer)
    set_optimizer_attribute(model,"tol",1e-8)
    set_optimizer_attribute(model,"max_iter",3000)
    @variable(model,280<=t<=460,start=start[1])
    @variable(model,.8<=a<=1.2,start=start[2])
    y,form=MathOptAI.add_predictor(model,LegacyCompletePredictor(abspath(directory)),[t,a];gray_box=true)
    set_start_value.(y,initial)
    @constraint(model,[i=1:3],y[i]>=0)
    @objective(model,Max,.7*a-y[1]-.5*((t-370)/90)^2)
    num_variables(model)==5 || error("Complete gray box must have two inputs and three outputs")
    println("COMPLETE gray box: MLP + package projection; variables = ",num_variables(model),"; no outer KKT equations")
    return (;model,t,a,y,cfg,bridge,form)
end

function run_legacy_complete_gray(directory)
    r=build_legacy_complete_gray(directory)
    (;model,t,a,y)=r
    ht=@constraint(model,t==391.41);ha=@constraint(model,a==1.0111)
    println("FIXED-INPUT COMPLETE GRAY CHECK")
    optimize!(model)
    termination_status(model) in (MOI.LOCALLY_SOLVED,MOI.OPTIMAL) || error("Fixed-input solve failed")
    function mismatch()
        expected=PythonCall.pyconvert(Vector{Float64},r.bridge.independent_value([value(t),value(a)]))
        return norm(value.(y)-expected,Inf)
    end
    println("Fixed-input independent package discrepancy = ",mismatch())
    mismatch()<=1e-6 || error("Complete-predictor forward mismatch")
    starts=value.(all_variables(model));delete(model,ht);delete(model,ha)
    set_start_value.(all_variables(model),starts)
    println("FREE-INPUT COMPLETE GRAY OPTIMIZATION (second solve time)")
    optimize!(model)
    println("Termination = ",termination_status(model))
    has_values(model) || error("No gray-box solution")
    discrepancy=mismatch()
    g,_=legacy_physics(value(t),value(a),value.(y),r.cfg)
    accepted=termination_status(model) in (MOI.LOCALLY_SOLVED,MOI.OPTIMAL) && discrepancy<=1e-6 &&
        minimum(value.(y))>=-1e-7 && norm(g,Inf)<=1e-6
    println("T = ",value(t),"; Cao = ",value(a),"; concentrations = ",value.(y))
    println("Objective = ",objective_value(model))
    println("Independent package/gray output discrepancy = ",discrepancy)
    println("Declared physical residuals = ",g)
    println("Extra B balance residual = ",2-value(y[2])-2*(value(a)-value(y[1])))
    println("Accepted = ",accepted)
    summary=Dict("embedding"=>"complete_gray","variables"=>num_variables(model),
        "T"=>value(t),"Cao"=>value(a),"concentrations"=>value.(y),
        "objective"=>objective_value(model),"forward_discrepancy"=>discrepancy,"accepted"=>accepted)
    open(joinpath(directory,"complete_gray_solution.json"),"w") do io;JSON3.write(io,summary);end
    accepted || error("Complete gray solution failed validation")
    return r
end
