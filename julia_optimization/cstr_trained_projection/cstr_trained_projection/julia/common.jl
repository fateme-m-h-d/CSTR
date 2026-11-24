using JuMP, LinearAlgebra, DelimitedFiles
import Ipopt, MathOptAI, PythonCall, JSON3

const ROOT = dirname(@__DIR__)
const PYDIR = joinpath(ROOT, "python")
PythonCall.pyimport("sys").path.insert(0, PYDIR)
const BRIDGE = PythonCall.pyimport("bridge")
const CSTRPY = PythonCall.pyimport("cstr_models")
const TORCH = PythonCall.pyimport("torch")
const MOI = JuMP.MOI

load_config(directory) = JSON3.read(read(joinpath(directory, "config.json"), String))
to_vector(x) = PythonCall.pyconvert(Vector{Float64}, x)
to_dicts(x) = PythonCall.pyconvert(Vector{Dict{String,Any}}, x)
operating_score(t, a, ca) = 0.7*a - ca - 0.5*((t-370)/90)^2

function physics(t, a, y, cfg)
    f = 1e13*exp(-90000/(8.314*t))
    r = 1e11*exp(-Float64(cfg.ear)/(8.314*t))
    g = [a-y[1]-10*f*y[1]*y[2]^2+10*r*y[3], a+2-sum(y)]
    row = [-1-10*f*y[2]^2, -20*f*y[1]*y[2], 10*r]
    J = [i == 1 ? row[j] : -1.0 for i in 1:2, j in 1:3]
    z = 0.0
    cross = -20*f*y[2]
    H = [z cross z; cross -20*f*y[1] z; z z z]
    return g, J, H
end

function auxiliary(model, values)
    @variable(model, [i in eachindex(values)], start = Float64(values[i]))
end

function linear_solve_layer(model, matrix, rhs, initial)
    solution = auxiliary(model, initial)
    @constraint(model, [i in eachindex(rhs)],
        sum(matrix[i,j]*solution[j] for j in eachindex(initial)) == rhs[i])
    return solution
end

function materialize_layer(model, expressions, initial)
    y = auxiliary(model, initial)
    @constraint(model, [i in eachindex(initial)], y[i] == expressions[i])
    return y
end

function weights(x, edges, fraction)
    width = (edges[2]-edges[1])*fraction
    cumulative = [1.0;
        [1/(1+exp(-(x-Float64(edge))/width)) for edge in edges[2:end-1]];
        0.0]
    return [cumulative[i]-cumulative[i+1] for i in 1:(length(edges)-1)]
end

function affine_system(t, a, cfg)
    if cfg.method == "mass"
        return reshape([-1.0, -1.0, -1.0], 1, 3), [-a-2]
    end
    wt = weights(t, Float64.(cfg.temperature_edges), Float64(cfg.gate_width_fraction))
    wa = weights(a, Float64.(cfg.feed_edges), Float64(cfg.gate_width_fraction))
    # Python flattens temperature-major, feed-minor.
    w = [wt[i]*wa[j] for i in eachindex(wt) for j in eachindex(wa)]
    B = [sum(w[r]*Float64(cfg.regions[r].B[i][j]) for r in eachindex(w))
         for i in 1:2, j in 1:3]
    rhs = [sum(w[r]*(Float64(cfg.regions[r].b[i]) -
                     Float64(cfg.regions[r].A[i][1])*t -
                     Float64(cfg.regions[r].A[i][2])*a) for r in eachindex(w))
           for i in 1:2]
    return B, rhs
end

# Expand the backbone with MathOptAI, initializing the supported layer outputs.
function backbone_full(model, t, a, directory, cfg, start)
    predictor = MathOptAI.PytorchModel(joinpath(directory, "backbone.pt"))
    inputs = [t/Float64(cfg.input_scale[1]), a/Float64(cfg.input_scale[2])]
    before = Set(all_variables(model))
    y, form = MathOptAI.add_predictor(model, predictor, inputs)
    added = [v for v in all_variables(model) if !(v in before)]
    layer_values = PythonCall.pyconvert(Vector{Vector{Float64}},
        BRIDGE.numeric_layers(directory,collect(start)))
    numeric = reduce(vcat,layer_values)
    if length(added) == length(numeric) && added[end-2:end] == y
        set_start_value.(added,numeric)
    end
    raw = [Float64(cfg.output_scale[i])*y[i] for i in 1:3]
    return raw
end

function projection_full(model, t, a, raw, directory, cfg, start)
    method = String(cfg.method)
    method == "baseline" && return raw
    trajectory = to_dicts(BRIDGE.stages(directory, collect(start)))
    if method in ("mass", "pl")
        B, rhs = affine_system(t, a, cfg)
        gram = [sum(B[i,k]*B[j,k] for k in 1:3) for i in eachindex(rhs), j in eachindex(rhs)]
        residual = [sum(B[i,k]*raw[k] for k in 1:3)-rhs[i] for i in eachindex(rhs)]
        dual = linear_solve_layer(model,gram,residual,trajectory[1]["dual"])
        corrected = [raw[i]-sum(B[j,i]*dual[j] for j in eachindex(rhs)) for i in 1:3]
        return materialize_layer(model,corrected,trajectory[1]["next"])
    end
    current, dual = raw, zeros(2)
    for guess in trajectory
        g, J, Ha = physics(t, a, current, cfg)
        if method == "enforce"
            gram = [sum(J[i,k]*J[j,k] for k in 1:3) for i in 1:2, j in 1:2]
            d = linear_solve_layer(model,gram,g,guess["dual"])
            corrected = [current[i]-sum(J[j,i]*d[j] for j in 1:2) for i in 1:3]
            current = materialize_layer(model,corrected,guess["next"])
        else
            H = [(i == j ? 1.0 : 0.0)+dual[1]*Ha[i,j] for i in 1:3, j in 1:3]
            K = [H transpose(J); J zeros(2,2)]
            stationary = [current[i]-raw[i]+sum(J[j,i]*dual[j] for j in 1:2) for i in 1:3]
            F = [stationary; g]
            normal = [sum(K[k,i]*K[k,j] for k in 1:5)+
                      (i == j ? Float64(cfg.regularization) : 0.0) for i in 1:5, j in 1:5]
            rhs = [-sum(K[k,i]*F[k] for k in 1:5) for i in 1:5]
            delta = linear_solve_layer(model,normal,rhs,guess["delta"])
            current = materialize_layer(model, current+Float64(cfg.damping)*delta[1:3], guess["next"])
            dual = materialize_layer(model, dual+Float64(cfg.damping)*delta[4:5], guess["dual"])
        end
    end
    return current
end

include("symbolic_reduced.jl")
include("shared_reduced.jl")

# Extend MathOptAI for the entire saved corrected model. The three standard
# keywords now control the SAME complete predictor, including its projection.
struct CorrectedPredictor <: MathOptAI.AbstractPredictor
    directory::String
    cfg::Any
    start::Tuple{Float64,Float64}
end

struct CorrectedFormulation <: MathOptAI.AbstractFormulation
    mode::String
    variables::Vector{JuMP.VariableRef}
end

function MathOptAI.add_predictor(model::JuMP.Model,p::CorrectedPredictor,x::Vector;
                                reduced_space::Bool=false,gray_box::Bool=false,kwargs...)
    reduced_space && gray_box && error("Choose one embedding mode")
    t,a = x
    before = Set(all_variables(model))
    mode = gray_box ? "gray" : reduced_space ? "reduced" : "full"
    if gray_box
        checkpoint = MathOptAI.PytorchModel(joinpath(p.directory,"complete.pt"))
        y, _ = MathOptAI.add_predictor(model,checkpoint,x;gray_box=true)
        expected = to_vector(BRIDGE.evaluate(p.directory,collect(p.start))["projected"])
        set_start_value.(y,expected)
    elseif reduced_space
        y = complete_symbolic_reduced(model,t,a,p.directory,p.cfg)
    else
        raw = backbone_full(model,t,a,p.directory,p.cfg,p.start)
        y = projection_full(model,t,a,raw,p.directory,p.cfg,p.start)
    end
    added = [v for v in all_variables(model) if !(v in before)]
    return y,CorrectedFormulation(mode,added)
end

function build_standard(directory, mode; start=(391.41,1.0111), nonnegative=true)
    mode == "reduced_shared" && return build_shared_reduced(directory;start,nonnegative)
    cfg = load_config(directory)
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "tol", 1e-8)
    set_optimizer_attribute(model, "max_iter", 3000)
    @variable(model, 280 <= t <= 460, start=start[1])
    @variable(model, 0.8 <= a <= 1.2, start=start[2])
    mode in ("full","reduced","gray") || error("Choose full, reduced, or gray")
    predictor = CorrectedPredictor(directory,cfg,start)
    y,_ = MathOptAI.add_predictor(model,predictor,[t,a];
        reduced_space=(mode == "reduced"),gray_box=(mode == "gray"))
    raw = nothing
    println("Complete-model embedding = ",mode,"; number of decision variables = ",num_variables(model))
    if nonnegative
        @constraint(model, [i in 1:3], y[i] >= 0)
    end
    @objective(model, Max, operating_score(t,a,y[1]))
    return (; model,t,a,y,raw,cfg)
end

function report(result, directory)
    (; model,t,a,y,cfg) = result
    println("Termination = ", termination_status(model), "; raw = ", raw_status(model))
    primal_status(model) == MOI.NO_SOLUTION && return false
    tv,av,yv = value(t),value(a),value.(y)
    python_result = BRIDGE.evaluate(directory, [tv,av])
    expected = to_vector(python_result["projected"])
    discrepancy = norm(yv-expected,Inf)
    original_residual, _, _ = physics(tv,av,yv,cfg)
    println("T = ",tv,"; CA0 = ",av,"; concentrations = ",yv)
    println("Embedded objective = ",objective_value(model))
    println("Original-model objective at selected inputs = ",python_result["original_objective"])
    println("Python/Julia projected-output discrepancy = ",discrepancy)
    println("Original A and summed residuals = ",original_residual)
    println("Python terminal projection residual = ",python_result["method_residual"])
    println("Extra B balance residual = ",2-yv[2]-2*(av-yv[1]))
    solver_ok = termination_status(model) in (MOI.OPTIMAL,MOI.LOCALLY_SOLVED)
    projected = String(cfg.method) in ("hardnet","enforce")
    physics_ok = !projected || norm(original_residual,Inf) <= 1e-6
    accepted = solver_ok && discrepancy <= 1e-6 && minimum(yv) >= -1e-7 && physics_ok &&
               PythonCall.pyconvert(Float64,python_result["method_residual"]) <= 1e-6
    println("Accepted under solver, forward-match, nonnegativity and enforced-physics checks = ",accepted)
    return accepted
end
