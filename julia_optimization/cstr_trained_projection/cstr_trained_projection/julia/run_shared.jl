# Standalone shared-expression optimization; no Python needed.
using JuMP, LinearAlgebra, DelimitedFiles
import MathOptAI, JSON3, Ipopt
include("shared_reduced.jl")

function run_shared(directory)
    points = readdlm(joinpath(directory,"parity.csv"),',',Float64;skipstart=1)
    result = build_shared_reduced(directory)
    (;model,t,a,y,cfg) = result
    println("FIXED-INPUT CHECK: solve starts")
    hold_t = @constraint(model,t == 391.41)
    hold_a = @constraint(model,a == 1.0111)
    optimize!(model)
    termination_status(model) in (JuMP.MOI.OPTIMAL,JuMP.MOI.LOCALLY_SOLVED) || error("Fixed solve failed")
    # Independently evaluate all saved parity points before free optimization.
    evaluator = JuMP.NLPEvaluator(model)
    JuMP.MOI.initialize(evaluator,[:Jac])
    actual = zeros(3)
    for row in eachrow(points)
        JuMP.MOI.eval_constraint(evaluator,actual,collect(row[1:2]))
        norm(actual-row[6:8],Inf) <= 1e-6 || error("Parity failed")
    end
    selected = value.([t,a])
    delete(model,hold_t);delete(model,hold_a)
    set_start_value.([t,a],selected)
    println("FREE-INPUT OPTIMIZATION: solve starts")
    optimize!(model)
    println("Termination = ",termination_status(model))
    if has_values(model)
        println("T = ",value(t),"; CA0 = ",value(a),"; concentrations = ",value.(y))
        println("Embedded objective = ",objective_value(model))
        println("Run julia/run.jl enforce reduced_shared for the independent Python/physics report.")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_shared(abspath(get(ARGS,1,joinpath(@__DIR__,"..","artifacts","enforce"))))
end
