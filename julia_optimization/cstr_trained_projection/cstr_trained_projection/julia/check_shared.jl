# Standalone check: Julia + saved JSON/CSV, no Python required.
using JuMP, LinearAlgebra, DelimitedFiles
import MathOptAI, JSON3, Ipopt
include("shared_reduced.jl")

function native_enforce(x,raw,cfg,steps)
    t,a = x
    kf = 1e13*exp(-90000/(8.314*t))
    kr = 1e11*exp(-Float64(cfg.ear)/(8.314*t))
    y = copy(raw)
    for _ in 1:steps
        ca,cb,cc = y
        g = [a-ca-10*kf*ca*cb^2+10*kr*cc,a+2-ca-cb-cc]
        J = [-1-10*kf*cb^2 -20*kf*ca*cb 10*kr; -1.0 -1.0 -1.0]
        y -= J' * ((J*J') \ g)
    end
    return y
end

function check_shared(directory)
    points = readdlm(joinpath(directory,"parity.csv"),',',Float64;skipstart=1)
    # 1 stage diagnoses construction separately; 40 (saved depth) is benchmark.
    for stages in (1,40)
        result = build_shared_reduced(directory;steps_override=stages)
        (;model,t,a,y,cfg) = result
        @assert stages <= Int(cfg.steps)
        evaluator = JuMP.NLPEvaluator(model)
        MOI = JuMP.MOI
        MOI.initialize(evaluator,[:Grad,:Jac,:Hess])
        actual = zeros(3)
        worst = 0.0
        for row in eachrow(points)
            x = collect(row[1:2])
            MOI.eval_constraint(evaluator,actual,x)
            reference = stages == Int(cfg.steps) ? row[6:8] :
                native_enforce(x,collect(row[3:5]),cfg,stages)
            worst = max(worst,norm(actual-reference,Inf))
        end
        println("Stages = ",stages,"; all-point forward discrepancy = ",worst)
        worst <= 1e-6 || error("Shared forward map fails parity")
        # Check the objective derivatives against centered differences.
        x = [391.41,1.0111]
        gradient = zeros(2)
        MOI.eval_objective_gradient(evaluator,gradient,x)
        h = [1e-3,1e-6]
        fd = zeros(2)
        for j in 1:2
            xp,xm = copy(x),copy(x)
            xp[j] += h[j]; xm[j] -= h[j]
            fd[j] = (MOI.eval_objective(evaluator,xp)-MOI.eval_objective(evaluator,xm))/(2h[j])
        end
        @assert isapprox(gradient,fd;atol=1e-6,rtol=1e-4)
        structure = MOI.hessian_lagrangian_structure(evaluator)
        entries = zeros(length(structure))
        MOI.eval_hessian_lagrangian(evaluator,entries,x,1.0,zeros(3))
        H = zeros(2,2)
        for ((i,j),v) in zip(structure,entries)
            H[i,j] += v
            i == j || (H[j,i] += v)
        end
        Hfd = zeros(2,2)
        for j in 1:2
            xp,xm = copy(x),copy(x);xp[j] += h[j];xm[j] -= h[j]
            gp,gm = zeros(2),zeros(2)
            MOI.eval_objective_gradient(evaluator,gp,xp)
            MOI.eval_objective_gradient(evaluator,gm,xm)
            Hfd[:,j] = (gp-gm)/(2h[j])
        end
        println("Hessian AD = ",H,"; FD = ",Hfd)
        @assert isapprox(H,Hfd;atol=1e-5,rtol=1e-3)
        println("PASS: forward parity, objective gradient/Hessian finite differences, two variables")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    check_shared(abspath(get(ARGS,1,joinpath(@__DIR__,"..","artifacts","enforce"))))
end
