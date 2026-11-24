# Native MathOptAI reduced MLP + symbolic projection arithmetic.
# No user-defined nonlinear operators and no Python derivative callbacks.
# Python is still used by common.jl for checkpoint import and parity reporting.

# JuMP/MOI expands shared nonlinear expressions into trees. Count their expanded
# size with memoization before handing an enormous expression to the solver.
function expanded_expression_size(x, memo, limit)
    if x isa JuMP.GenericNonlinearExpr
        haskey(memo,x) && return memo[x]
        count = 1
        for arg in x.args
            count = min(limit+1,count+expanded_expression_size(arg,memo,limit))
            count > limit && break
        end
        memo[x] = count
        return count
    elseif x isa JuMP.GenericAffExpr
        return 1 + 3*length(x.terms)
    elseif x isa JuMP.GenericQuadExpr
        return 1 + 5*length(x.terms) + expanded_expression_size(x.aff,memo,limit)
    end
    return 1
end

function symbolic_budget!(values, label)
    limit = parse(Int,get(ENV,"CSTR_SYMBOLIC_MAX_NODES","2000000"))
    limit > 0 || error("CSTR_SYMBOLIC_MAX_NODES must be positive")
    memo = IdDict{Any,Int}()
    count = 0
    for value in values
        count = min(limit+1,count+expanded_expression_size(value,memo,limit))
        count > limit && error("Symbolic reduced-space expression exceeds $limit expanded nodes at $label. " *
            "Stopped before solver conversion; no callback fallback or change of projection depth was used. " *
            "The saved $(label) computation must be preserved for parity. " *
            "Use full/gray for this checkpoint, or increase CSTR_SYMBOLIC_MAX_NODES only if resources permit.")
    end
    println("Symbolic expression nodes at ",label,": ",count)
    return
end

# Algebraic Gaussian elimination for the small positive-definite systems used
# here. Fixed ordering avoids symbolic comparisons/pivot branches. Do not use
# this for general indefinite systems. No inv(), external solver, or operator.
function symbolic_spd_solve(matrix, rhs)
    n = length(rhs)
    A = Any[matrix[i,j] for i in 1:n,j in 1:n]
    b = Any[rhs[i] for i in 1:n]
    for k in 1:n-1
        for i in k+1:n
            factor = A[i,k]/A[k,k]
            for j in k+1:n
                A[i,j] = A[i,j]-factor*A[k,j]
            end
            b[i] = b[i]-factor*b[k]
        end
    end
    solution = Any[0.0 for _ in 1:n]
    for i in n:-1:1
        solution[i] = (b[i]-sum(A[i,j]*solution[j] for j in i+1:n;init=0.0))/A[i,i]
    end
    return solution
end

function projection_symbolic(t,a,raw,cfg)
    method = String(cfg.method)
    method == "baseline" && return raw
    if method == "mass"
        correction = (a+2-sum(raw))/3
        return [raw[i]+correction for i in 1:3]
    elseif method == "pl"
        B,rhs = affine_system(t,a,cfg)
        gram = [sum(B[i,k]*B[j,k] for k in 1:3) for i in 1:2,j in 1:2]
        residual = [sum(B[i,k]*raw[k] for k in 1:3)-rhs[i] for i in 1:2]
        dual = symbolic_spd_solve(gram,residual)
        return [raw[i]-sum(B[j,i]*dual[j] for j in 1:2) for i in 1:3]
    end
    method in ("enforce","hardnet") || error("Unknown projection: $method")
    current,dual = raw,Any[0.0,0.0]
    for stage in 1:Int(cfg.steps)
        g,J,Ha = physics(t,a,current,cfg)
        if method == "enforce"
            gram = [sum(J[i,k]*J[j,k] for k in 1:3) for i in 1:2,j in 1:2]
            correction = symbolic_spd_solve(gram,g)
            current = [current[i]-sum(J[j,i]*correction[j] for j in 1:2) for i in 1:3]
        else
            H = [(i == j ? 1.0 : 0.0)+dual[1]*Ha[i,j] for i in 1:3,j in 1:3]
            K = [H transpose(J);J zeros(2,2)]
            stationary = [current[i]-raw[i]+sum(J[j,i]*dual[j] for j in 1:2) for i in 1:3]
            F = [stationary;g]
            normal = [sum(K[k,i]*K[k,j] for k in 1:5)+
                (i == j ? Float64(cfg.regularization) : 0.0) for i in 1:5,j in 1:5]
            rhs = [-sum(K[k,i]*F[k] for k in 1:5) for i in 1:5]
            delta = symbolic_spd_solve(normal,rhs)
            current = [current[i]+Float64(cfg.damping)*delta[i] for i in 1:3]
            dual = [dual[j]+Float64(cfg.damping)*delta[j+3] for j in 1:2]
        end
        symbolic_budget!([current;dual],"$method stage $stage")
    end
    return current
end

function complete_symbolic_reduced(model,t,a,directory,cfg)
    original_variables = num_variables(model)
    original_constraints = num_constraints(model;count_variable_in_set_constraints=true)
    predictor = MathOptAI.PytorchModel(joinpath(directory,"backbone.pt"))
    inputs = [t/Float64(cfg.input_scale[1]),a/Float64(cfg.input_scale[2])]
    scaled,_ = MathOptAI.add_predictor(model,predictor,inputs;reduced_space=true)
    raw = [Float64(cfg.output_scale[i])*scaled[i] for i in 1:3]
    symbolic_budget!(raw,"MLP")
    y = projection_symbolic(t,a,raw,cfg)
    symbolic_budget!(y,"complete predictor")
    num_variables(model) == original_variables || error("Reduced embedding added variables")
    num_constraints(model;count_variable_in_set_constraints=true) == original_constraints ||
        error("Reduced embedding added linking constraints")
    println("Native reduced MLP + symbolic projection; no projection callbacks")
    return y
end
