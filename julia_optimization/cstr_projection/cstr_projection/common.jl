using JuMP, LinearAlgebra
import Ipopt, MathOptAI, PythonCall

const TM = (280.0, 460.0)
const AM = (0.8, 1.2)
const YSCALE = [1.1923875, 1.9897989, 2.85705345]
const MASS = [-1.0, -1.0, -1.0]
const MIDPOINT = (391.0, 1.01)

kf(t) = 1e13 * exp(-90000 / (8.314 * t))
kr(t) = 1e11 * exp(-100000 / (8.314 * t))
function residual(t, a, y)
    return [a - y[1] - 10 * kf(t) * y[1] * y[2]^2 + 10 * kr(t) * y[3],
            a + 2 - y[1] - y[2] - y[3]]
end

function jac_y(t, y)
    row = [-1 - 10 * kf(t) * y[2]^2, -20 * kf(t) * y[1] * y[2], 10 * kr(t)]
    return [i == 1 ? row[j] : MASS[j] for i in 1:2, j in 1:3]
end

function hess_kinetic(t, y, i, j)
    if i == 1 && j == 2 || i == 2 && j == 1
        return -20 * kf(t) * y[2]
    elseif i == 2 && j == 2
        return -20 * kf(t) * y[1]
    end
    return 0.0
end

# Original data-generator branch: B depletion is 2x and C formation is 3x.
function reference_outlets(t, a)
    lo, hi = 0.0, min(a, 1.0)
    for _ in 1:52
        x = (lo + hi) / 2
        y = [a - x, 2 - 2x, 3x]
        if residual(t, a, y)[1] > 0
            hi = x
        else
            lo = x
        end
    end
    x = (lo + hi) / 2
    return [a - x, 2 - 2x, 3x]
end

score(t, a, ca) = 0.7 * a - ca - 0.5 * ((t - 370) / 90)^2

function operating_inputs(model, region, start)
    @variable(model, region.t[1] <= t <= region.t[2], start = start[1])
    @variable(model, region.a[1] <= a <= region.a[2], start = start[2])
    return t, a
end

function raw_network(model, t, a, path)
    predictor = MathOptAI.PytorchModel(path)
    y, _ = MathOptAI.add_predictor(model, predictor, [t / 460, a / 1.2])
    return [YSCALE[i] * y[i] for i in 1:3]
end

function stage(model, n, initial)
    return @variable(model, [i in 1:n], start = initial[i])
end

function set_outer_objective(model, t, a, y)
    @constraint(model, [i in 1:3], y[i] >= 0)
    @objective(model, Max, score(t, a, y[1]))
end

function collect_result(model, t, a, y, raw; diagnostic = () -> 0.0)
    primal_status(model) == JuMP.MOI.NO_SOLUTION && return nothing
    tv, av = value(t), value(a)
    concentrations = value.(y)
    all(isfinite, [tv; av; concentrations]) || return nothing
    return (status = termination_status(model), t = tv, a = av,
        y = concentrations, raw = value.(raw), objective = objective_value(model),
        original_objective = score(tv, av, reference_outlets(tv, av)[1]),
        physics = residual(tv, av, concentrations),
        b_balance = 2 - concentrations[2] - 2 * (av - concentrations[1]),
        method_residual = diagnostic())
end

function ipopt_model()
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "tol", 1e-8)
    set_optimizer_attribute(model, "max_iter", 3000)
    return model
end
