include("common.jl")
include("methods/affine.jl")
include("methods/enforce.jl")
include("methods/hardnet.jl")

function solve_single(method, region, start, path, steps, nt, na)
    model = ipopt_model()
    if method == "hardnet"
        start = (391.41, 1.0111)
    end
    t, a = operating_inputs(model, region, start)
    initial = reference_outlets(start...)
    diagnostic = () -> 0.0
    if method == "exact"
        y = stage(model, 3, initial)
        g = residual(t, a, y)
        @constraint(model, [i in 1:2], g[i] == 0)
        @constraint(model, 2 - y[2] == 2 * (a - y[1]))
        raw = y
    else
        raw = raw_network(model, t, a, path)
        if method == "baseline"
            y = raw
        elseif method == "pl"
            coefficients = region_coefficients(region)
            y = project_affine(model, t, a, raw, coefficients)
            diagnostic = () -> affine_error(t, a, y, coefficients)
        elseif method == "pl-smooth"
            y, diagnostic = project_smooth(model, t, a, raw, initial, nt, na)
        elseif method == "enforce"
            y = project_enforce(model, t, a, raw, initial, steps)
            diagnostic = () -> norm(residual(value(t), value(a), value.(y)), Inf)
        elseif method == "hardnet"
            damping = 0.5
            trajectory = hardnet_start_trajectory(
                path, start[1], start[2], steps, damping,
            )
            y, diagnostic = project_hardnet_kkt(model, t, a, raw, last(trajectory))
            unset_silent(model)
        else
            error("Unknown method: $method")
        end
    end
    set_outer_objective(model, t, a, y)

    if method == "hardnet"
        hold_t = @constraint(model, t == start[1])
        hold_a = @constraint(model, a == start[2])

        optimize!(model)
        fixed_status = termination_status(model)
        println("Fixed-input HardNet exit: ", raw_status(model))

        if !(fixed_status in (JuMP.MOI.LOCALLY_SOLVED, JuMP.MOI.OPTIMAL)) ||
        primal_status(model) != JuMP.MOI.FEASIBLE_POINT
            return nothing
        end

        fixed_kkt = diagnostic()
        println("Fixed-input KKT residual: ", fixed_kkt)
        if !isfinite(fixed_kkt) || fixed_kkt > 1e-6
            return nothing
        end

        # Save values before changing the model.
        variables = all_variables(model)
        starts = value.(variables)

        delete(model, hold_t)
        delete(model, hold_a)
        set_start_value.(variables, starts)
    end



    optimize!(model)
    if method == "hardnet"
        println("Ipopt exit: ", raw_status(model))
    end
    return collect_result(model, t, a, y, raw; diagnostic)
end

function run_case(method, activation, nt, na, steps)
    method in ("exact", "baseline", "bilevel", "pl", "pl-smooth", "enforce", "hardnet") ||
        error("Choose exact, baseline, bilevel, pl, pl-smooth, enforce, or hardnet")
    activation in ("tanh", "softplus", "sigmoid", "relu") || error("Unknown activation")
    method == "bilevel" && activation != "tanh" && error("Use tanh for bilevel")
    min(nt, na, steps) >= 1 || error("Grid sizes and step count must be positive")
    path = joinpath(@__DIR__, "checkpoints", "cstr_nn_$activation.pt")
    regional = method in ("pl", "bilevel")
    ts, feeds = range(TM...; length = (regional ? nt : 1) + 1),
             range(AM...; length = (regional ? na : 1) + 1)

    best = nothing
    for i in 1:(length(ts)-1), j in 1:(length(feeds)-1)
        box = (t = (ts[i], ts[i+1]), a = (feeds[j], feeds[j+1]))
        start = regional ? (sum(box.t) / 2, sum(box.a) / 2) : MIDPOINT
        result = if method == "bilevel"
            # solve_bilevel(box, start, path, region_coefficients(box))
            mass_only = (A = [0.0 1.0], B = [-1.0 -1.0 -1.0], b = [-2.0])
            solve_bilevel(box, start, path, mass_only)
        else
            solve_single(method, box, start, path, steps, nt, na)
        end
        println("Region ", (i, j), " → ", result)
        if isnothing(result) ||
           !(result.status in (JuMP.MOI.LOCALLY_SOLVED, JuMP.MOI.OPTIMAL)) ||
           minimum(result.y) < -1e-7 || result.method_residual > 1e-6
            println("Excluded: solver status or projection check failed.")
            continue
        end
        if isnothing(best) || result.objective > best.objective
            best = result
        end
    end
    println("Best accepted candidate for ", method, " → ", best)
    return best
end

function main(args)
    method = get(args, 1, "baseline")
    activation = get(args, 2, "tanh")
    nt = parse(Int, get(args, 3, "4"))
    na = parse(Int, get(args, 4, "3"))
    steps = parse(Int, get(args, 5, "6"))
    return run_case(method, activation, nt, na, steps)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if get(ARGS, 1, "") == "bilevel"
        include("methods/bilevel.jl")
    end
    main(ARGS)
end
