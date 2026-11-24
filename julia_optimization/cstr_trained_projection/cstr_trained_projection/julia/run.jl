include("common.jl")
if get(ARGS,2,"") == "bilevel"
    include("bilevel.jl")
end

function run_case(method, mode, artifact_root)
    directory = abspath(joinpath(artifact_root,method))
    isfile(joinpath(directory,"config.json")) || error("Train first: python python/train.py --method $method")
    mode in ("full","reduced","gray","bilevel","reduced_shared") ||
        error("Choose full, reduced, reduced_shared, gray, or bilevel")
    result = if mode == "bilevel"
        build_bilevel(directory)
    else
        build_standard(directory,mode)
    end
    (;model,t,a,y) = result

    # Verify feasibility and forward equality with the SAME input held fixed.
    constraint_owner = mode == "bilevel" ? BL.Upper(model) : model
    hold_t = @constraint(constraint_owner,t == 391.41)
    hold_a = @constraint(constraint_owner,a == 1.0111)
    optimize!(model)
    println("FIXED-INPUT CHECK")
    primal_status(model) == MOI.FEASIBLE_POINT || error("Fixed-input solve did not reach feasibility.")
    expected = to_vector(BRIDGE.evaluate(directory,[391.41,1.0111])["projected"])
    mismatch = norm(value.(y)-expected,Inf)
    mismatch <= 1e-6 || error("Forward-map mismatch at fixed inputs: $mismatch")
    variables = mode == "bilevel" ? [t; a; result.raw; y] : all_variables(model)
    starts = value.(variables)
    delete(model, hold_t)
    delete(model, hold_a)
    set_start_value.(variables,starts)

    optimize!(model)
    println("FREE-INPUT OPTIMIZATION")
    report(result,directory)
end

if abspath(PROGRAM_FILE) == @__FILE__
    method = get(ARGS,1,"mass")
    mode = get(ARGS,2,"full")
    artifact_root = get(ARGS,3,joinpath(ROOT,"artifacts"))
    run_case(method,mode,artifact_root)
end
