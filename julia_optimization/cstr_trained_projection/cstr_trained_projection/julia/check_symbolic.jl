include("common.jl")

# First check the same symbolic projection builder with numerical inputs.
# This tests fixed-order elimination against the saved Python forward pass.
function check_numeric(method,artifact_root)
    directory = abspath(joinpath(artifact_root,method))
    cfg = load_config(directory)
    points = readdlm(joinpath(directory,"parity.csv"),',',Float64;skipstart=1)
    worst = 0.0
    for index in axes(points,1)
        t,a = points[index,1:2]
        raw = to_vector(BRIDGE.evaluate(directory,[t,a])["raw"])
        y = projection_symbolic(t,a,raw,cfg)
        difference = norm(y-points[index,6:8],Inf)
        isfinite(difference) || error("Nonfinite projection at row $index")
        worst = max(worst,difference)
    end
    println(method," numerical projection-builder maximum discrepancy = ",worst)
    worst <= 1e-6 || error("Symbolic algebra does not preserve the numerical forward map")
end

function check_structure(method,artifact_root)
    result = build_standard(abspath(joinpath(artifact_root,method)),"reduced";nonnegative=false)
    @assert num_variables(result.model) == 2
    @assert num_constraints(result.model;count_variable_in_set_constraints=false) == 0
    println("PASS: two operating variables; no projection or MLP linking constraints")
end

if abspath(PROGRAM_FILE) == @__FILE__
    method = get(ARGS,1,"mass")
    root = get(ARGS,3,joinpath(ROOT,"artifacts"))
    check_numeric(method,root)
    get(ARGS,2,"structure") == "numeric" || check_structure(method,root)
end
