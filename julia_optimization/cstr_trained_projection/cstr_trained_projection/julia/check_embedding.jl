include("common.jl")
if get(ARGS,2,"") == "bilevel"
    include("bilevel.jl")
end

function main(method,mode,artifact_root)
    directory = abspath(joinpath(artifact_root,method))
    points = readdlm(joinpath(directory,"parity.csv"),',',Float64;skipstart=1)
    # Include off-grid and high-temperature points as well as the reference.
    indices = unique([1,clamp(round(Int,size(points,1)/2),1,size(points,1)),size(points,1)-2,
                      size(points,1)-1,size(points,1)])
    for index in indices
        start = (points[index,1],points[index,2])
        result = if mode == "bilevel"
                build_bilevel(directory;start,nonnegative=false)
        else
            build_standard(directory,mode;start,nonnegative=false)
        end
        (;model,t,a,y) = result
        set_silent(model)
        constraint_owner = mode == "bilevel" ? BL.Upper(model) : model
        @constraint(constraint_owner,t == start[1])
        @constraint(constraint_owner,a == start[2])
        optimize!(model)
        primal_status(model) == MOI.FEASIBLE_POINT || error("Failed fixed-input solve at $start")
        difference = norm(value.(y)-points[index,6:8],Inf)
        println(start," max projected output difference = ",difference)
        difference <= 1e-6 || error("Projection embedding does not match the training forward pass")
    end
    println("All selected fixed-input parity checks passed.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(get(ARGS,1,"mass"),get(ARGS,2,"full"),get(ARGS,3,joinpath(ROOT,"artifacts")))
end
