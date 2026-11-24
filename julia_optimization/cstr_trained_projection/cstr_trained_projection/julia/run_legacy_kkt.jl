include("legacy_hardnet_kkt.jl")
include("legacy_complete_gray.jl")
function main()
    directory=abspath(get(ARGS,1,joinpath(LEGACY_ROOT,"artifacts_legacy","hardnet")))
    backbone=get(ARGS,2,"full")
    if backbone=="gray"
        return run_legacy_complete_gray(directory)
    end
    rows=readdlm(joinpath(directory,"parity.csv"),',',Float64;skipstart=1)
    k=argmin((rows[:,1].-391.41).^2 .+(rows[:,2].-1.0111).^2)
    r=build_legacy_kkt(directory;backbone,start=Tuple(rows[k,1:2]),y_start=rows[k,6:8])
    (;model,t,a,y)=r
    ht=@constraint(model,t==rows[k,1]);ha=@constraint(model,a==rows[k,2])
    println("FIXED-INPUT CHECK")
    optimize!(model);legacy_report(r) || error("Fixed-input KKT solve failed")
    difference=norm(value.(y)-rows[k,6:8],Inf)
    println("Fixed-input forward discrepancy = ",difference)
    difference<=1e-6 || error("Different projection root or inaccurate package forward solve")
    starts=value.(all_variables(model));delete(model,ht);delete(model,ha)
    set_start_value.(all_variables(model),starts)
    println("FREE-INPUT OPTIMIZATION; use this second solve's Ipopt time")
    optimize!(model);legacy_report(r) || error("Free-input optimization failed")
    python=get(ENV,"CSTR_LEGACY_PYTHON",get(ENV,"JULIA_PYTHONCALL_EXE","python"))
    checker=joinpath(LEGACY_ROOT,"python","legacy_forward.py")
    expected=JSON3.read(read(`$python $checker $directory $(value(t)) $(value(a))`,String))
    discrepancy=norm(value.(y)-Float64.(expected),Inf)
    println("Package/direct-KKT output discrepancy at selected optimum = ",discrepancy)
    solution=Dict("T"=>value(t),"Cao"=>value(a),"concentrations"=>value.(y),
        "objective"=>objective_value(model),"forward_discrepancy"=>discrepancy,"backbone_embedding"=>backbone,
        "accepted_as_trained_predictor"=>(discrepancy<=1e-6))
    open(joinpath(directory,"direct_kkt_"*backbone*".json"),"w") do io;JSON3.write(io,solution);end
    discrepancy<=1e-6 || error("Outer KKT solution differs from the package projection; do not accept as the same predictor")
end
main()
