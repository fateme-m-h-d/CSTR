import Pkg
Pkg.activate(@__DIR__)
Pkg.add(["JuMP", "Ipopt", "MathOptAI", "PythonCall", "JSON3", "BilevelJuMP"])
