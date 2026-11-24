# Fixed-depth AdaNP ablation. Each QP is centered at the current iterate.
function project_enforce(model, t, a, raw, initial, steps)
    current = raw
    for _ in 1:steps
        next = stage(model, 3, initial)
        dual = stage(model, 2, zeros(2))
        g = residual(t, a, current)
        J = jac_y(t, current)
        @constraint(model, [i in 1:3],
            next[i] - current[i] + sum(J[j,i] * dual[j] for j in 1:2) == 0)
        @constraint(model, [j in 1:2],
            g[j] + sum(J[j,i] * (next[i] - current[i]) for i in 1:3) == 0)
        current = next
    end
    return current
end
