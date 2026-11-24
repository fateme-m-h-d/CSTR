# Physical-coordinate, equality-only Newton-KKT variant.
# Every step remains anchored to the ORIGINAL raw prediction in stationarity.

function project_hardnet_kkt(model, t, a, raw, guess)
    y = stage(model, 3, guess.y)
    lambda = stage(model, 2, guess.lambda)

    g = residual(t, a, y)
    J = jac_y(t, y)

    @constraint(model, [j in 1:2], g[j] == 0)
    @constraint(model, [i in 1:3],
        y[i] - raw[i] + sum(J[j, i] * lambda[j] for j in 1:2) == 0
    )

    function kkt_error()
        tv, av = value(t), value(a)
        yv, rawv, lv = value.(y), value.(raw), value.(lambda)
        stationarity = yv - rawv + jac_y(tv, yv)' * lv
        return norm([stationarity; residual(tv, av, yv)], Inf)
    end

    return y, kkt_error
end

function hardnet_start_trajectory(path, t, a, steps, damping)
    torch = PythonCall.pyimport("torch")
    network = torch.load(path; map_location = "cpu", weights_only = false)
    network = network.double().eval()

    input = torch.tensor([[t / 460, a / 1.2]]; dtype = torch.float64)
    scaled = PythonCall.pyconvert(
        Vector{Float64}, network(input).detach().reshape(-1).tolist()
    )
    raw_start = YSCALE .* scaled

    y = copy(raw_start)
    lambda = zeros(2)
    identity3 = Matrix{Float64}(I, 3, 3)
    trajectory = []

    for _ in 1:steps
        J = jac_y(t, y)
        H = identity3 + lambda[1] *
            [hess_kinetic(t, y, i, j) for i in 1:3, j in 1:3]
        F = [y - raw_start + J' * lambda; residual(t, a, y)]
        delta = [H J'; J zeros(2, 2)] \ (-F)

        dy, dl = delta[1:3], delta[4:5]
        y = y + damping * dy
        lambda = lambda + damping * dl
        push!(trajectory, (dy = dy, dl = dl, y = y, lambda = lambda))
    end

    return trajectory
end

function project_hardnet(model, t, a, raw, trajectory; damping = 1.0)
    current = raw
    dual = zeros(2)
    for guess in trajectory
        dy = stage(model, 3, guess.dy)
        dl = stage(model, 2, guess.dl)
        next = stage(model, 3, guess.y)
        next_dual = stage(model, 2, guess.lambda)
        g = residual(t, a, current)
        J = jac_y(t, current)
        H = [(i == j ? 1.0 : 0.0) + dual[1] * hess_kinetic(t, current, i, j)
             for i in 1:3, j in 1:3]
        @constraint(model, [i in 1:3],
            current[i] - raw[i] + sum(J[j,i] * dual[j] for j in 1:2) +
            sum(H[i,j] * dy[j] for j in 1:3) +
            sum(J[j,i] * dl[j] for j in 1:2) == 0)
        @constraint(model, [j in 1:2],
            g[j] + sum(J[j,i] * dy[i] for i in 1:3) == 0)
        @constraint(model, [i in 1:3], next[i] == current[i] + damping * dy[i])
        @constraint(model, [j in 1:2], next_dual[j] == dual[j] + damping * dl[j])
        current, dual = next, next_dual
    end
    function kkt_error()
        tv, av, yv = value(t), value(a), value.(current)
        stationarity = yv - value.(raw) + jac_y(tv, yv)' * value.(dual)
        return norm([stationarity; residual(tv, av, yv)], Inf)
    end
    return current, kkt_error
end
