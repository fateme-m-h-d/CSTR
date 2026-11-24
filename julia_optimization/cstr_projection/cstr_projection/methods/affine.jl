# Joint Taylor expansion in (T, CA0, CA, CB, CC) at a feasible grid anchor.
function region_coefficients(region)
    t, a = sum(region.t) / 2, sum(region.a) / 2
    y = reference_outlets(t, a)
    dt = (-10 * kf(t) * 90000 * y[1] * y[2]^2 +
           10 * kr(t) * 100000 * y[3]) / (8.314 * t^2)
    B = jac_y(t, y)
    A = [dt 1.0; 0.0 1.0]
    b = A * [t, a] + B * y - residual(t, a, y)
    return (; A, B, b)
end

function project_affine(model, t, a, raw, coefficients)
    (; A, B, b) = coefficients
    gain = (B * B') \ B
    P = Matrix{Float64}(I, 3, 3) - gain' * B
    D = -gain' * A
    offset = gain' * b
    return @expression(model, [i in 1:3],
        sum(P[i,j] * raw[j] for j in 1:3) + D[i,1] * t + D[i,2] * a + offset[i])
end

function affine_error(t, a, y, coefficients)
    (; A, B, b) = coefficients
    return norm(A * [value(t), value(a)] + B * value.(y) - b, Inf)
end

function sigmoid_weights(x, edges, width)
    cuts = [1 / (1 + exp(-(x - edge) / width)) for edge in edges[2:end-1]]
    cumulative = [1.0; cuts; 0.0]
    return [cumulative[i] - cumulative[i+1] for i in 1:(length(edges)-1)]
end

# Smooth comparison variant: project onto one blended constraint system.
function project_smooth(model, t, a, raw, initial, nt, na)
    ts = collect(range(TM...; length = nt + 1))
    feeds = collect(range(AM...; length = na + 1))
    wt = sigmoid_weights(t, ts, (ts[2] - ts[1]) / 8)
    wa = sigmoid_weights(a, feeds, (feeds[2] - feeds[1]) / 8)
    coeff = [region_coefficients((t = (ts[i], ts[i+1]),
                                  a = (feeds[j], feeds[j+1])))
             for i in 1:nt, j in 1:na]
    w = [wt[i] * wa[j] for i in 1:nt, j in 1:na]
    row = [sum(w[i,j] * coeff[i,j].B[1,k] for i in 1:nt, j in 1:na)
           for k in 1:3]
    rhs = sum(w[i,j] * (coeff[i,j].b[1] - coeff[i,j].A[1,1] * t - a)
              for i in 1:nt, j in 1:na)
    y = stage(model, 3, initial)
    dual = stage(model, 2, zeros(2))
    @constraint(model, [i in 1:3],
        y[i] - raw[i] + row[i] * dual[1] + MASS[i] * dual[2] == 0)
    @constraint(model, sum(row[i] * y[i] for i in 1:3) == rhs)
    @constraint(model, sum(y) == a + 2)
    function blended_error()
        return max(abs(dot(value.(row), value.(y)) - value(rhs)),
                   abs(value(a) + 2 - sum(value.(y))))
    end
    return y, blended_error
end
