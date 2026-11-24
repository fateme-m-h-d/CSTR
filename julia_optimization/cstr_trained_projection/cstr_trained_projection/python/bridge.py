"""Callbacks and stage values used by Julia; uses the saved training configuration."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from cstr_models import load_model, physics, jacobian_y, hessian_a, reference_numpy


@lru_cache(maxsize=16)
def get_model(directory):
    return load_model(directory)


def evaluate(directory, values):
    model = get_model(str(directory))
    x = torch.tensor(np.asarray(values, dtype=float))
    with torch.no_grad():
        raw, projected = model.raw(x), model(x)
        errors = physics(x, projected, model.cfg)
    b = 2-projected[1]-2*(x[1]-projected[0])
    original_ca = reference_numpy(x.numpy(), model.cfg)[0, 0]
    def score(ca):
        return float(.7*x[1]-ca-.5*((x[0]-370)/90)**2)
    method_residual = 0.0
    if model.cfg["method"] == "mass":
        method_residual = abs(float(errors[1]))
    elif model.cfg["method"] == "pl":
        with torch.no_grad():
            B, rhs = model.projection.affine_system(x)
            method_residual = float((B@projected-rhs).abs().max())
    elif model.cfg["method"] == "enforce":
        method_residual = float(errors.abs().max())
    elif model.cfg["method"] == "hardnet":
        final = stages(str(directory), values)[-1]
        dual = torch.tensor(final["dual"])
        stationarity = projected-raw+jacobian_y(x, projected, model.cfg).T@dual
        method_residual = float(torch.cat((stationarity, errors)).abs().max())
    return dict(raw=raw.tolist(), projected=projected.tolist(), method_residual=method_residual,
                physics=errors.tolist(), b_balance=float(b),
                objective=score(projected[0]), original_objective=score(original_ca))


def numeric_layers(directory, values):
    model = get_model(str(directory))
    current = torch.tensor(np.asarray(values, dtype=float))/model.input_scale
    outputs = []
    with torch.no_grad():
        for layer in model.backbone:
            current = layer(current)
            outputs.append(current.tolist())
    return outputs


def stages(directory, values):
    """Exact initial values for every unrolled solve and update."""
    model = get_model(str(directory))
    cfg = model.cfg
    x = torch.tensor(np.asarray(values, dtype=float))
    with torch.no_grad():
        raw = model.raw(x)
        if cfg["method"] in ("mass", "pl"):
            B, rhs = model.projection.affine_system(x)
            d = torch.linalg.solve(B@B.T, B@raw-rhs)
            return [dict(dual=d.tolist(), next=(raw-B.T@d).tolist())]
        y, dual, output = raw, torch.zeros(2), []
        for _ in range(cfg["steps"]):
            g, J = physics(x, y, cfg), jacobian_y(x, y, cfg)
            if cfg["method"] == "enforce":
                d = torch.linalg.solve(J@J.T, g)
                y = y-J.T@d
                output.append(dict(dual=d.tolist(), next=y.tolist()))
            else:
                H = torch.eye(3)+dual[0]*hessian_a(x, y, cfg)
                K = torch.cat((torch.cat((H, J.T), -1),
                               torch.cat((J, torch.zeros(2, 2)), -1)), -2)
                F = torch.cat((y-raw+J.T@dual, g))
                delta = torch.linalg.solve(K.T@K+cfg["regularization"]*torch.eye(5), -K.T@F)
                y = y+cfg["damping"]*delta[:3]
                dual = dual+cfg["damping"]*delta[3:]
                output.append(dict(delta=delta.tolist(), next=y.tolist(), dual=dual.tolist()))
    return output


def projection_value(directory, component, values):
    model = get_model(str(directory))
    z = torch.tensor(np.asarray(values, dtype=float))
    with torch.no_grad():
        result = model.projection(z[:2], z[2:])
    return float(result[int(component)])


def projection_gradient(directory, component, values):
    model = get_model(str(directory))
    z = torch.tensor(np.asarray(values, dtype=float))
    def f(v):
        return model.projection(v[:2], v[2:])[int(component)]
    return torch.func.jacrev(f)(z).detach().numpy()


def projection_hessian(directory, component, values):
    model = get_model(str(directory))
    z = torch.tensor(np.asarray(values, dtype=float))
    def f(v):
        return model.projection(v[:2], v[2:])[int(component)]
    return torch.func.hessian(f)(z).detach().numpy()


def complete_value(directory, component, values):
    model = get_model(str(directory))
    x = torch.tensor(np.asarray(values, dtype=float))
    with torch.no_grad():
        result = model(x)
    return float(result[int(component)])


def complete_gradient(directory, component, values):
    model = get_model(str(directory))
    x = torch.tensor(np.asarray(values, dtype=float))
    return torch.func.jacrev(model)(x)[int(component)].detach().numpy()


def complete_hessian(directory, component, values):
    model = get_model(str(directory))
    x = torch.tensor(np.asarray(values, dtype=float))
    return torch.func.hessian(model)(x)[int(component)].detach().numpy()
