"""Differentiable CSTR models. All arithmetic in projection.forward is retained for training."""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn

torch.set_default_dtype(torch.float64)


def default_config(method, ear=100000.0, steps=40):
    return dict(method=method, ear=ear, steps=steps, damping=0.5, regularization=1e-6,
                temperature_edges=np.linspace(280, 460, 12).tolist(),
                feed_edges=np.linspace(0.8, 1.2, 4).tolist(), gate_width_fraction=0.125,
                activation="tanh", hidden=32, projection_metric="physical_euclidean",
                projected_constraints=["A_kinetic", "summed_concentration"],
                objective=dict(feed_penalty=0.3, temperature_penalty=0.5,
                               temperature_reference=370.0, temperature_scale=90.0))


def kinetics(t, cfg):
    return 1e13 * torch.exp(-90000.0 / (8.314*t)), 1e11 * torch.exp(-cfg["ear"]/(8.314*t))


def physics(x, y, cfg):
    t, a = x.unbind(-1)
    ca, cb, cc = y.unbind(-1)
    f, r = kinetics(t, cfg)
    return torch.stack((a-ca-10*f*ca*cb**2+10*r*cc, a+2-ca-cb-cc), -1)


def jacobian_y(x, y, cfg):
    f, r = kinetics(x[..., 0], cfg)
    ca, cb, _ = y.unbind(-1)
    row = torch.stack((-1-10*f*cb**2, -20*f*ca*cb, 10*r), -1)
    return torch.stack((row, -torch.ones_like(row)), -2)


def hessian_a(x, y, cfg):
    f, _ = kinetics(x[..., 0], cfg)
    z = torch.zeros_like(f)
    cross = -20*f*y[..., 1]
    diagonal = -20*f*y[..., 0]
    return torch.stack((z, cross, z, cross, diagonal, z, z, z, z), -1).reshape(*f.shape, 3, 3)


def reference_numpy(x, cfg):
    x = np.atleast_2d(x)
    t, a = x.T
    f = 1e13*np.exp(-90000/(8.314*t))
    r = 1e11*np.exp(-cfg["ear"]/(8.314*t))
    lo, hi = np.zeros_like(a), np.minimum(a, 1.0)
    for _ in range(64):
        extent = (lo+hi)/2
        g = (1+30*r)*extent-10*f*(a-extent)*(2-2*extent)**2
        hi = np.where(g > 0, extent, hi)
        lo = np.where(g <= 0, extent, lo)
    extent = (lo+hi)/2
    return np.column_stack((a-extent, 2-2*extent, 3*extent))


def region_rows(cfg):
    rows = []
    te, ae = cfg["temperature_edges"], cfg["feed_edges"]
    for i in range(len(te)-1):
        for j in range(len(ae)-1):
            t, a = (te[i]+te[i+1])/2, (ae[j]+ae[j+1])/2
            y = torch.tensor(reference_numpy([[t, a]], cfg)[0])
            x = torch.tensor([t, a])
            B = jacobian_y(x, y, cfg).numpy()
            f, r = kinetics(x[0], cfg)
            dt = float((-10*f*90000*y[0]*y[1]**2+10*r*cfg["ear"]*y[2])/(8.314*t*t))
            A = np.array([[dt, 1.0], [0.0, 1.0]])
            b = A@x.numpy()+B@y.numpy()-physics(x, y, cfg).numpy()
            rows.append(dict(A=A.tolist(), B=B.tolist(), b=b.tolist()))
    return rows


def interval_weights(v, edges, fraction):
    width = (edges[1]-edges[0])*fraction
    cuts = torch.sigmoid((v[..., None]-v.new_tensor(edges[1:-1]))/width)
    cumulative = torch.cat((torch.ones_like(v[..., None]), cuts,
                            torch.zeros_like(v[..., None])), -1)
    return cumulative[..., :-1]-cumulative[..., 1:]


class Projection(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def affine_system(self, x):
        cfg = self.cfg
        if cfg["method"] == "mass":
            B = x.new_tensor([[-1.0, -1.0, -1.0]]).expand(*x.shape[:-1], 1, 3)
            return B, (-x[..., 1]-2)[..., None]
        wt = interval_weights(x[..., 0], cfg["temperature_edges"], cfg["gate_width_fraction"])
        wa = interval_weights(x[..., 1], cfg["feed_edges"], cfg["gate_width_fraction"])
        weights = (wt[..., :, None]*wa[..., None, :]).flatten(-2)
        rows = cfg["regions"]
        A = x.new_tensor([row["A"] for row in rows])
        B = x.new_tensor([row["B"] for row in rows])
        b = x.new_tensor([row["b"] for row in rows])
        blended_B = torch.einsum("...r,rij->...ij", weights, B)
        rhs = b-torch.einsum("rij,...j->...ri", A, x)
        return blended_B, torch.einsum("...r,...ri->...i", weights, rhs)

    def forward(self, x, raw):
        cfg = self.cfg
        if cfg["method"] == "baseline":
            return raw
        if cfg["method"] in ("mass", "pl"):
            B, rhs = self.affine_system(x)
            dual = torch.linalg.solve(B@B.transpose(-2, -1),
                                      (B@raw[..., None]).squeeze(-1)-rhs)
            return raw-(B.transpose(-2, -1)@dual[..., None]).squeeze(-1)
        y, dual = raw, torch.zeros_like(raw[..., :2])
        eye3 = torch.eye(3, dtype=raw.dtype, device=raw.device)
        eye5 = torch.eye(5, dtype=raw.dtype, device=raw.device)
        zero22 = raw.new_zeros(*raw.shape[:-1], 2, 2)
        for _ in range(cfg["steps"]):
            g, J = physics(x, y, cfg), jacobian_y(x, y, cfg)
            if cfg["method"] == "enforce":
                correction = torch.linalg.solve(J@J.transpose(-2, -1), g)
                # Exact tangent-plane projection at each stage.
                y = y-(J.transpose(-2, -1)@correction[..., None]).squeeze(-1)
            else:
                stationarity = y-raw+(J.transpose(-2, -1)@dual[..., None]).squeeze(-1)
                H = eye3+dual[..., 0, None, None]*hessian_a(x, y, cfg)
                K = torch.cat((torch.cat((H, J.transpose(-2, -1)), -1),
                               torch.cat((J, zero22), -1)), -2)
                F = torch.cat((stationarity, g), -1)
                # Regularized Gauss-Newton on the physical-coordinate KKT residual.
                normal = K.transpose(-2, -1)@K+cfg["regularization"]*eye5
                rhs = -(K.transpose(-2, -1)@F[..., None]).squeeze(-1)
                delta = torch.linalg.solve(normal, rhs)
                y = y+cfg["damping"]*delta[..., :3]
                dual = dual+cfg["damping"]*delta[..., 3:]
        return y


class ProjectedModel(nn.Module):
    """Physical input -> scaled MLP -> physical projection -> physical output."""
    def __init__(self, backbone, cfg):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        self.projection = Projection(cfg)
        self.register_buffer("input_scale", torch.tensor(cfg["input_scale"]))
        self.register_buffer("output_scale", torch.tensor(cfg["output_scale"]))

    def raw(self, x):
        return self.backbone(x/self.input_scale)*self.output_scale

    def forward(self, x):
        return self.projection(x, self.raw(x))


def make_backbone(cfg):
    activation = dict(tanh=nn.Tanh, softplus=nn.Softplus, sigmoid=nn.Sigmoid)[cfg["activation"]]
    h = cfg["hidden"]
    return nn.Sequential(nn.Linear(2, h), activation(), nn.Linear(h, h), activation(), nn.Linear(h, 3))


def load_model(directory):
    return torch.load(Path(directory)/"complete.pt", map_location="cpu", weights_only=False).eval()


def read_config(directory):
    return json.loads((Path(directory)/"config.json").read_text())
