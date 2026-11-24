"""Check saved forwards, projection gradients, and derivatives used by Julia callbacks."""
import argparse
from pathlib import Path
import numpy as np
import torch
from cstr_models import load_model, physics
from bridge import (projection_value, projection_gradient, projection_hessian,
                    complete_gradient, complete_hessian)

ROOT = Path(__file__).resolve().parents[1]


def check(directory):
    model = load_model(directory)
    points = torch.tensor([[391.41, 1.0111], [382.78591974, 1.2], [447.9235, .94952]])
    # Finite differences of the COMPLETE corrected predictor.
    x = points[0].clone().requires_grad_(True)
    analytical = torch.func.jacrev(model)(x).detach().numpy()
    numerical = np.zeros((3, 2))
    for j, epsilon in enumerate([1e-3, 1e-6]):
        direction = torch.zeros_like(x)
        direction[j] = epsilon
        numerical[:, j] = ((model(x+direction)-model(x-direction))/(2*epsilon)).detach().numpy()
    np.testing.assert_allclose(analytical, numerical, rtol=1e-3, atol=1e-5)
    # Verify weights receive gradients THROUGH the projection.
    prediction = model(points)
    loss = (prediction/model.output_scale).square().mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(float(p.grad.abs().sum()) for p in model.backbone.parameters() if p.grad is not None) > 0
    # Check the exact callback's gradients and Hessians in all five inputs.
    raw = model.raw(x).detach()
    z = torch.cat((x.detach(), raw)).numpy()
    gradient = projection_gradient(str(directory), 0, z)
    hessian = projection_hessian(str(directory), 0, z)
    assert np.isfinite(gradient).all() and np.isfinite(hessian).all()
    np.testing.assert_allclose(hessian, hessian.T, rtol=1e-5, atol=1e-5)
    for i in range(5):
        epsilon = 1e-4 if i == 0 else 1e-6
        direction = np.zeros(5)
        direction[i] = epsilon
        fd = (projection_value(str(directory), 0, z+direction) -
              projection_value(str(directory), 0, z-direction))/(2*epsilon)
        np.testing.assert_allclose(gradient[i], fd, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(complete_gradient(str(directory), 0, points[0].numpy()),
                               analytical[0], rtol=1e-7, atol=1e-8)
    complete_H = complete_hessian(str(directory), 0, points[0].numpy())
    assert np.isfinite(complete_H).all()
    np.testing.assert_allclose(complete_H, complete_H.T, rtol=1e-5, atol=1e-5)
    # Test the vmap/jacrev operations used by torch-backed gray-box oracles.
    vmapped = torch.vmap(model)(points)
    np.testing.assert_allclose(vmapped.detach().numpy(), model(points).detach().numpy(),atol=1e-8,rtol=1e-7)
    with torch.no_grad():
        residual = physics(points, model(points), model.cfg)
    print(directory.name, "gradient and serialization checks passed;",
          "max physical residual at check points", float(residual.abs().max()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=ROOT/"artifacts")
    args = parser.parse_args()
    for name in ["baseline", "mass", "pl", "hardnet", "enforce"]:
        check(args.artifacts/name)
