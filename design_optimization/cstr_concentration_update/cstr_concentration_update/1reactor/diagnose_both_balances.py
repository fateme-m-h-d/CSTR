from pathlib import Path
import torch

BASE = Path(__file__).resolve().parent

model = torch.load(
    BASE / "bilinear_nn_focused.pt",
    map_location="cpu",
    weights_only=False,
)
model.eval()

# Both-balance formulation + product specification + kinetics
# force rate[A] = -0.095
rA = -0.095
target = -0.94

V = torch.linspace(0.0, 10.0, 10001).reshape(-1, 1)

rA_col = torch.full_like(V, rA)

X = torch.cat(
    [
        rA_col / 10.0,
        V / 10.0,
    ],
    dim=1,
)

with torch.no_grad():
    pred = 100.0 * model(X)

residual = torch.abs(pred - target)

idx = torch.argmin(residual)

print("Required rate[A] =", rA)
print("Required NN output =", target)

print("\nClosest point inside V in [0,10]:")
print("V =", V[idx].item())
print("NN output =", pred[idx].item())
print("Residual =", residual[idx].item())

print("\nNN output range on this slice:")
print("minimum =", pred.min().item())
print("maximum =", pred.max().item())