from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent

models = [
    "bilinear_nn.pt",
    "bilinear_nn_focused.pt",
]

# Exact physical one-reactor solution
rA_exact = -0.095
V_exact = 0.94 / 0.095
rAV_exact = rA_exact * V_exact

print("Exact physical operating point")
print("rate[A] =", rA_exact)
print("V       =", V_exact)
print("rA*V    =", rAV_exact)


for filename in models:

    path = BASE_DIR / filename

    try:
        model = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        model = torch.load(
            path,
            map_location="cpu",
        )

    model.eval()

    X = torch.tensor(
        [[rA_exact / 10.0, V_exact / 10.0]],
        dtype=torch.float32,
    )

    with torch.no_grad():
        prediction = 100.0 * model(X).item()

    print("\n----------------------------")
    print(filename)
    print("----------------------------")

    print("Exact rA*V =", rAV_exact)
    print("NN rA*V    =", prediction)
    print("Error       =", prediction - rAV_exact)