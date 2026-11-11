"""Run: python train_bilinear.py

Train z=x*y with added coverage of small negative rates and zero axes.
Save bilinear_nn_focused.pt; preserve the old bilinear_nn.pt.
Julia scaling stays [rate[A]/10, V/10], output multiplied by 100.
No process equation is used as a training label: every label is exact x*y.
"""
from copy import deepcopy
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = Path(__file__).resolve().parent
EPOCHS = 600
BATCH_SIZE = 512
NAMES = ("global", "small_negative_rates", "near_exact_operation", "zero_axes")
GROUP_WEIGHTS = (1.0, 10.0, 10.0, 2.0)


def uniform(n, low, high, generator):
    return low + (high - low) * torch.rand(n, 1, generator=generator)


def sample(n, group, generator):
    if group == 0:
        x = uniform(n, -10, 10, generator)
        y = uniform(n, 0, 10, generator)
    elif group == 1:
        x = uniform(n, -0.5, 0, generator)
        y = uniform(n, 0, 10, generator)
    elif group == 2:
        # Cover a neighborhood, not just the exact operating point.
        x = uniform(n, -0.15, -0.05, generator)
        y = uniform(n, 7, 10, generator)
    else:
        x = uniform(n, -10, 10, generator)
        y = uniform(n, 0, 10, generator)
        x[:n // 2] = 0
        y[n // 2:] = 0
    return torch.cat((x / 10, y / 10), dim=1), x * y / 100


def validation_score(model, groups):
    model.eval()
    with torch.no_grad():
        mses = [((model(x) - z) ** 2).mean().item() for x, z in groups]
    return sum(w * mse for w, mse in zip(GROUP_WEIGHTS, mses)) / sum(GROUP_WEIGHTS)


def report(model, label, x, z):
    with torch.no_grad():
        error = 100 * (model(x) - z)
    print(f"{label}: physical RMSE={error.square().mean().sqrt().item():.8g}, "
          f"MAE={error.abs().mean().item():.8g}, "
          f"max |error|={error.abs().max().item():.8g}", flush=True)


def process_checks(model):
    # Diagnostics for the uploaded single-CSTR model only.
    points = torch.tensor([
        [-0.095, 0.94 / 0.095],
        [-0.2172874359755112, 4.326066034717796],
    ])
    with torch.no_grad():
        predictions = (100 * model(points / 10)).flatten()
    print("\nImportant points (physical units):")
    for point, pred in zip(points, predictions):
        exact = (point[0] * point[1]).item()
        print(f"rate={point[0].item():.9g}, V={point[1].item():.9g}, "
              f"exact={exact:.9g}, NN={pred.item():.9g}, "
              f"error={pred.item() - exact:+.9g}")

    # Retained B balance and stoichiometry imply rate[A]*V=-0.94.
    # The rate and volume bounds imply 0.094 <= V <= 10.
    volumes = torch.linspace(0.094, 10, 20001).reshape(-1, 1)
    rates = -0.94 / volumes
    x = torch.cat((rates / 10, volumes / 10), dim=1)
    # A balance plus retained kinetics/splitter would require this NN output.
    required = 0.94 / (1.9 * volumes) - 0.99
    with torch.no_grad():
        residual = (100 * model(x) - required).flatten()
    report(model, "Process curve product accuracy", x, rates * volumes / 100)
    crossings = (residual[:-1] * residual[1:] <= 0).sum().item()
    print(f"A-balance residual range on grid: [{residual.min().item():+.9g}, "
          f"{residual.max().item():+.9g}]")
    print(f"Grid intervals bracketing a possible root: {crossings}")
    print("Grid checks and low RMSE are not feasibility certificates. "
          "Rerun the original Julia optimization with this new model.")


def main():
    torch.manual_seed(0)
    train_gen = torch.Generator().manual_seed(1)
    val_gen = torch.Generator().manual_seed(2)
    test_gen = torch.Generator().manual_seed(3)
    shuffle_gen = torch.Generator().manual_seed(4)
    counts = (8000, 8000, 8000, 4000)
    groups = [sample(n, i, train_gen) for i, n in enumerate(counts)]
    # Normalize so each group's weight is independent of its sample count.
    weights = [torch.full((n, 1), sum(counts) * w / (n * sum(GROUP_WEIGHTS)))
               for n, w in zip(counts, GROUP_WEIGHTS)]
    data = TensorDataset(torch.cat([g[0] for g in groups]),
                         torch.cat([g[1] for g in groups]), torch.cat(weights))
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True,
                        generator=shuffle_gen)
    val_groups = [sample(2000, i, val_gen) for i in range(4)]
    test_groups = [sample(5000, i, test_gen) for i in range(4)]

    # Keep the same architecture for the same MathOptAI embedding workflow.
    model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(),
                          nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.3)
    best_score, best_state, best_epoch = float("inf"), None, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for x, z, w in loader:
            optimizer.zero_grad()
            loss = (w * (model(x) - z).square()).mean()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(x)
        scheduler.step()
        score = validation_score(model, val_groups)
        if score < best_score:
            best_score, best_state, best_epoch = score, deepcopy(model.state_dict()), epoch
        if epoch == 1 or epoch % 50 == 0:
            print(f"Epoch {epoch}/{EPOCHS}: mean weighted scaled MSE="
                  f"{loss_sum / len(data):.8g}, validation weighted physical RMSE="
                  f"{100 * score ** 0.5:.8g}", flush=True)

    if best_state is None:
        raise RuntimeError("No finite validation checkpoint was obtained.")
    model.load_state_dict(best_state)
    model.eval()
    path = BASE_DIR / "bilinear_nn_focused.pt"
    torch.save(model, path)
    print(f"\nSaved best validation checkpoint (epoch {best_epoch}): {path}")
    print("Julia: change the predictor filename to bilinear_nn_focused.pt.")
    print("Keep the same scaling and original process equations/objective.")
    print("\nIndependent test results:")
    for name, (x, z) in zip(NAMES, test_groups):
        report(model, name, x, z)
    process_checks(model)


if __name__ == "__main__":
    main()
