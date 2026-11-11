"""
Train the MOLAR-FLOW reaction-extent surrogate for the 5-reactor CSTR model.

Inputs:
    FAin, FBin, Qin, V
Output:
    xi

For A -> B:
    FAout = FAin - xi
    FBout = FBin + xi

Exact CSTR relation:
    xi = k*V/Qin^2 * (FAin - xi)*(FBin + xi)

The 5-reactor design operates at much lower V than the 1-reactor optimum,
so this trainer keeps global coverage and adds extra low-volume/design-region
samples.

Scaling used by Julia:
    inputs / 10
    output xi / 10

Saved model:
    extent_nn_5reactor.pt
"""

from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = Path(__file__).resolve().parent
K = 2.0
EPOCHS = 600
BATCH_SIZE = 512


def uniform(n, low, high, generator):
    return low + (high - low) * torch.rand(n, 1, generator=generator)


def exact_extent(FAin, FBin, Qin, V):
    a = K * V / (Qin ** 2)
    b = 1.0 - a * (FAin - FBin)
    c = -a * FAin * FBin

    disc = torch.clamp(b**2 - 4.0 * a * c, min=0.0)
    safe_a = torch.clamp(a, min=1e-12)

    xi_root = (-b + torch.sqrt(disc)) / (2.0 * safe_a)
    xi = torch.where(a < 1e-10, torch.zeros_like(a), xi_root)
    return xi


def pack(Qin, CAin, V):
    FAin = CAin * Qin
    FBin = (1.0 - CAin) * Qin
    xi = exact_extent(FAin, FBin, Qin, V)

    X = torch.cat(
        (FAin / 10.0, FBin / 10.0, Qin / 10.0, V / 10.0),
        dim=1,
    )
    y = xi / 10.0
    return X, y


def sample_global(n, generator):
    Qin = uniform(n, 1.0, 10.0, generator)
    CAin = uniform(n, 0.02, 0.99, generator)
    V = uniform(n, 0.0, 10.0, generator)
    return pack(Qin, CAin, V)


def sample_low_volume(n, generator):
    Qin = uniform(n, 1.0, 3.0, generator)
    CAin = uniform(n, 0.02, 0.99, generator)
    V = uniform(n, 0.0, 2.0, generator)
    return pack(Qin, CAin, V)


def sample_design_region(n, generator):
    # Broader than the exact 5-reactor optimum, but focused near the region
    # visited by the multi-reactor design.
    Qin = uniform(n, 1.0, 2.0, generator)
    CAin = uniform(n, 0.05, 0.95, generator)
    V = uniform(n, 0.2, 1.5, generator)
    return pack(Qin, CAin, V)


def physical_report(model, X, y, label):
    model.eval()
    with torch.no_grad():
        pred = model(X)
        err = 10.0 * (pred - y)

    print(
        f"{label}: "
        f"RMSE={err.square().mean().sqrt().item():.8g}, "
        f"MAE={err.abs().mean().item():.8g}, "
        f"max|error|={err.abs().max().item():.8g}"
    )


def main():
    torch.manual_seed(0)

    train_gen = torch.Generator().manual_seed(1)
    val_gen = torch.Generator().manual_seed(2)
    test_gen = torch.Generator().manual_seed(3)

    X_global, y_global = sample_global(50000, train_gen)
    X_low, y_low = sample_low_volume(30000, train_gen)
    X_design, y_design = sample_design_region(30000, train_gen)

    w_global = torch.ones_like(y_global)
    w_low = 3.0 * torch.ones_like(y_low)
    w_design = 5.0 * torch.ones_like(y_design)

    X_train = torch.cat((X_global, X_low, X_design), dim=0)
    y_train = torch.cat((y_global, y_low, y_design), dim=0)
    weights = torch.cat((w_global, w_low, w_design), dim=0)

    loader = DataLoader(
        TensorDataset(X_train, y_train, weights),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    X_val, y_val = sample_global(10000, val_gen)
    X_test_global, y_test_global = sample_global(20000, test_gen)
    X_test_low, y_test_low = sample_low_volume(10000, test_gen)
    X_test_design, y_test_design = sample_design_region(10000, test_gen)

    model = nn.Sequential(
        nn.Linear(4, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=200, gamma=0.3
    )

    best_val = float("inf")
    best_state = None
    best_epoch = None

    for epoch in range(1, EPOCHS + 1):
        model.train()

        for X, y, w in loader:
            optimizer.zero_grad()
            pred = model(X)
            loss = (w * (pred - y).square()).mean()
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = (model(X_val) - y_val).square().mean().item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch

        if epoch == 1 or epoch % 50 == 0:
            print(
                f"Epoch {epoch}/{EPOCHS}: "
                f"validation physical RMSE = {10.0 * val_loss**0.5:.8g}"
            )

    model.load_state_dict(best_state)
    model.eval()

    save_path = BASE_DIR / "extent_nn_5reactor.pt"
    torch.save(model, save_path)

    print("\nSaved model:", save_path)
    print("Best epoch:", best_epoch)

    physical_report(model, X_test_global, y_test_global, "Global test")
    physical_report(model, X_test_low, y_test_low, "Low-volume test")
    physical_report(model, X_test_design, y_test_design, "5-reactor design-region test")


if __name__ == "__main__":
    main()
