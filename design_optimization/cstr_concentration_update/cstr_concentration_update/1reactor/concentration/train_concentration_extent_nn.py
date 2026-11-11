"""
Train a concentration-based CSTR reaction-source surrogate.

NN mapping
----------
Inputs:
    C_A : reactor outlet concentration of A [mol/L]
    C_B : reactor outlet concentration of B [mol/L]
    V   : reactor volume [L]

Output:
    xi  : positive reaction extent/source magnitude [mol/s]

For A -> B with
    rate_A = -k*C_A*C_B,
we have
    xi = -rate_A*V = k*C_A*C_B*V.

This is the concentration-variable analogue of the previous reaction-extent
surrogate. The same xi will be used with opposite signs in the A and B
component balances, preserving stoichiometry exactly.

Scaling used by Julia:
    C_A, C_B are already O(1), so they are not scaled.
    V_scaled = V / 10
    xi_scaled = xi / 10

Saved model:
    concentration_extent_nn.pt
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


def exact_xi(CA, CB, V):
    """Exact positive reaction source magnitude [mol/s]."""
    return K * CA * CB * V


def sample_global(n, generator):
    """
    Sample physically consistent concentration states.

    For this A -> B model, total A+B concentration is 1 mol/L, so
        C_B = 1 - C_A.

    Sample the full concentration and reactor-volume ranges that the
    optimization may explore.
    """
    CA = uniform(n, 0.0, 1.0, generator)
    CB = 1.0 - CA
    V = uniform(n, 0.0, 10.0, generator)

    xi = exact_xi(CA, CB, V)

    X = torch.cat((CA, CB, V / 10.0), dim=1)
    y = xi / 10.0

    return X, y


def sample_focused(n, generator):
    """
    Extra samples near the product-composition / optimal-volume region:
        C_A ~ 0.05
        C_B ~ 0.95
        V   ~ 9.9 L
    """
    CA = uniform(n, 0.01, 0.15, generator)
    CB = 1.0 - CA
    V = uniform(n, 7.0, 10.0, generator)

    xi = exact_xi(CA, CB, V)

    X = torch.cat((CA, CB, V / 10.0), dim=1)
    y = xi / 10.0

    return X, y


def physical_report(model, X, y, label):
    model.eval()
    with torch.no_grad():
        prediction = model(X)
        error = 10.0 * (prediction - y)

    print(
        f"{label}: "
        f"RMSE={error.square().mean().sqrt().item():.8g}, "
        f"MAE={error.abs().mean().item():.8g}, "
        f"max|error|={error.abs().max().item():.8g}"
    )


def exact_operating_point_check(model):
    CA = 0.05
    CB = 0.95
    V = 0.94 / (K * CA * CB)

    X = torch.tensor([[CA, CB, V / 10.0]], dtype=torch.float32)
    exact = exact_xi(
        torch.tensor([[CA]], dtype=torch.float32),
        torch.tensor([[CB]], dtype=torch.float32),
        torch.tensor([[V]], dtype=torch.float32),
    ).item()

    model.eval()
    with torch.no_grad():
        pred = 10.0 * model(X).item()

    print("\n--- Exact operating-point check ---")
    print("C_A =", CA)
    print("C_B =", CB)
    print("V   =", V)
    print("Exact xi =", exact)
    print("NN xi    =", pred)
    print("Error    =", pred - exact)


def main():
    torch.manual_seed(0)

    train_gen = torch.Generator().manual_seed(1)
    val_gen = torch.Generator().manual_seed(2)
    test_gen = torch.Generator().manual_seed(3)

    # Global coverage + extra density near the design region.
    X_global, y_global = sample_global(50000, train_gen)
    X_focus, y_focus = sample_focused(20000, train_gen)

    w_global = torch.ones_like(y_global)
    w_focus = 5.0 * torch.ones_like(y_focus)

    X_train = torch.cat((X_global, X_focus), dim=0)
    y_train = torch.cat((y_global, y_focus), dim=0)
    weights = torch.cat((w_global, w_focus), dim=0)

    loader = DataLoader(
        TensorDataset(X_train, y_train, weights),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    X_val, y_val = sample_global(10000, val_gen)
    X_test, y_test = sample_global(20000, test_gen)
    X_test_focus, y_test_focus = sample_focused(5000, test_gen)

    model = nn.Sequential(
        nn.Linear(3, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=200,
        gamma=0.3,
    )

    best_val = float("inf")
    best_state = None
    best_epoch = None

    for epoch in range(1, EPOCHS + 1):
        model.train()

        for X, y, w in loader:
            optimizer.zero_grad()
            prediction = model(X)
            loss = (w * (prediction - y).square()).mean()
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

    save_path = BASE_DIR / "concentration_extent_nn.pt"
    torch.save(model, save_path)

    print("\nSaved model:", save_path)
    print("Best epoch:", best_epoch)

    physical_report(model, X_test, y_test, "Global test")
    physical_report(model, X_test_focus, y_test_focus, "Focused test")
    exact_operating_point_check(model)


if __name__ == "__main__":
    main()
