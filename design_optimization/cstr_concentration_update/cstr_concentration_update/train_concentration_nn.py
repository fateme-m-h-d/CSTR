"""
Train a concentration-based reaction-source surrogate for the 5-reactor CSTR
superstructure.

The SAME physical mapping is used at every active reactor n:

    (C_A,n, C_B,n, V_n) -> xi_n

with

    xi_n = k * C_A,n * C_B,n * V_n

for the reaction A -> B and k = 2.

Why a separate 5-reactor training file?
---------------------------------------
The 1-reactor optimum is near V ~ 9.9 L and C_A ~ 0.05, whereas the 5-reactor
superstructure operates in a much lower-volume region (the exact 5-reactor
solution is around V ~ 0.61 L per active reactor) and spans a broader range of
concentrations along the reactor train. This file therefore keeps global
coverage but adds extra density in the low-volume, multi-reactor design region.

Scaling expected by Julia:
    C_A, C_B: unscaled, already in [0,1]
    V_scaled = V / 10
    xi_scaled = xi / 10

Saved model:
    concentration_extent_nn_5reactor.pt
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
    """Exact positive reaction-source magnitude [mol/s]."""
    return K * CA * CB * V


def pack(CA, CB, V):
    xi = exact_xi(CA, CB, V)
    X = torch.cat((CA, CB, V / 10.0), dim=1)
    y = xi / 10.0
    return X, y


def sample_global(n, generator):
    """Full physically consistent domain used by the optimization."""
    CA = uniform(n, 0.0, 1.0, generator)
    CB = 1.0 - CA
    V = uniform(n, 0.0, 10.0, generator)
    return pack(CA, CB, V)


def sample_low_volume(n, generator):
    """
    Extra coverage where multi-reactor designs are likely to operate.
    Keep the full concentration range but emphasize V <= 2 L.
    """
    CA = uniform(n, 0.0, 1.0, generator)
    CB = 1.0 - CA
    V = uniform(n, 0.0, 2.0, generator)
    return pack(CA, CB, V)


def sample_design_region(n, generator):
    """
    Additional density around the concentration/volume region traversed by the
    exact 5-reactor design, while remaining broader than one single optimum.
    """
    CA = uniform(n, 0.02, 0.80, generator)
    CB = 1.0 - CA
    V = uniform(n, 0.2, 1.5, generator)
    return pack(CA, CB, V)


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


def five_reactor_reference_check(model):
    """
    Diagnostic only: evaluate the NN at the exact 5-reactor concentration
    profile previously obtained. These points are NOT hard-coded into training.
    """
    CA_vals = torch.tensor(
        [[0.05], [0.1029073738195608], [0.20573455572774543],
         [0.38774472085224376], [0.6521692502747114]],
        dtype=torch.float32,
    )
    CB_vals = 1.0 - CA_vals
    V_vals = torch.full_like(CA_vals, 0.612400970206008)

    X = torch.cat((CA_vals, CB_vals, V_vals / 10.0), dim=1)
    exact = exact_xi(CA_vals, CB_vals, V_vals)

    model.eval()
    with torch.no_grad():
        pred = 10.0 * model(X)

    print("\n--- Five-reactor reference-point check ---")
    for n in range(5):
        print(
            f"n={n+1}: "
            f"CA={CA_vals[n].item():.8f}, "
            f"CB={CB_vals[n].item():.8f}, "
            f"V={V_vals[n].item():.8f}, "
            f"exact xi={exact[n].item():.8f}, "
            f"NN xi={pred[n].item():.8f}, "
            f"error={(pred[n]-exact[n]).item():.8f}"
        )


def main():
    torch.manual_seed(0)

    train_gen = torch.Generator().manual_seed(1)
    val_gen = torch.Generator().manual_seed(2)
    test_gen = torch.Generator().manual_seed(3)

    # Global coverage plus extra density in the region relevant to the
    # multi-reactor design problem.
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

    save_path = BASE_DIR / "concentration_extent_nn_5reactor.pt"
    torch.save(model, save_path)

    print("\nSaved model:", save_path)
    print("Best epoch:", best_epoch)

    physical_report(model, X_test_global, y_test_global, "Global test")
    physical_report(model, X_test_low, y_test_low, "Low-volume test")
    physical_report(model, X_test_design, y_test_design, "5-reactor design-region test")
    five_reactor_reference_check(model)


if __name__ == "__main__":
    main()
