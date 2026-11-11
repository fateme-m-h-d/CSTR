"""
Train a CSTR reaction-extent surrogate.

Inputs in physical units:
    FAin : A molar flow entering reactor [mol/s]
    FBin : B molar flow entering reactor [mol/s]
    Qin  : volumetric flow entering reactor [L/s]
    V    : reactor volume [L]

Output:
    xi   : reaction extent [mol/s]

For A -> B:
    FAout = FAin - xi
    FBout = FBin + xi

Exact CSTR relation:
    xi = k*V/Qin^2 * (FAin - xi)*(FBin + xi)

Saved model:
    extent_nn.pt

Scaling used by Julia later:
    inputs / 10
    output xi / 10
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
    return low + (high - low) * torch.rand(
        n, 1, generator=generator
    )


def exact_extent(FAin, FBin, Qin, V):
    """
    Solve analytically:

        xi = a (FAin - xi)(FBin + xi)

    where:

        a = k V / Qin^2

    Rearranging:

        a xi^2
        + [1 - a(FAin-FBin)] xi
        - a FAin FBin = 0

    We take the physical nonnegative root.
    """

    a = K * V / (Qin ** 2)

    b = 1.0 - a * (FAin - FBin)

    c = -a * FAin * FBin

    discriminant = b**2 - 4.0 * a * c

    discriminant = torch.clamp(
        discriminant,
        min=0.0,
    )

    # Avoid division by zero for V approximately zero
    safe_a = torch.clamp(
        a,
        min=1e-12,
    )

    xi_root = (
        -b + torch.sqrt(discriminant)
    ) / (2.0 * safe_a)

    xi = torch.where(
        a < 1e-10,
        torch.zeros_like(a),
        xi_root,
    )

    return xi


def sample_global(n, generator):
    """
    Generate physically consistent inlet states.

    Total A+B concentration is 1 mol/L, therefore:

        FAin + FBin = Qin

    We sample:
        Qin    in [1, 10] L/s
        CAin   in [0.05, 0.99] mol/L
        V      in [0, 10] L

    Then:
        FAin = CAin * Qin
        FBin = (1 - CAin) * Qin
    """

    Qin = uniform(
        n, 1.0, 10.0, generator
    )

    CAin = uniform(
        n, 0.05, 0.99, generator
    )

    V = uniform(
        n, 0.0, 10.0, generator
    )

    FAin = CAin * Qin

    FBin = (1.0 - CAin) * Qin

    xi = exact_extent(
        FAin,
        FBin,
        Qin,
        V,
    )

    X = torch.cat(
        (
            FAin / 10.0,
            FBin / 10.0,
            Qin / 10.0,
            V / 10.0,
        ),
        dim=1,
    )

    y = xi / 10.0

    return X, y


def sample_focused(n, generator):
    """
    Extra sampling near the exact one-reactor optimum.

    Exact solution is approximately:

        Qin   = 1.20
        CAin  = 0.8335
        V     = 9.895
        xi    = 0.94
    """

    Qin = uniform(
        n, 1.0, 2.0, generator
    )

    CAin = uniform(
        n, 0.75, 0.90, generator
    )

    V = uniform(
        n, 7.0, 10.0, generator
    )

    FAin = CAin * Qin

    FBin = (1.0 - CAin) * Qin

    xi = exact_extent(
        FAin,
        FBin,
        Qin,
        V,
    )

    X = torch.cat(
        (
            FAin / 10.0,
            FBin / 10.0,
            Qin / 10.0,
            V / 10.0,
        ),
        dim=1,
    )

    y = xi / 10.0

    return X, y


def physical_report(model, X, y, label):

    model.eval()

    with torch.no_grad():

        prediction = model(X)

        # Convert scaled error back to mol/s
        error = 10.0 * (
            prediction - y
        )

    print(
        f"{label}: "
        f"RMSE={error.square().mean().sqrt().item():.8g}, "
        f"MAE={error.abs().mean().item():.8g}, "
        f"max|error|={error.abs().max().item():.8g}"
    )


def exact_operating_point_check(model):

    # Reconstructed from your exact optimization
    FAin = 0.9999856092389877
    FBin = 0.19972657554076595
    Qin = 1.1997121847797536
    V = 9.894681081430791

    X = torch.tensor(
        [[
            FAin / 10.0,
            FBin / 10.0,
            Qin / 10.0,
            V / 10.0,
        ]],
        dtype=torch.float32,
    )

    exact_xi = exact_extent(
        torch.tensor([[FAin]]),
        torch.tensor([[FBin]]),
        torch.tensor([[Qin]]),
        torch.tensor([[V]]),
    ).item()

    model.eval()

    with torch.no_grad():

        predicted_xi = (
            10.0 * model(X).item()
        )

    print(
        "\n--- Exact operating-point check ---"
    )

    print("FAin =", FAin)
    print("FBin =", FBin)
    print("Qin  =", Qin)
    print("V    =", V)

    print("Exact xi =", exact_xi)
    print("NN xi    =", predicted_xi)

    print(
        "Error    =",
        predicted_xi - exact_xi,
    )


def main():

    torch.manual_seed(0)

    train_gen = torch.Generator().manual_seed(1)
    val_gen = torch.Generator().manual_seed(2)
    test_gen = torch.Generator().manual_seed(3)

    # ------------------------------------------------------
    # Training data
    # ------------------------------------------------------

    X_global, y_global = sample_global(
        40000,
        train_gen,
    )

    X_focus, y_focus = sample_focused(
        20000,
        train_gen,
    )

    # Give focused samples somewhat greater importance
    w_global = torch.ones_like(y_global)

    w_focus = (
        5.0 * torch.ones_like(y_focus)
    )

    X_train = torch.cat(
        (X_global, X_focus),
        dim=0,
    )

    y_train = torch.cat(
        (y_global, y_focus),
        dim=0,
    )

    weights = torch.cat(
        (w_global, w_focus),
        dim=0,
    )

    dataset = TensorDataset(
        X_train,
        y_train,
        weights,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # ------------------------------------------------------
    # Validation/test data
    # ------------------------------------------------------

    X_val, y_val = sample_global(
        10000,
        val_gen,
    )

    X_test, y_test = sample_global(
        20000,
        test_gen,
    )

    X_test_focus, y_test_focus = (
        sample_focused(
            5000,
            test_gen,
        )
    )

    # ------------------------------------------------------
    # NN
    # ------------------------------------------------------

    model = nn.Sequential(
        nn.Linear(4, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    scheduler = (
        torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=200,
            gamma=0.3,
        )
    )

    best_val = float("inf")
    best_state = None
    best_epoch = None

    # ------------------------------------------------------
    # Training
    # ------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        for X, y, w in loader:

            optimizer.zero_grad()

            prediction = model(X)

            loss = (
                w
                * (prediction - y).square()
            ).mean()

            loss.backward()

            optimizer.step()

        scheduler.step()

        model.eval()

        with torch.no_grad():

            val_loss = (
                (
                    model(X_val)
                    - y_val
                )
                .square()
                .mean()
                .item()
            )

        if val_loss < best_val:

            best_val = val_loss

            best_state = deepcopy(
                model.state_dict()
            )

            best_epoch = epoch

        if (
            epoch == 1
            or epoch % 50 == 0
        ):

            print(
                f"Epoch {epoch}/{EPOCHS}: "
                f"validation physical RMSE = "
                f"{10.0 * val_loss**0.5:.8g}"
            )

    model.load_state_dict(
        best_state
    )

    model.eval()

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    save_path = (
        BASE_DIR
        / "extent_nn.pt"
    )

    torch.save(
        model,
        save_path,
    )

    print(
        "\nSaved model:",
        save_path,
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    # ------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------

    physical_report(
        model,
        X_test,
        y_test,
        "Global test",
    )

    physical_report(
        model,
        X_test_focus,
        y_test_focus,
        "Focused test",
    )

    exact_operating_point_check(
        model
    )


if __name__ == "__main__":
    main()