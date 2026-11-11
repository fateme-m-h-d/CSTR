from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILES = [
    "bilinear_nn.pt",
    "bilinear_nn_focused.pt",
]


def load_model(path):

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

    return model


for model_name in MODEL_FILES:

    print("\n======================================")
    print(model_name)
    print("======================================")

    model = load_model(BASE_DIR / model_name)

    # -------------------------------------------------
    # Remaining exact process equations force:
    #
    # rate[B]*V = +0.94
    # rate[A] = -rate[B]
    #
    # therefore:
    #
    # rate[A] = -0.94 / V
    #
    # rate[A] is bounded below by -10,
    # therefore V >= 0.094
    # -------------------------------------------------

    volumes = torch.linspace(
        0.094,
        10.0,
        20001,
    ).reshape(-1, 1)

    rates = -0.94 / volumes

    X = torch.cat(
        (
            rates / 10.0,
            volumes / 10.0,
        ),
        dim=1,
    )

    with torch.no_grad():
        nn_values = 100.0 * model(X)

    # -------------------------------------------------
    # For the rest of the process equations to hold,
    # the A balance + splitter + kinetics require:
    #
    # NN_required(V)
    #   = 0.94/(1.9 V) - 0.99
    # -------------------------------------------------

    required = (
        0.94 / (1.9 * volumes)
        - 0.99
    )

    residual = (
        nn_values - required
    ).flatten()

    print(
        "Residual min/max:",
        residual.min().item(),
        residual.max().item(),
    )

    # -------------------------------------------------
    # Find sign changes = possible intersections
    # -------------------------------------------------

    roots = []

    for i in range(len(volumes) - 1):

        r1 = residual[i].item()
        r2 = residual[i + 1].item()

        if r1 == 0 or r1 * r2 < 0:

            V1 = volumes[i].item()
            V2 = volumes[i + 1].item()

            # Linear interpolation for approximate root
            if r2 != r1:
                Vroot = (
                    V1
                    - r1 * (V2 - V1)
                    / (r2 - r1)
                )
            else:
                Vroot = V1

            roots.append(Vroot)

    if len(roots) == 0:

        print("NO feasible intersections found.")

    else:

        print(
            "\nApproximate feasible intersections:"
        )

        for Vroot in roots:

            rA = -0.94 / Vroot

            inp = torch.tensor(
                [[
                    rA / 10.0,
                    Vroot / 10.0,
                ]],
                dtype=torch.float32,
            )

            with torch.no_grad():

                g = (
                    100.0
                    * model(inp).item()
                )

            PA = 0.99 + g
            PB = 0.95

            print("\nV =", Vroot)
            print("rate[A] =", rA)

            print(
                "Exact rate[A]*V =",
                rA * Vroot,
            )

            print(
                "NN rate[A]*V =",
                g,
            )

            print(
                "NN multiplication error =",
                g - rA * Vroot,
            )

            print("P[A] =", PA)
            print("P[B] =", PB)

            print(
                "P[A] + P[B] =",
                PA + PB,
            )