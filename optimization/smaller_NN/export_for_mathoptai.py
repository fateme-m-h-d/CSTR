import torch
import torch.nn as nn
import argparse

from test import NN


# ---------------------------------------------------------
# Activation module
# ---------------------------------------------------------
def get_activation(name):

    if name == "relu":
        return nn.ReLU()

    elif name == "softplus":
        return nn.Softplus()

    elif name == "tanh":
        return nn.Tanh()

    elif name == "sigmoid":
        return nn.Sigmoid()

    else:
        raise ValueError(
            f"Unknown activation: {name}"
        )


# ---------------------------------------------------------
# Export function
# ---------------------------------------------------------
def export_model(checkpoint_path, output_path):

    # Load checkpoint
    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    # Read model information automatically
    input_dim = ckpt["input_dim"]
    hidden_dim = ckpt["hidden_dim"]
    hidden_num = ckpt["hidden_num"]
    output_dim = ckpt["output_dim"]
    activation = ckpt["activation"]

    print("Checkpoint information:")
    print("  input_dim   =", input_dim)
    print("  hidden_dim  =", hidden_dim)
    print("  hidden_num  =", hidden_num)
    print("  output_dim  =", output_dim)
    print("  activation  =", activation)


    # -----------------------------------------------------
    # Re-create original network
    # -----------------------------------------------------
    model = NN(
        input_dim,
        hidden_dim,
        hidden_num,
        output_dim,
        activation=activation,
    )

    model.load_state_dict(
        ckpt["state_dict"]
    )

    model.eval()


    # -----------------------------------------------------
    # Convert to nn.Sequential
    # -----------------------------------------------------
    layers = []

    for layer in model.layers[:-1]:

        layers.append(layer)

        layers.append(
            get_activation(activation)
        )

    layers.append(
        model.layers[-1]
    )

    sequential_model = nn.Sequential(
        *layers
    )

    sequential_model.eval()


    # -----------------------------------------------------
    # Verify conversion
    # -----------------------------------------------------
    x_test = torch.randn(
        10,
        input_dim,
    )

    with torch.no_grad():

        y_original = model(x_test)

        y_sequential = sequential_model(
            x_test
        )

    max_difference = torch.max(
        torch.abs(
            y_original - y_sequential
        )
    ).item()

    print(
        "Maximum prediction difference:",
        max_difference,
    )

    assert torch.allclose(
        y_original,
        y_sequential,
        atol=1e-7,
    )


    # -----------------------------------------------------
    # Save for MathOptAI
    # -----------------------------------------------------
    torch.save(
        sequential_model,
        output_path,
    )

    print()
    print(
        "Saved MathOptAI model to:",
        output_path,
    )

    print()
    print(sequential_model)


# ---------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--output",
        default="cstr_nn.pt",
        type=str,
    )

    args = parser.parse_args()

    export_model(
        args.checkpoint,
        args.output,
    )