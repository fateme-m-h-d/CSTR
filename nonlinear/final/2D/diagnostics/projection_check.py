from pathlib import Path
import sys

import torch

# Make the current 2D/src directory importable without changing the repository's
# existing non-package imports such as "from models import ...".
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models import get_masks_2d
from utils import (
    LoadData,
    LoadModel,
    compute_violation_original_nonlinear,
    get_violation,
)

DEVICE = "cpu"


def projection_only_forward(model, X, Y):
    """Project the true output Y using the active 2D regional KKT map."""
    masks = get_masks_2d(X, model.T_edges, model.C_edges)

    output = torch.zeros_like(Y)
    for region, (fc1, fc2) in enumerate(
        zip(model.fc1_list, model.fc2_list)
    ):
        y_proj_region = fc1(Y) + fc2(X)
        output = output + (
            y_proj_region * masks[:, region:region + 1]
        )

    return output


def run_projection_check(args, data):
    model = LoadModel(args, data)
    model.to(DEVICE)
    model.eval()

    all_output_errors = []
    all_pl_violations = []
    all_nl_violations = []

    with torch.no_grad():
        for X, Y in data["test_loader"]:
            X = X.to(DEVICE)
            Y = Y.to(DEVICE)

            Y_proj = projection_only_forward(model, X, Y)

            output_error = torch.abs(Y_proj - Y)
            pl_violation = torch.abs(
                get_violation(args, data, X, Y_proj)
            )
            nl_violation = torch.abs(
                compute_violation_original_nonlinear(
                    X,
                    Y_proj,
                    data["scaler"],
                    device=DEVICE,
                )
            )

            all_output_errors.append(output_error.detach().cpu())
            all_pl_violations.append(pl_violation.detach().cpu())
            all_nl_violations.append(nl_violation.detach().cpu())

    output_errors = torch.cat(all_output_errors, dim=0)
    pl_violations = torch.cat(all_pl_violations, dim=0)
    nl_violations = torch.cat(all_nl_violations, dim=0)

    results = {
        "projection_check_output_MAE_scaled":
            output_errors.mean().item(),

        "projection_check_output_MAE_Ca_scaled":
            output_errors[:, 0].mean().item(),
        "projection_check_output_MAE_Cb_scaled":
            output_errors[:, 1].mean().item(),
        "projection_check_output_MAE_Cc_scaled":
            output_errors[:, 2].mean().item(),

        "projection_check_PL_violation":
            pl_violations.mean().item(),
        "projection_check_PL_reaction_violation":
            pl_violations[:, 0].mean().item(),
        "projection_check_PL_mass_balance_violation":
            pl_violations[:, 1].mean().item(),

        "projection_check_original_nonlinear_violation":
            nl_violations.mean().item(),
        "projection_check_original_reaction_violation":
            nl_violations[:, 0].mean().item(),
        "projection_check_original_mass_balance_violation":
            nl_violations[:, 1].mean().item(),

        "projection_check_original_reaction_max_violation":
            nl_violations[:, 0].max().item(),

        "n_output_values":
            int(output_errors.numel()),
        "n_constraint_values":
            int(pl_violations.numel()),
        "n_data_points":
            int(output_errors.shape[0]),
    }

    print(f"requested_dtype: float{args.dtype}")
    print(f"dataset_tensor_dtype: {data['dataset'].X.dtype}")
    print(f"A_dtype: {data['A_list'][0].dtype}")
    print(f"B_dtype: {data['B_list'][0].dtype}")
    print(f"b_dtype: {data['b_list'][0].dtype}")
    print(f"model_projection_dtype: {model.fc1_list[0].weight.dtype}")

    for key, value in results.items():
        print(f"{key}: {value}")

    return results


def build_parser():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="KKThPINN")
    parser.add_argument(
        "--model_id",
        type=str,
        default="projection_check",
    )
    parser.add_argument("--input_dim", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--hidden_num", type=int, default=2)
    parser.add_argument("--z0_dim", type=int, default=3)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument(
        "--dtype",
        type=int,
        default=64,
        choices=[32, 64],
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="cstr",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data.csv",
    )
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument(
        "--job",
        type=str,
        default="projection_check",
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--run", type=int, default=0)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    torch.set_default_dtype(
        torch.float64 if args.dtype == 64 else torch.float32
    )

    data = LoadData(args)
    run_projection_check(args, data)
