import torch

from src.utils import LoadData, LoadModel, get_violation, compute_violation_original_nonlinear


device = "cpu"

def projection_only_forward(model, X, Y):
    masks = model.get_masks_1d(X)

    fixed_outputs = []

    for r, (fc1, fc2) in enumerate(zip(model.fc1_list, model.fc2_list)):
        y_proj_r = fc1(Y) + fc2(X)

        mask_r = masks[:, r:r+1]
        fixed_outputs.append(y_proj_r * mask_r)

    return sum(fixed_outputs)


def run_projection_check(args, data):
    model = LoadModel(args, data)
    model.to(device)
    model.eval()

    all_output_errors = []
    all_pl_violations = []
    all_nl_violations = []

    with torch.no_grad():
        for X, Y in data["test_loader"]:
            X = X.to(device)
            Y = Y.to(device)

            Y_proj = projection_only_forward(model, X, Y)

            output_error = torch.abs(Y_proj - Y)

            pl_violation = get_violation(
                args,
                data,
                X,
                Y_proj,
            )

            nl_violation = compute_violation_original_nonlinear(
                X,
                Y_proj,
                data["scaler"],
                device=device,
            )

            # Keep dimensions:
            # output_error: [batch_size, 3]
            # pl_violation: [batch_size, 2]
            # nl_violation: [batch_size, 2]
            all_output_errors.append(
                output_error.detach().cpu()
            )

            all_pl_violations.append(
                torch.abs(pl_violation).detach().cpu()
            )

            all_nl_violations.append(
                torch.abs(nl_violation).detach().cpu()
            )

    output_errors = torch.cat(all_output_errors, dim=0)
    pl_violations = torch.cat(all_pl_violations, dim=0)
    nl_violations = torch.cat(all_nl_violations, dim=0)

    results = {
        # Output displacement
        "projection_check_output_MAE_scaled":
            output_errors.mean().item(),

        # PL constraints
        "projection_check_PL_violation":
            pl_violations.mean().item(),

        "projection_check_PL_reaction_violation":
            pl_violations[:, 0].mean().item(),

        "projection_check_PL_mass_balance_violation":
            pl_violations[:, 1].mean().item(),

        # Original constraints
        "projection_check_original_nonlinear_violation":
            nl_violations.mean().item(),

        "projection_check_original_reaction_violation":
            nl_violations[:, 0].mean().item(),

        "projection_check_original_mass_balance_violation":
            nl_violations[:, 1].mean().item(),

        # Counts
        "n_output_values":
            int(output_errors.numel()),

        "n_constraint_values":
            int(pl_violations.numel()),

        "n_data_points":
            int(output_errors.shape[0]),
    }

    for key, value in results.items():
        print(f"{key}: {value}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="KKThPINN")
    parser.add_argument("--model_id", type=str, default="projection_check")
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--hidden_num", type=int, default=2)
    parser.add_argument("--z0_dim", type=int, default=3)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mu", type=float, default=1)
    parser.add_argument("--dtype", type=int, default=64, choices=[32, 64],)
    parser.add_argument("--dataset_type", type=str, default="cstr")
    parser.add_argument("--dataset_path", type=str, default="data.csv")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--job", type=str, default="projection_check")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--run", type=int, default=0)

    args = parser.parse_args()
    
    torch.set_default_dtype(
        torch.float64 if args.dtype == 64 else torch.float32
    )

    data = LoadData(args)
    run_projection_check(args, data)