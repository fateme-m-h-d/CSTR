import argparse
import time
import torch
from utils import LoadData
from train import run_training, evaluate_model


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="KKThPINN", help="NN,  PINN, KKThPINN")
    parser.add_argument("--model_id", type=str, default="MODELID")
    parser.add_argument("--input_dim", type=int, default=1, help="1 for isothermal CSTR: Cao only")
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--hidden_num", type=int, default=2)
    parser.add_argument("--z0_dim", type=int, default=3, help="Ca, Cb, Cc")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mu", type=float, default=1)
    parser.add_argument("--mu_rxn", type=float, default=0.0)
    parser.add_argument("--mu_mb", type=float, default=0.0)
    parser.add_argument("--max_subiter", default=500, type=int)
    parser.add_argument("--eta", default=0.8, type=float)
    parser.add_argument("--sigma", default=2, type=float)
    parser.add_argument("--mu_safe", default=1e9, type=float)
    parser.add_argument("--dtype", default=64, type=int)
    parser.add_argument("--dataset_type", type=str, default="cstr")
    parser.add_argument("--dataset_path", type=str, default="data.csv")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--job", type=str, required=True, help="train or experiment")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--run", type=int, default=0)
    return parser.parse_args()


def main(args):
    if args.model in ["NN", "KKThPINN"]:
        args.loss_type = "MSE"
    elif args.model == 'PINN':
            args.loss_type = 'PINN'
    else:
        raise ValueError("Model not supported")

    # args.run = 0
    data = LoadData(args)

    if args.job == "train":
        run_training(args, data)
    elif args.job == "experiment":
        print(f"\n\nEvaluating {args.model} at run {args.run}")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        eval_start = time.perf_counter()
        scores = evaluate_model(data, args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"Evaluation time : {time.perf_counter() - eval_start:.4f} s")
        print(scores)
    else:
        raise ValueError("job must be train or experiment")


if __name__ == "__main__":
    args = add_arguments()
    print(args)
    main(args)
