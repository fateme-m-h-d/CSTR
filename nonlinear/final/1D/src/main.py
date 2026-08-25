import argparse
import time

import torch

from .train import evaluate_model, run_training
from .utils import LoadData


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="KKThPINN", choices=["NN", "KKThPINN"])
    parser.add_argument("--model_id", type=str, default="MODELID")
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--hidden_num", type=int, default=2)
    parser.add_argument("--z0_dim", type=int, default=3)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mu", type=float, default=1)
    parser.add_argument("--max_subiter", default=500, type=int)
    parser.add_argument("--eta", default=0.8, type=float)
    parser.add_argument("--sigma", default=2, type=float)
    parser.add_argument("--mu_safe", default=1e9, type=float)
    parser.add_argument("--dtype", default=64, type=int)
    parser.add_argument("--dataset_type", type=str, default="cstr")
    parser.add_argument("--dataset_path", type=str, default="data.csv")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--job", type=str, required=True, choices=["train", "experiment", "projection_check"])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--run", type=int, default=0)
    return parser.parse_args()


def main(args):
    if args.dtype == 64:
        torch.set_default_dtype(torch.float64)
    elif args.dtype == 32:
        torch.set_default_dtype(torch.float32)
    else:
        raise ValueError("--dtype must be 32 or 64")
    
    args.loss_type = "MSE"
    data = LoadData(args)

    if args.job == "train":
        run_training(args, data)
        return

    if args.job == "experiment":
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        eval_start = time.perf_counter()
        scores = evaluate_model(data, args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"Evaluation time: {time.perf_counter() - eval_start:.4f} s")
        print(scores)
        return

    raise ValueError("projection_check is handled by diagnostics.projection_check")


if __name__ == "__main__":
    main(add_arguments())
