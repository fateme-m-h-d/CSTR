import argparse
import time

import torch

from utils import LoadData
from train import evaluate_model, run_training


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="KKThPINN")
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--input_dim", type=int, default=2)
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
    parser.add_argument("--dtype", default=64, type=int, choices=[32, 64])
    parser.add_argument("--dataset_type", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--job", type=str)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.0,
        help="Gaussian noise std as a fraction of each training output std.",
    )

    parser.add_argument(
        "--noise_seed",
        type=int,
        default=1234,
        help="Random seed used to generate training-data noise.",
    )
    return parser.parse_args()


def main(args):
    
    torch.set_default_dtype(
    torch.float64
    if args.dtype == 64
    else torch.float32
    )
    
    args.run = 0

    if args.job == "train":
        args.loss_type = "MSE"
        data = LoadData(args)
        run_training(args, data)
        return

    if args.job == "experiment":
        print(f"\n\nEvaluating {args.model} at run {args.run}")
        data = LoadData(args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        eval_start = time.perf_counter()
        scores = evaluate_model(data, args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        eval_time = time.perf_counter() - eval_start
        print(f"Evaluation time : {eval_time:.4f} s")
        print(scores)


if __name__ == "__main__":
    arguments = add_arguments()
    print(arguments)
    main(arguments)
