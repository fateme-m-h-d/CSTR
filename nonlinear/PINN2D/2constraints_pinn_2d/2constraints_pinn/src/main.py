import argparse
import random
import time

import numpy as np
import torch

from utils import LoadData
from train import evaluate_model, run_training


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="KKThPINN",
        choices=["NN", "PINN", "KKThPINN"],
    )
    parser.add_argument("--model_id", type=str, default="MODELID")
    parser.add_argument("--input_dim", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--hidden_num", type=int, default=2)
    parser.add_argument("--z0_dim", type=int, default=3)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mu_rxn", type=float, default=0.0)
    parser.add_argument("--mu_mb", type=float, default=0.0)
    parser.add_argument("--max_subiter", default=500, type=int)
    parser.add_argument("--eta", default=0.8, type=float)
    parser.add_argument("--sigma", default=2, type=float)
    parser.add_argument("--mu_safe", default=1e9, type=float)
    parser.add_argument("--dtype", default=64, type=int) #could change to 32 for float32
    parser.add_argument("--dataset_type", type=str, default="cstr")
    parser.add_argument("--dataset_path", type=str, default="data.csv")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument(
        "--job", type=str, required=True, choices=["train", "experiment"]
    )
    parser.add_argument(
        "--eval_split", type=str, default="test", choices=["val", "test"]
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Defaults to --run so repetitions are distinct and reproducible.",
    )
    return parser.parse_args()


def main(args):
    seed = args.run if args.seed is None else args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if args.model == "PINN":
        args.loss_type = "PINN"
    else:
        args.loss_type = "MSE"

    data = LoadData(args)

    if args.job == "train":
        run_training(args, data)
        return

    if args.job == "experiment":
        print(f"\n\nEvaluating {args.model} at run {args.run}")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        eval_start = time.perf_counter()
        scores = evaluate_model(data, args, split=args.eval_split)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        eval_time = time.perf_counter() - eval_start
        print(f"Evaluation time : {eval_time:.4f} s")
        print(scores)


if __name__ == "__main__":
    arguments = add_arguments()
    print(arguments)
    main(arguments)
