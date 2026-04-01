from utils import LoadData
from train import run_training, evaluate_model
import argparse
import time
import copy
import torch


def add_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='KKThPINN', help='NN, KKThPINN')
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--input_dim', type=int, default=2,
                        help='1 for cstr')
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--hidden_num', type=int, default=2)
    parser.add_argument('--z0_dim', type=int, default=3,
                        help='3 for cstr')

    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4) #changed
    parser.add_argument('--mu', type=float, default=1)
    parser.add_argument("--max_subiter", default=500, type=int)
    parser.add_argument("--eta", default=0.8, type=float)
    parser.add_argument("--sigma", default=2, type=float)
    parser.add_argument("--mu_safe", default=1e+9, type=float)
    parser.add_argument("--dtype", default=32, type=int)

    parser.add_argument('--dataset_type', type=str, help='choose from cstr')
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--job', type=str, help='choose from train, experiment')
    # parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--runs', type=int, default=1, help='Total runs if you want to loop')
    parser.add_argument('--run', type=int, default=0, help='Current run index')

    args = parser.parse_args()
    return args



def main(args):
    if args.job == 'train':
        if args.model == 'NN':
            args.loss_type = 'MSE'
        elif args.model == 'KKThPINN':
            args.loss_type = 'MSE'
        

        args.run = 0
        data = LoadData(args)
        run_training(args, data)

    elif args.job == 'experiment':
        # for i in range(args.runs):
        #     for model_name in ['NN', 'KKThPINN']:
        #         args.model = model_name
        #         if args.model == 'NN':
        #             args.loss_type = 'MSE'
        #         elif args.model == 'KKThPINN':
        #             args.loss_type = 'MSE'

        #         args.run = i
                args.run = 0
                print(f'\n\nEvaluating {args.model} at run {args.run}')
                data = LoadData(args)
                
                # # -------- training time --------
                # if torch.cuda.is_available():
                #     torch.cuda.synchronize()
                # train_start = time.perf_counter()

                # run_training(args, data)

                # if torch.cuda.is_available():
                #     torch.cuda.synchronize()
                # train_time = time.perf_counter() - train_start
                # # run_training(args, data)
                # print(f"Training time   : {train_time:.4f} s")
                # -------- evaluation time only --------
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                eval_start = time.perf_counter()

                scores = evaluate_model(data, args)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                eval_time = time.perf_counter() - eval_start
                # scores = evaluate_model(data, args)
                
                print(f"Evaluation time : {eval_time:.4f} s")
                print(scores)


if __name__ == '__main__':
    args = add_arguments()
    print(args)
    main(args)
    