from utils import LoadData
from train import run_training, evaluate_model
import argparse
import time
import copy


def add_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='KKThPINN', help='NN, KKThPINN')
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--input_dim', type=int, default=1,
                        help='1 for cstr')
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--hidden_num', type=int, default=2)
    parser.add_argument('--z0_dim', type=int, default=4,
                        help='4 for cstr')
    # add_arguments()
    parser.add_argument('--z0_inner_dim', type=int, default=3,
                    help='inner NN output dim (e.g., 3 for Ca,Cb,Cc)')

    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4) #changed
    parser.add_argument('--mu', type=float, default=1)
    parser.add_argument("--max_subiter", default=500, type=int)
    parser.add_argument("--eta", default=0.8, type=float)
    parser.add_argument("--sigma", default=2, type=float)
    parser.add_argument("--mu_safe", default=1e+9, type=float)
    parser.add_argument("--dtype", default=64, type=int)

    parser.add_argument('--dataset_type', type=str, help='choose from cstr')
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--job', type=str, help='choose from train, experiment')
    # parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--runs', type=int, default=1, help='Total runs if you want to loop')
    parser.add_argument('--run', type=int, default=0, help='Current run index')
    
    
    parser.add_argument('--weighted_cb_loss', action='store_true',
                    help='Use weighted Cb loss based on 2*Cb_mid')

    parser.add_argument('--cb_loss_segments', type=int, default=10,
                        help='Number of Cb segments between 0 and 1 for Cb_mid weighting')

    parser.add_argument('--cb_loss_min', type=float, default=0.0)
    parser.add_argument('--cb_loss_max', type=float, default=1.0)
    
    parser.add_argument('--z4_neg_penalty', action='store_true',
                    help='Add penalty for negative projected z4 = kr*Cb^2')

    # parser.add_argument('--beta_z4', type=float, default=0.0,
    #                 help='Weight for negative z4 penalty')
    
    
    parser.add_argument('--z4_margin_penalty', action='store_true',
                    help='Add margin penalty to keep projected z4 above a positive threshold')

    parser.add_argument('--beta_z4', type=float, default=0.0,
                        help='Weight for z4 margin penalty')

    parser.add_argument('--z4_margin', type=float, default=0.0,
                        help='Positive margin for physical z4')
    
    parser.add_argument(
        "--z4_activation",
        type=str,
        default="raw",
        choices=["raw", "softplus_eps", "relu"],
        help="Activation used for recovered h = Ca*Cb^2"
    )
    
    parser.add_argument(
        "--kkt_warmup_epochs",
        type=int,
        default=0,
        help="Number of first epochs to train without KKT projection"
    )
    
    
    parser.add_argument("--ca_weight", type=float, default=0.25)
    parser.add_argument("--cb1_weight", type=float, default=0.25)
    parser.add_argument("--cb2_weight", type=float, default=0.25)
    parser.add_argument("--cc_weight", type=float, default=0.25)
    
    
    parser.add_argument(
        "--lambda_cb_consistency",
        type=float,
        default=0.0,
        help="Penalty weight for enforcing Cb1 close to Cb2"
    )
    
    
    parser.add_argument(
        "--cao_regions_for_worst",
        type=int,
        default=90,
        help="Number of Cao regions for worst-region violation calculation"
    )
    args = parser.parse_args()
    return args



# def main(args):
#     if args.job == 'train':
#         if args.model == 'NN':
#             args.loss_type = 'MSE'
#         elif args.model == 'KKThPINN':
#             args.loss_type = 'MSE'
        

#         # args.run = 0
#         data = LoadData(args)
#         # import torch
#         # with torch.no_grad():
#         #     Xfull = []
#         #     Yfull = []
#         #     for Xb, Yb in data['test_loader']:
#         #         Xfull.append(Xb); Yfull.append(Yb)
#         #     Xfull = torch.cat(Xfull, 0)
#         #     Yfull = torch.cat(Yfull, 0)

#         #     # scaled-space residual on ground-truth labels
#         #     res = data['A'] @ Xfull.T + data['B'] @ Yfull.T - data['b'].repeat(1, Xfull.shape[0])
#         #     print("DEBUG labels residual mean|·|:", res.abs().mean().item(), "max|·|:", res.abs().max().item())

#         # from scaler_utils import scaler as sc2
#         # import numpy as np
#         # diff = np.max(np.abs(data['scaler'].scale_ - sc2.scale_))
#         # print("DEBUG scaler max|Δ|:", diff)
#         run_training(args, data)

#     elif args.job == 'experiment':
#         # for i in range(args.runs):
#         #     for model_name in ['NN', 'KKThPINN']:
#         #         args.model = model_name
#         #         if args.model == 'NN':
#         #             args.loss_type = 'MSE'
#         #         elif args.model == 'KKThPINN':
#         #             args.loss_type = 'MSE'

#         #         args.run = i
#                 # args.run = 0
#                 print(f'\n\nEvaluating {args.model} at run {args.run}')
#                 data = LoadData(args)
                
#                 import torch
#                 with torch.no_grad():
#                     Xfull = []
#                     Yfull = []
#                     for Xb, Yb in data['test_loader']:
#                         Xfull.append(Xb); Yfull.append(Yb)
#                     Xfull = torch.cat(Xfull, 0)
#                     Yfull = torch.cat(Yfull, 0)

#                     # scaled-space residual on ground-truth labels
#                     # res = data['A'] @ Xfull.T + data['B'] @ Yfull.T - data['b'].repeat(1, Xfull.shape[0])
#                     # print("DEBUG labels residual mean|·|:", res.abs().mean().item(), "max|·|:", res.abs().max().item())

#                 # from scaler_utils import scaler as sc2
#                 # import numpy as np
#                 # diff = np.max(np.abs(data['scaler'].scale_ - sc2.scale_))
#                 # print("DEBUG scaler max|Δ|:", diff)
#                 # run_training(args, data)
#                 scores = evaluate_model(data, args)
#                 print(scores)

def main(args):
    if args.model == 'NN':
        args.loss_type = 'MSE'
    elif args.model == 'KKThPINN':
        args.loss_type = 'MSE'

    base_model_id = args.model_id

    if args.job == 'train':
        for i in range(args.runs):
            args.run = i
            args.model_id = base_model_id

            print(f'\n\nTraining {args.model_id}, activation={args.z4_activation}, run {args.run}')
            data = LoadData(args)
            run_training(args, data)

    elif args.job == 'experiment':
        for i in range(args.runs):
            args.run = i
            args.model_id = base_model_id

            print(f'\n\nEvaluating {args.model_id}, activation={args.z4_activation}, run {args.run}')
            data = LoadData(args)
            scores = evaluate_model(data, args)
            print(scores)

    elif args.job == 'train_experiment':
        for i in range(args.runs):
            args.run = i
            args.model_id = base_model_id

            print(f'\n\nTraining {args.model_id}, activation={args.z4_activation}, run {args.run}')
            data = LoadData(args)
            run_training(args, data)

            print(f'\n\nEvaluating {args.model_id}, activation={args.z4_activation}, run {args.run}')
            data = LoadData(args)
            scores = evaluate_model(data, args)
            print(scores)
            
            
if __name__ == '__main__':
    args = add_arguments()
    print(args)
    main(args)
    