# compare_violations.py
import torch
import numpy as np
from utils import LoadData, get_violation
from train import load_weights
from models import NN, NNOPT

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float64)

# ---- set these to match your run ----
class Args:  # minimal args object
    model = "KKThPINN"          # "NN" or "KKThPINN"
    model_id = "MODELID"        # e.g., "MODELID"
    dataset_type = "cstr"
    dataset_path = "./data.csv"
    input_dim = 1
    hidden_dim = 32
    hidden_num = 2
    z0_dim = 5
    val_ratio = 0.2
    batch_size = 16
    lr = 1e-4
    loss_type = "MSE"
    run = 0
    dtype = 64                 # you trained with float64

args = Args()

def build_model(args, data):
    if args.model == 'NN':
        model = NN(args.input_dim, args.hidden_dim, args.hidden_num, args.z0_dim)
    else:
        model = NNOPT(args.input_dim, args.hidden_dim, args.hidden_num, args.z0_dim,
                      data['A'], data['B'], data['b'])
    return model.double().to(device)

@torch.no_grad()
def split_mean_abs_violation(loader, data, model):
    tot = 0.0
    n = 0
    for X, Y in loader:
        X = X.to(device); Y = Y.to(device)
        pred = model(X)
        v = get_violation(args, data, X, pred)          # SAME as training
        tot += torch.abs(v.view(-1)).mean().item()
        n += 1
    return tot / max(n, 1)

def main():
    # Load data & model
    data = LoadData(args)
    model = build_model(args, data)
    load_weights(model, args.model_id, args)
    model.eval()

    # 1) Global mean abs violation on each split (exactly like training curves)
    train_v = split_mean_abs_violation(data['train_loader'], data, model)
    val_v   = split_mean_abs_violation(data['val_loader'],   data, model)
    test_v  = split_mean_abs_violation(data['test_loader'],  data, model)
    print(f"[GLOBAL mean |violation|] train={train_v:.6e}  val={val_v:.6e}  test={test_v:.6e}")

    # 2) “Your plot” style but averaged like above for 503–600 K slice
    #    Build inputs in *scaled* space exactly as in your residual code.
    import pickle, os
    with open(os.path.join(os.getcwd(), 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    T = np.linspace(503, 600, 200)
    X4 = np.column_stack([T, np.zeros_like(T), np.zeros_like(T),
                          np.zeros_like(T), np.zeros_like(T), np.zeros_like(T)])
    X4_scaled = scaler.transform(X4)
    x_scaled = torch.tensor(X4_scaled[:, :1], dtype=torch.float64, device=device)

    z_scaled = model(x_scaled)
    # compute violation with the SAME function and SAME (scaled) A,B,b
    v_slice = get_violation(args, data, x_scaled, z_scaled)     # shape: (m, N)
    slice_mean_abs = torch.abs(v_slice.view(-1)).mean().item()
    print(f"[503–600 K mean |violation|] {slice_mean_abs:.6e}")

if __name__ == "__main__":
    main()
