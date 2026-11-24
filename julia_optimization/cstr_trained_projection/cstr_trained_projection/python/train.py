"""Train each architecture through its projection on one saved data split."""
import argparse
import copy
import csv
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from cstr_models import (ProjectedModel, default_config, make_backbone, physics,
                         reference_numpy, region_rows)

ROOT = Path(__file__).resolve().parents[1]


def prepare(data_file, output, cfg, seed):
    data = np.genfromtxt(data_file, delimiter=",", skip_header=1)
    if data.ndim != 2 or data.shape[1] != 5 or not np.isfinite(data).all():
        raise ValueError("Expected finite T,Cao,Ca,Cb,Cc columns.")
    reference = reference_numpy(data[:, :2], cfg)
    if np.max(np.abs(reference-data[:, 2:])) > 1e-5:
        raise ValueError("Data do not match the selected reverse activation energy and reference equations.")
    permutation = np.random.RandomState(seed).permutation(len(data))
    ntrain, nval = int(0.6*len(data)), int(0.2*len(data))
    split = dict(train=permutation[:ntrain].tolist(),
                 validation=permutation[ntrain:ntrain+nval].tolist(),
                 test=permutation[ntrain+nval:].tolist())
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(data_sha256=hashlib.sha256(Path(data_file).read_bytes()).hexdigest(),
                    indices=split, split_seed=seed)
    protocol_path = output/"split.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("This output folder already has a different data split. Choose another output folder.")
    protocol_path.write_text(json.dumps(protocol, indent=2))
    cfg["input_scale"] = np.max(np.abs(data[:, :2]), axis=0).tolist()
    cfg["output_scale"] = np.max(np.abs(data[:, 2:]), axis=0).tolist()
    return torch.tensor(data[:, :2]), torch.tensor(data[:, 2:]), split


def train_one(method, args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = default_config(method, args.ear, args.steps)
    cfg.update(activation=args.activation, hidden=args.hidden)
    cfg.update(epochs=args.epochs, learning_rate=args.lr, batch_size=args.batch_size,
               model_seed=args.seed, split_seed=args.split_seed, dtype="float64")
    cfg["warmup_epochs"] = min(args.warmup, max(0, args.epochs-1)) if method in ("hardnet", "enforce") else 0
    cfg["displacement_weight"] = 0.1 if method == "enforce" else 0.0
    x, truth, split = prepare(args.data, args.output, cfg, args.split_seed)
    if method == "pl":
        cfg["regions"] = region_rows(cfg)
    model = ProjectedModel(make_backbone(cfg), cfg)
    # A shared, data-based initialization makes all backbones start at the same prediction.
    with torch.no_grad():
        model.backbone[-1].weight.mul_(0.01)
        mean = (truth[split["train"]]/model.output_scale).mean(0)
        model.backbone[-1].bias.copy_(mean)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    best_loss, best_state, history = float("inf"), None, []
    train_index = torch.tensor(split["train"])
    val_index = torch.tensor(split["validation"])
    for epoch in range(args.epochs):
        projected = epoch >= cfg["warmup_epochs"]
        model.train()
        order = train_index[torch.randperm(len(train_index), generator=generator)]
        total = 0.0
        for ids in order.split(args.batch_size):
            optimizer.zero_grad()
            raw = model.raw(x[ids])
            prediction = model.projection(x[ids], raw) if projected else raw
            loss = (((prediction-truth[ids])/model.output_scale)**2).mean()
            if projected and cfg["displacement_weight"]:
                loss = loss+cfg["displacement_weight"]*(((prediction-raw)/model.output_scale)**2).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"{method} nonfinite loss at epoch {epoch+1}. No sample was silently dropped.")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError(f"{method} nonfinite training gradient at epoch {epoch+1}.")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total += float(loss.detach())*len(ids)
        model.eval()
        with torch.no_grad():
            # Validation always uses the complete deployed projection.
            val = (((model(x[val_index])-truth[val_index])/model.output_scale)**2).mean().item()
        if not np.isfinite(val):
            raise RuntimeError(f"{method} nonfinite projected validation loss.")
        if projected and val < best_loss:
            best_loss, best_state = val, copy.deepcopy(model.state_dict())
        history.append([epoch+1, total/len(train_index), val, int(projected)])
        if (epoch+1) % 100 == 0 or epoch == args.epochs-1:
            print(method, epoch+1, "projected validation MSE", val, flush=True)
    model.load_state_dict(best_state)
    model.eval()
    directory = args.output/method
    directory.mkdir(parents=True, exist_ok=True)
    cfg["selected_projected_validation_mse"] = best_loss
    (directory/"config.json").write_text(json.dumps(cfg, indent=2))
    torch.save(model, directory/"complete.pt")
    torch.save(model.backbone, directory/"backbone.pt")
    layers = []
    for layer in model.backbone:
        if isinstance(layer, torch.nn.Linear):
            layers.append(dict(kind="linear", weight=layer.weight.detach().tolist(), bias=layer.bias.detach().tolist()))
        else:
            layers.append(dict(kind=cfg["activation"]))
    (directory/"layers.json").write_text(json.dumps(layers))
    with (directory/"history.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "training_loss", "projected_validation_mse", "projection_active_in_training"])
        writer.writerows(history)
    with torch.no_grad():
        prediction = model(x)
        raw = model.raw(x)
        errors = physics(x, prediction, cfg)
    ids = split["test"]
    report = dict(test_scaled_rmse=float((((prediction[ids]-truth[ids])/model.output_scale)**2).mean().sqrt()),
                  test_max_physical_A_residual=float(errors[ids, 0].abs().max()),
                  test_max_summed_residual=float(errors[ids, 1].abs().max()),
                  test_min_concentration=float(prediction[ids].min()),
                  checkpoint_selection="minimum projected validation MSE after warmup")
    (directory/"metrics.json").write_text(json.dumps(report, indent=2))
    # Include all data inputs plus off-grid inputs for cross-language checks.
    points = torch.cat((x, torch.tensor([[391.41, 1.0111], [382.78591974, 1.2], [447.9235, .94952]])))
    with torch.no_grad():
        parity = torch.cat((points, model.raw(points), model(points)), -1).numpy()
    np.savetxt(directory/"parity.csv", parity, delimiter=",",
               header="T,Cao,raw_A,raw_B,raw_C,projected_A,projected_B,projected_C", comments="")
    print(directory, report, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["all", "baseline", "mass", "pl", "hardnet", "enforce"], default="all")
    parser.add_argument("--data", type=Path, default=ROOT/"data/cstr.csv")
    parser.add_argument("--output", type=Path, default=ROOT/"artifacts")
    parser.add_argument("--ear", type=float, default=100000.0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--activation", choices=["tanh", "softplus", "sigmoid"], default="tanh")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    if args.epochs < 1 or args.steps < 1:
        parser.error("epochs and steps must be positive")
    methods = ["baseline", "mass", "pl", "hardnet", "enforce"] if args.method == "all" else [args.method]
    for method in methods:
        train_one(method, args)
