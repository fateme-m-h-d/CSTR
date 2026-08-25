import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"

# The original repository uses imports such as
#     from train import ...
#     from utils import ...
# rather than package-relative imports.
#
# Adding src/ here lets us reuse those files without changing
# the existing repository import structure.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train import (
    run_training,
    validation_original_nonlinear,
)

from utils import LoadData


# ============================================================
# Reproducibility
# ============================================================

def set_training_seed(seed):
    """
    Reset all random-number generators before every Optuna
    trial.

    This makes the network initialization and training-data
    shuffling comparable across different partition ratios.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Run linearization
# ============================================================

def run_linearization(
    nT_regions,
    nC_regions,
    ratio_T,
    ratio_C,
    reverse_T=False,
    reverse_C=False,
):
    """
    Generate region_edges.npz, lin_params.csv, and
    ABb_matrices.csv for one proposed pair of ratios.
    """

    command = [
        sys.executable,
        "-m",
        "src.linearization",

        "--nT_regions",
        str(nT_regions),

        "--nC_regions",
        str(nC_regions),

        "--ratio_T",
        str(ratio_T),

        "--ratio_C",
        str(ratio_C),
    ]

    if reverse_T:
        command.append("--reverse_T")

    if reverse_C:
        command.append("--reverse_C")

    subprocess.run(
        command,
        cwd=BASE_DIR,
        check=True,
    )


# ============================================================
# Training arguments
# ============================================================

def build_training_args(args):
    """
    Create the argument object expected by the existing
    run_training() and LoadData() functions.
    """

    training_args = argparse.Namespace()

    # Model
    training_args.model = "KKThPINN"
    training_args.model_id = "HPO"
    training_args.input_dim = 2
    training_args.hidden_dim = args.hidden_dim
    training_args.hidden_num = args.hidden_num
    training_args.z0_dim = 3

    # Training
    training_args.optimizer = args.optimizer
    training_args.epochs = args.epochs
    training_args.batch_size = args.batch_size
    training_args.lr = args.lr

    # Existing loss settings
    training_args.loss_type = "MSE"
    training_args.mu = 1.0
    training_args.max_subiter = 500
    training_args.eta = 0.8
    training_args.sigma = 2.0
    training_args.mu_safe = 1.0e9

    # Precision
    training_args.dtype = args.dtype

    # Data
    training_args.dataset_type = "cstr"
    training_args.dataset_path = str(
        BASE_DIR / "data.csv"
    )

    training_args.val_ratio = args.val_ratio

    # Required by existing LoadData()
    training_args.job = "train"

    # Keep same run index for all HPO trials.
    # HPO evaluates the returned in-memory model, not
    # the saved checkpoint.
    training_args.run = 0
    training_args.runs = 1

    return training_args


# ============================================================
# Dimension-general ratio proposal
# ============================================================

def suggest_ratios(
    trial,
    dimension_names,
    ratio_min,
    ratio_max,
):
    """
    Dimension-general HPO component.

    For d input dimensions, this creates exactly d
    hyperparameters:

        ratio_0, ratio_1, ..., ratio_(d-1)

    Here the names are T and C, so the parameters become

        ratio_T
        ratio_C
    """

    ratios = {}

    for name in dimension_names:

        ratios[name] = trial.suggest_float(
            f"ratio_{name}",
            ratio_min,
            ratio_max,
        )

    return ratios


# ============================================================
# Objective
# ============================================================

def objective(
    trial,
    args,
):
    """
    One Optuna trial:

        ratios
          ↓
        linearization
          ↓
        LoadData
          ↓
        train final-epoch KKThPINN
          ↓
        validation original nonlinear violation
          ↓
        Optuna objective
    """

    # --------------------------------------------------------
    # 1. Propose one ratio per input dimension
    # --------------------------------------------------------

    ratios = suggest_ratios(
        trial=trial,
        dimension_names=["T", "C"],
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
    )

    ratio_T = ratios["T"]
    ratio_C = ratios["C"]

    print("\n" + "=" * 70)
    print(
        f"Trial {trial.number}: "
        f"ratio_T={ratio_T:.8f}, "
        f"ratio_C={ratio_C:.8f}"
    )
    print("=" * 70)

    try:

        # ----------------------------------------------------
        # 2. Generate PL partition/linearization
        # ----------------------------------------------------

        run_linearization(
            nT_regions=args.nT_regions,
            nC_regions=args.nC_regions,
            ratio_T=ratio_T,
            ratio_C=ratio_C,
            reverse_T=args.reverse_T,
            reverse_C=args.reverse_C,
        )

        # ----------------------------------------------------
        # 3. Reset seed
        #
        # Every ratio therefore starts from the same NN
        # initialization and the same stochastic training
        # sequence.
        # ----------------------------------------------------

        set_training_seed(args.training_seed)

        # ----------------------------------------------------
        # 4. Existing training framework
        # ----------------------------------------------------

        training_args = build_training_args(args)

        torch.set_default_dtype(
            torch.float64
            if training_args.dtype == 64
            else torch.float32
        )

        data = LoadData(training_args)

        model = run_training(
            training_args,
            data,
        )

        # IMPORTANT:
        #
        # model here is the FINAL in-memory model after
        # args.epochs epochs.
        #
        # We are NOT loading the checkpoint.
        # Therefore the old checkpoint behavior does not
        # change the HPO comparison.
        # ----------------------------------------------------

        # ----------------------------------------------------
        # 5. HPO objective:
        # original nonlinear REACTION violation on VAL set
        # ----------------------------------------------------

        val_nl_violation = (
            validation_original_nonlinear(
                model=model,
                data=data,
                constraint_index=0,
            )
        )

        print(
            "\nValidation original nonlinear "
            f"violation = {val_nl_violation:.12e}"
        )

        # Store useful information in the Optuna trial.
        trial.set_user_attr(
            "nT_regions",
            args.nT_regions,
        )

        trial.set_user_attr(
            "nC_regions",
            args.nC_regions,
        )

        trial.set_user_attr(
            "num_regions",
            args.nT_regions * args.nC_regions,
        )

        return float(val_nl_violation)

    except (
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
        np.linalg.LinAlgError,
    ) as error:

        print(
            f"\nTrial {trial.number} failed:"
        )
        print(error)

        raise optuna.TrialPruned(
            str(error)
        )


# ============================================================
# Save study
# ============================================================

def save_study_results(
    study,
    args,
):
    """
    Save all trials and the best ratio combination.
    """

    # trials_path = (
    #     BASE_DIR
    #     / "ratio_tuning_trials.csv"
    # )

    # best_path = (
    #     BASE_DIR
    #     / "best_ratios.json"
    # )
    tag = f"{args.nT_regions}x{args.nC_regions}"

    trials_path = BASE_DIR / f"ratio_tuning_trials_{tag}.csv"
    best_path = BASE_DIR / f"best_ratios_{tag}.json"

    trials_df = study.trials_dataframe(
        attrs=(
            "number",
            "value",
            "params",
            "state",
        )
    )

    trials_df.to_csv(
        trials_path,
        index=False,
    )

    best_result = {
        "nT_regions": args.nT_regions,
        "nC_regions": args.nC_regions,

        "ratio_T": float(
            study.best_params["ratio_T"]
        ),

        "ratio_C": float(
            study.best_params["ratio_C"]
        ),

        "validation_original_nonlinear_violation":
            float(study.best_value),

        "training_seed": args.training_seed,
        "epochs": args.epochs,
        "ratio_min": args.ratio_min,
        "ratio_max": args.ratio_max,
    }

    with open(
        best_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            best_result,
            file,
            indent=4,
        )

    print(
        f"\nSaved: {trials_path.name}"
    )

    print(
        f"Saved: {best_path.name}"
    )


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Tune geometric PL partition ratios using "
            "validation original nonlinear constraint "
            "violation."
        )
    )

    # --------------------------------------------------------
    # Fixed number of segments
    # --------------------------------------------------------

    parser.add_argument(
        "--nT_regions",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--nC_regions",
        type=int,
        default=3,
    )

    # --------------------------------------------------------
    # Optuna
    # --------------------------------------------------------

    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--ratio_min",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--ratio_max",
        type=float,
        default=1.00,
    )

    parser.add_argument(
        "--optuna_seed",
        type=int,
        default=1234,
    )

    # --------------------------------------------------------
    # Direction of shrinking
    # --------------------------------------------------------

    parser.add_argument(
        "--reverse_T",
        action="store_true",
    )

    parser.add_argument(
        "--reverse_C",
        action="store_true",
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1.0e-4,
    )

    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--hidden_num",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
    )

    parser.add_argument(
        "--dtype",
        type=int,
        choices=[32, 64],
        default=64,
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.20,
    )

    # Same NN initialization/training RNG for every ratio.
    parser.add_argument(
        "--training_seed",
        type=int,
        default=0,
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    if not (
        0.0 < args.ratio_min
        <= args.ratio_max
        <= 1.0
    ):
        raise ValueError(
            "Require "
            "0 < ratio_min <= ratio_max <= 1."
        )

    if args.n_trials < 1:
        raise ValueError(
            "n_trials must be >= 1."
        )

    # --------------------------------------------------------
    # Reproducible Optuna search
    # --------------------------------------------------------

    sampler = optuna.samplers.TPESampler(
        seed=args.optuna_seed,
    )

    storage_path = BASE_DIR / (
        f"ratio_tuning_{args.nT_regions}x{args.nC_regions}.db"
    )

    study_name = (
        f"PL_KKT_hPINN_ratio_tuning_"
        f"{args.nT_regions}x{args.nC_regions}"
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=study_name,
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
    )

    # --------------------------------------------------------
    # Always evaluate the old uniform partition first.
    #
    # ratio_T = 1
    # ratio_C = 1
    #
    # This gives us a direct baseline inside the same HPO run.
    # --------------------------------------------------------

    if len(study.trials) == 0:
        study.enqueue_trial(
            {
                "ratio_T": 1.0,
                "ratio_C": 1.0,
            }
        )

    # --------------------------------------------------------
    # Run HPO
    # --------------------------------------------------------

    finished_trials = [
    trial
    for trial in study.trials
    if trial.state.is_finished()
    ]

    existing_trials = len(finished_trials)

    remaining_trials = max(
        0,
        args.n_trials - existing_trials,
    )

    print(
        f"Existing trials: {existing_trials}"
    )

    print(
        f"Target total trials: {args.n_trials}"
    )

    print(
        f"Remaining trials: {remaining_trials}"
    )

    if remaining_trials > 0:
        study.optimize(
            lambda trial: objective(
                trial,
                args,
            ),
            n_trials=remaining_trials,
        )
    else:
        print(
            "Requested number of trials already completed."
        )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("=" * 70)

    print(
        f"Best validation nonlinear violation: "
        f"{study.best_value:.12e}"
    )

    print(
        f"Best ratio_T: "
        f"{study.best_params['ratio_T']:.8f}"
    )

    print(
        f"Best ratio_C: "
        f"{study.best_params['ratio_C']:.8f}"
    )

    save_study_results(
        study,
        args,
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # The files currently on disk correspond to the LAST
    # Optuna trial, which may not be the best trial.
    #
    # Therefore regenerate the PL artifacts using the BEST
    # ratios before exiting.
    # --------------------------------------------------------

    print(
        "\nRegenerating linearization using "
        "the best ratios..."
    )

    run_linearization(
        nT_regions=args.nT_regions,
        nC_regions=args.nC_regions,

        ratio_T=study.best_params[
            "ratio_T"
        ],

        ratio_C=study.best_params[
            "ratio_C"
        ],

        reverse_T=args.reverse_T,
        reverse_C=args.reverse_C,
    )

    print(
        "\nThe current region_edges.npz, "
        "lin_params.csv, and ABb_matrices.csv "
        "now correspond to the BEST ratios."
    )


if __name__ == "__main__":
    main()