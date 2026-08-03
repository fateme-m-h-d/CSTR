import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import math
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from enforce.core.model import ENFORCE, ENFORCEConfig
from enforce.data.data_utils import scale_data
from enforce.engines.evaluate import EvaluationConfig, Evaluator
from enforce.engines.train import Trainer, TrainingConfig


# ============================================================
# 1. Reproducibility and device
# ============================================================

SEED = 42
DEVICE = torch.device("cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"Using device: {DEVICE}")
print(f"CUDA visible to this process: {torch.cuda.is_available()}")


# ============================================================
# 2. CSTR constants
# ============================================================

TAU = 10.0
CBO = 2.0
CCO = 0.0

T_ISO = 350.0
R = 8.314

AFO = 1.0e13
EAF = 90000.0

ARO = 1.0e11
EAR = 80000.0

KF = AFO * math.exp(-EAF / (R * T_ISO))
KR = ARO * math.exp(-EAR / (R * T_ISO))

print(f"kf = {KF:.10e}")
print(f"kr = {KR:.10e}")


# ============================================================
# 3. Load data
# ============================================================

df = pd.read_csv("data.csv")

required_columns = ["Cao", "Ca", "Cb", "Cc"]
missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in data.csv: {missing_columns}. "
        f"Expected columns: {required_columns}"
    )

# Input: Cao
X = df[["Cao"]].to_numpy(dtype=np.float32)

# Outputs: Ca, Cb, Cc
Y = df[["Ca", "Cb", "Cc"]].to_numpy(dtype=np.float32)


# ============================================================
# 4. Train/test split
# ============================================================

# This preserves the 80/20 split from the code you sent.
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=SEED,
    shuffle=True,
)


# ============================================================
# 5. Scale data
# ============================================================

(
    X_train_scaled,
    Y_train_scaled,
    X_test_scaled,
    Y_test_scaled,
    scaling_parameters,
) = scale_data(
    X_train,
    Y_train,
    X_test,
    Y_test,
)


# ============================================================
# 6. Define the physical equality constraints
# ============================================================

def cstr_constraints(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    ENFORCE passes physical, unscaled inputs and outputs here.

    x shape: [batch_size, 1]
        x[:, 0] = Cao

    y shape: [batch_size, 3]
        y[:, 0] = Ca
        y[:, 1] = Cb
        y[:, 2] = Cc

    Returns
    -------
    Tensor of shape [batch_size, 2]:
        column 0 = g1, nonlinear reaction balance
        column 1 = g2, overall material balance
    """

    cao = x[:, 0]

    ca = y[:, 0]
    cb = y[:, 1]
    cc = y[:, 2]

    # Nonlinear reaction constraint:
    # Cao - Ca - kf*Ca*Cb^2*tau + kr*Cc*tau = 0
    g1 = (
        cao
        - ca
        - KF * ca * cb.pow(2) * TAU
        + KR * cc * TAU
    )

    # Overall material-balance constraint:
    # Cao - Ca + Cbo - Cb + Cco - Cc = 0
    g2 = (
        cao
        - ca
        + CBO
        - cb
        + CCO
        - cc
    )

    return torch.stack((g1, g2), dim=1)


# ============================================================
# 7. Scaling tensors required by ENFORCE
# ============================================================

scaling_input = (
    torch.as_tensor(
        scaling_parameters["input_mean"],
        dtype=torch.float32,
        device=DEVICE,
    ),
    torch.as_tensor(
        scaling_parameters["input_std"],
        dtype=torch.float32,
        device=DEVICE,
    ),
)

scaling_output = (
    torch.as_tensor(
        scaling_parameters["output_mean"],
        dtype=torch.float32,
        device=DEVICE,
    ),
    torch.as_tensor(
        scaling_parameters["output_std"],
        dtype=torch.float32,
        device=DEVICE,
    ),
)


# ============================================================
# 8. Configure ENFORCE
# ============================================================

model_config = ENFORCEConfig(
    input_neurons=1,
    output_neurons=3,

    hidden_neurons=64,
    hidden_layers=2,

    # Mean residual tolerance used during training.
    training_tolerance=1.0e-5,

    # Maximum residual tolerance used during inference.
    inference_tolerance=1.0e-8,

    max_it=100,

    supervised=True,

    # Penalizes movement from raw NN output to projected output.
    weight_loss_displacement=0.5,

    # Apply projection from the first epoch.
    epoch_start_hard_constrained=0,

    # Skip projection during training when it worsens prediction loss.
    ada_np_auto_activation=True,

    random_seed=SEED,
)


# weighting_option=1 means an ordinary Euclidean projection.
model = ENFORCE(
    scaling_input=scaling_input,
    scaling_output=scaling_output,
    c=cstr_constraints,
    config=model_config,
    constrained=True,
    weighting_option=1,
)

# Explicit CPU placement. ENFORCE stores the scaling quantities as
# ordinary tensor attributes, so move them explicitly as well.
model = model.to(DEVICE)
model.device = "cpu"

model.mean_input = model.mean_input.to(DEVICE)
model.std_input = model.std_input.to(DEVICE)
model.mean_output = model.mean_output.to(DEVICE)
model.std_output = model.std_output.to(DEVICE)


# ============================================================
# 9. Convert data to CPU tensors
# ============================================================

X_train_tensor = torch.as_tensor(
    X_train_scaled,
    dtype=torch.float32,
    device=DEVICE,
)

Y_train_tensor = torch.as_tensor(
    Y_train_scaled,
    dtype=torch.float32,
    device=DEVICE,
)

X_test_tensor = torch.as_tensor(
    X_test_scaled,
    dtype=torch.float32,
    device=DEVICE,
)

Y_test_tensor = torch.as_tensor(
    Y_test_scaled,
    dtype=torch.float32,
    device=DEVICE,
)


# ============================================================
# 10. Verify all components are on CPU
# ============================================================

print("\nDevice check")
print(f"Model parameters: {next(model.parameters()).device}")
print(f"mean_input:       {model.mean_input.device}")
print(f"std_input:        {model.std_input.device}")
print(f"mean_output:      {model.mean_output.device}")
print(f"std_output:       {model.std_output.device}")
print(f"Training X:       {X_train_tensor.device}")
print(f"Training Y:       {Y_train_tensor.device}")

devices = {
    str(next(model.parameters()).device),
    str(model.mean_input.device),
    str(model.std_input.device),
    str(model.mean_output.device),
    str(model.std_output.device),
    str(X_train_tensor.device),
    str(Y_train_tensor.device),
    str(X_test_tensor.device),
    str(Y_test_tensor.device),
}

if devices != {"cpu"}:
    raise RuntimeError(
        f"CPU setup failed. Found tensors on these devices: {devices}"
    )


# ============================================================
# 11. Train
# ============================================================

training_config = TrainingConfig(
    batch_size=64,
    epochs=2000,
    learning_rate=1.0e-3,
    random_seed=SEED,
)

trainer = Trainer(
    model=model,
    config=training_config,
)

model = trainer.fit(
    X_train_tensor,
    Y_train_tensor,
)


# ============================================================
# 12. Evaluate
# ============================================================

# Do not place this inside torch.no_grad().
# ENFORCE needs autograd to compute dc/dy during projection.
evaluator = Evaluator(
    model,
    EvaluationConfig(),
)

result = evaluator.evaluate(
    X_test_tensor,
    Y_test_tensor,
    scaling_parameters,
)

# Evaluator returns unscaled NumPy predictions.
predictions = np.asarray(result.predictions, dtype=np.float64)

if predictions.ndim != 2 or predictions.shape[1] != 3:
    raise RuntimeError(
        "Expected ENFORCE predictions with shape [N, 3], "
        f"but received {predictions.shape}."
    )

ca_pred = predictions[:, 0]
cb_pred = predictions[:, 1]
cc_pred = predictions[:, 2]

# Use float64 for post-processing metrics.
cao_test = X_test[:, 0].astype(np.float64)
Y_test_metric = Y_test.astype(np.float64)


# ============================================================
# 13. RMSE
# ============================================================

rmse_ca = np.sqrt(
    np.mean((ca_pred - Y_test_metric[:, 0]) ** 2)
)
rmse_cb = np.sqrt(
    np.mean((cb_pred - Y_test_metric[:, 1]) ** 2)
)
rmse_cc = np.sqrt(
    np.mean((cc_pred - Y_test_metric[:, 2]) ** 2)
)

rmse_overall = np.sqrt(
    np.mean((predictions - Y_test_metric) ** 2)
)


# ============================================================
# 14. Constraint violations in physical units
# ============================================================

g1 = (
    cao_test
    - ca_pred
    - KF * ca_pred * cb_pred**2 * TAU
    + KR * cc_pred * TAU
)

g2 = (
    cao_test
    - ca_pred
    + CBO
    - cb_pred
    + CCO
    - cc_pred
)

g1_mean_abs = np.mean(np.abs(g1))
g2_mean_abs = np.mean(np.abs(g2))

g1_max_abs = np.max(np.abs(g1))
g2_max_abs = np.max(np.abs(g2))

all_residuals = np.column_stack((g1, g2))

overall_violation = np.mean(np.abs(all_residuals))
worst_violation = np.max(np.abs(all_residuals))


# ============================================================
# 15. Print results
# ============================================================

print("\n========== ENFORCE CSTR RESULTS ==========")

print("\nRMSE")
print(f"Ca RMSE:       {rmse_ca:.10e}")
print(f"Cb RMSE:       {rmse_cb:.10e}")
print(f"Cc RMSE:       {rmse_cc:.10e}")
print(f"Overall RMSE:  {rmse_overall:.10e}")

print("\nMean absolute constraint violation")
print(f"g1 violation:  {g1_mean_abs:.10e}")
print(f"g2 violation:  {g2_mean_abs:.10e}")
print(f"Overall:       {overall_violation:.10e}")

print("\nWorst-case absolute constraint violation")
print(f"max |g1|:      {g1_max_abs:.10e}")
print(f"max |g2|:      {g2_max_abs:.10e}")
print(f"overall max:   {worst_violation:.10e}")

print("\nENFORCE evaluator metrics")
for key, value in result.metrics.items():
    print(f"{key}: {value}")


# ============================================================
# 16. Save predictions
# ============================================================

output_df = pd.DataFrame(
    {
        "Cao": cao_test,
        "Ca_true": Y_test_metric[:, 0],
        "Cb_true": Y_test_metric[:, 1],
        "Cc_true": Y_test_metric[:, 2],
        "Ca_pred": ca_pred,
        "Cb_pred": cb_pred,
        "Cc_pred": cc_pred,
        "g1": g1,
        "g2": g2,
        "abs_g1": np.abs(g1),
        "abs_g2": np.abs(g2),
    }
)

output_df.to_csv(
    "enforce_cstr_predictions.csv",
    index=False,
)

print("\nSaved: enforce_cstr_predictions.csv")
