from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)

BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------
# Generate data for z = x*y
# --------------------------------------------------
N_train = 5000
N_test = 1000

x_train = -10 + 20 * torch.rand(N_train, 1)
y_train = 10 * torch.rand(N_train, 1)
z_train = x_train * y_train

x_test = -10 + 20 * torch.rand(N_test, 1)
y_test = 10 * torch.rand(N_test, 1)
z_test = x_test * y_test

# --------------------------------------------------
# Scale inputs and output
# x in [-10,10] -> [-1,1]
# y in [0,10]   -> [0,1]
# z in [-100,100] -> [-1,1]
# --------------------------------------------------
X_train = torch.cat((x_train / 10.0, y_train / 10.0), dim=1)
Z_train = z_train / 100.0

X_test = torch.cat((x_test / 10.0, y_test / 10.0), dim=1)
Z_test = z_test / 100.0

train_loader = DataLoader(
    TensorDataset(X_train, Z_train),
    batch_size=256,
    shuffle=True
)

# --------------------------------------------------
# ReLU neural network
# --------------------------------------------------
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# --------------------------------------------------
# Train
# --------------------------------------------------
epochs = 300

for epoch in range(epochs):
    model.train()

    for X_batch, Z_batch in train_loader:
        optimizer.zero_grad()

        pred = model(X_batch)
        loss = criterion(pred, Z_batch)

        loss.backward()
        optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}: loss = {loss.item():.6e}")

# --------------------------------------------------
# Test in ORIGINAL units
# --------------------------------------------------
model.eval()

with torch.no_grad():
    pred_scaled = model(X_test)

    z_pred = pred_scaled * 100.0

    rmse = torch.sqrt(torch.mean((z_pred - z_test)**2))
    mae = torch.mean(torch.abs(z_pred - z_test))
    max_error = torch.max(torch.abs(z_pred - z_test))

print("\nTest results")
print("RMSE:", rmse.item())
print("MAE:", mae.item())
print("Maximum absolute error:", max_error.item())

# --------------------------------------------------
# Save model
# --------------------------------------------------
model_path = BASE_DIR / "bilinear_nn.pt"
torch.save(model, model_path)

print(f"\nSaved: {model_path}")

test_points = torch.tensor([
    [2.0, 3.0],
    [-4.0, 5.0],
    [8.0, 7.0],
    [-2.0, 9.0]
])

x = test_points[:, 0:1]
y = test_points[:, 1:2]

X_scaled = torch.cat((x / 10.0, y / 10.0), dim=1)

with torch.no_grad():
    pred = model(X_scaled) * 100.0

for i in range(len(test_points)):
    exact = (x[i] * y[i]).item()
    predicted = pred[i].item()
    print(
        f"x={x[i].item():.1f}, y={y[i].item():.1f}, "
        f"exact={exact:.4f}, NN={predicted:.4f}, "
        f"error={predicted-exact:.4f}"
    )