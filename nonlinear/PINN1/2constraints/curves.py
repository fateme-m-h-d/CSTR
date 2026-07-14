import os
# If there's no display (typical on servers), use Agg
if not os.environ.get("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- config ----------
outdir = Path("./plots")
outdir.mkdir(parents=True, exist_ok=True)

# ---------- load ----------
loss_KKThPINN = np.load('./data/learning_curves/cstr/KKThPINN/0.2/KKThPINN_n150_seg30_train_losses_run10.npy') 
val_loss_KKThPINN = np.load('./data/learning_curves/cstr/KKThPINN/0.2/KKThPINN_n150_seg30_val_losses_run10.npy')  
violation_KKThPINN = np.load('./data/learning_curves/cstr/KKThPINN/0.2/KKThPINN_n150_seg30_train_violations_run10.npy')  
val_violation_KKThPINN = np.load('./data/learning_curves/cstr/KKThPINN/0.2/KKThPINN_n150_seg30_val_violations_run10.npy') 

loss_NN = np.load('./data/learning_curves/cstr/NN/0.2/NN_n150_train_losses_run10.npy')  
val_loss_NN = np.load('./data/learning_curves/cstr/NN/0.2/NN_n150_val_losses_run10.npy')  
violation_NN = np.load('./data/learning_curves/cstr/NN/0.2/NN_n150_train_violations_run10.npy') 
val_violation_NN = np.load('./data/learning_curves/cstr/NN/0.2/NN_n150_val_violations_run10.npy') 

print(f"KKThPINN Training Loss (last epoch): {loss_KKThPINN[-1]:.6f}")
print(f"KKThPINN Validation Loss (last epoch): {val_loss_KKThPINN[-1]:.6f}")
print(f"NN Training Loss (last epoch): {loss_NN[-1]:.6f}")
print(f"NN Validation Loss (last epoch): {val_loss_NN[-1]:.6f}")

# ---------- loss plot ----------
plt.figure(figsize=(10, 5))
plt.grid(True, alpha=0.3)
plt.plot(loss_KKThPINN, 'g--', label='KKThPINN Training Loss')
plt.plot(val_loss_KKThPINN, 'g', label='KKThPINN Validation Loss')
plt.plot(loss_NN, 'b--', label='NN Training Loss')
plt.plot(val_loss_NN, 'b', label='NN Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.ylim((0, 0.1))
plt.legend()
plt.tight_layout()
plt.savefig(outdir / "loss_over_epochs.png", dpi=300)
plt.close()

# ---------- violation plot ----------
plt.figure(figsize=(10, 5))
plt.grid(True, alpha=0.3)
plt.plot(violation_KKThPINN, 'g--', label='KKThPINN Training Violation')
plt.plot(val_violation_KKThPINN, 'g', label='KKThPINN Validation Violation')
plt.plot(violation_NN, 'b--', label='NN Training Violation')
plt.plot(val_violation_NN, 'b', label='NN Validation Violation')
plt.title('Violation over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Violation')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig(outdir / "violation_over_epochs.png", dpi=300)
plt.close()

print(f"Saved plots to: {outdir.resolve()}")
