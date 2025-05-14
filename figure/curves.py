import numpy as np
import matplotlib.pyplot as plt

loss_KKThPINN = np.load('./data/learning_curves/cstr/KKThPINN/0.2/MODELID_train_losses_run0.npy') 
val_loss_KKThPINN = np.load('./data/learning_curves/cstr/KKThPINN/0.2/MODELID_val_losses_run0.npy')  
violation_KKThPINN = np.load('./data/learning_curves/cstr/KKThPINN/0.2/MODELID_train_violations_run0.npy')  
val_violation_KKThPINN = np.load('./data/learning_curves/cstr/KKThPINN/0.2/MODELID_val_violations_run0.npy') 

loss_NN = np.load('./data/learning_curves/cstr/NN/0.2/MODELID_train_losses_run0.npy')  
val_loss_NN = np.load('./data/learning_curves/cstr/NN/0.2/MODELID_val_losses_run0.npy')  
violation_NN = np.load('./data/learning_curves/cstr/NN/0.2/MODELID_train_violations_run0.npy') 
val_violation_NN = np.load('./data/learning_curves/cstr/NN/0.2/MODELID_val_violations_run0.npy') 

# Print the last epoch losses for KKThPINN
print(f"KKThPINN Training Loss (Epoch 1000): {loss_KKThPINN[-1]}")
print(f"KKThPINN Validation Loss (Epoch 1000): {val_loss_KKThPINN[-1]}")

# Print the last epoch losses for NN
print(f"NN Training Loss (Epoch 1000): {loss_NN[-1]}")
print(f"NN Validation Loss (Epoch 1000): {val_loss_NN[-1]}")

# loss_ECNN = np.load('./data/learning_curves/cstr/ECNN/0.2/MODELID_train_losses_run0.npy')  
# val_loss_ECNN = np.load('./data/learning_curves/cstr/ECNN/0.2/MODELID_val_losses_run0.npy')  
# violation_ECNN = np.load('./data/learning_curves/cstr/ECNN/0.2/MODELID_train_violations_run0.npy')  
# val_violation_ECNN = np.load('./data/learning_curves/cstr/ECNN/0.2/MODELID_val_violations_run0.npy')


# Plotting the loss curve
plt.figure(figsize=(10, 5))
plt.grid()
plt.plot(loss_KKThPINN, 'g--', label='KKThPINN Training Loss')
plt.plot(val_loss_KKThPINN, 'g', label='KKThPINN Validation Loss')

# plt.plot(loss_ECNN, 'r--', label='ECNN Training Loss')
# plt.plot(val_loss_ECNN, 'r', label='ECNN Validation Loss')

# plt.plot(loss_NN, 'b--', label=' NN Training Loss')
# plt.plot(val_loss_NN, 'b', label='NN Validation Loss')

plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.ylim((0, 0.03))
plt.legend()
plt.show()


# Plotting the accuracy curve
plt.figure(figsize=(10, 5))
plt.plot(violation_KKThPINN, 'g--', label='KKThPINN Training violation')
plt.plot(val_violation_KKThPINN, 'g', label='KKThPINN Validation violation')

# plt.plot(violation_ECNN, 'r--', label='ECNN Training violation')
# plt.plot(val_violation_ECNN, 'r', label='ECNN Validation violation')

# plt.plot(violation_NN, 'b--', label='NN Training violation')
# plt.plot(val_violation_NN, 'b', label='NN Validation violation')

plt.title('violation over Epochs')
plt.xlabel('Epochs')
plt.ylabel('violation')
plt.yscale('log')
plt.legend()
plt.show()