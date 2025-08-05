import numpy as np
import torch
import matplotlib.pyplot as plt

# Temperature grid
x = np.linspace(280, 600, 500)          # shape (500,)
x_t = torch.tensor(x, dtype=torch.float32)
transition_points=[0.375*800, 0.425*800, 0.45*800, 0.5*800, 0.625*800]
steepness = 500000

def custom_sigmoid(x_tensor, transition_point, steepness):
    transition_point = torch.tensor(transition_point, dtype=torch.float32)  # shape (5,)
    transition_width = 100.0 / steepness
    w = (x_tensor.unsqueeze(1) - transition_point) / transition_width      # (500,5)
    return torch.sigmoid(w)    

num_regions = len(transition_points) + 1
masks = []

for i in range(num_regions):
    if i == 0:
        mask = 1.0 - custom_sigmoid(x_t, transition_points[0], steepness)
    elif i == num_regions - 1:
        mask = custom_sigmoid(x_t, transition_points[-1], steepness)
    else:
        mask = ( custom_sigmoid(x_t, transition_points[i-1], steepness) *
                 (1.0 - custom_sigmoid(x_t, transition_points[i], steepness)) )
    masks.append(mask.detach().numpy())# (500,5)

plt.figure(figsize=(9,5))
for i, m in enumerate(masks):
    plt.plot(x, m, label=f"mask {i}")
plt.xlabel('x (Temperature)')
plt.ylabel('custom_sigmoid')
plt.title('Custom sigmoids at different transition points')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
