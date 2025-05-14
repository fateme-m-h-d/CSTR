import torch
from models import NN, NNOPT, ECNN
import pandas as pd

input_dim = 1
hidden_dim = 32
hidden_num = 2
z0_dim = 2

#Linearized parameters
coeff = pd.read_csv("/Users/simonnguyen/Library/CloudStorage/OneDrive-UniversityofWaterloo/Undergrad Terms UW/4A/CHE 499/CSTR_example/2024-11-23/coefficients.csv").to_numpy()[0]
a,c,d,e,f,g,h,i = -coeff[0],-coeff[1],-coeff[2],coeff[3],-coeff[4],-coeff[5],-coeff[6],coeff[7]
A = torch.tensor([[d], [h]])
B = torch.tensor([[a, c], [f, g]])
b = torch.tensor([e, i])

device = "cuda" if torch.cuda.is_available() else "cpu"
NN_path = '/Users/simonnguyen/Library/CloudStorage/OneDrive-UniversityofWaterloo/Undergrad Terms UW/4A/CHE 499/Fateme File/models/cstr/NN/0.2/MODELID_0.2_1.pth'
KKT_path = '/Users/simonnguyen/Library/CloudStorage/OneDrive-UniversityofWaterloo/Undergrad Terms UW/4A/CHE 499/Fateme File/models/cstr/KKThPINN/0.2/MODELID_0.2_1.pth'
NN_model = NN(input_dim, hidden_dim, hidden_num, z0_dim)
KKT_model = NNOPT(input_dim, hidden_dim, hidden_num, z0_dim, A, B, b)
NN_model.load_state_dict(torch.load(NN_path,weights_only=False)['state_dict'])
KKT_model.load_state_dict(torch.load(KKT_path,weights_only=False)['state_dict'])
NN_model.eval()
KKT_model.eval()
print(NN_model)
print(KKT_model)


