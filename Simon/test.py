import torch
import pandas as pd

print(torch.tensor([[1,2],[2,3]]))
#Linearized parameters
coeff = pd.read_csv("/Users/simonnguyen/Library/CloudStorage/OneDrive-UniversityofWaterloo/Undergrad Terms UW/4A/CHE 499/CSTR_example/coefficients.csv").to_numpy()[0]
a,c,d,e,f,g,h,i = -coeff[0],-coeff[1],-coeff[2],coeff[3],-coeff[4],-coeff[5],-coeff[6],coeff[7]
print(a,c,d,e,f,g,h,i)
print(torch.tensor([[d], [h]],dtype=torch.float32))