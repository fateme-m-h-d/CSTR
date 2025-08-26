import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sym
from sympy import symbols, Eq, solve
from scipy.optimize import fsolve
from utils import load_data, get_scaledABb


Cao = 1 #mol/L
Cbo = 2 #mol/L
Cco = 0 #mol/L
V = 10 #L
Q = 1 #L/s
tau = V/Q #s

#Parameters to tuning to obtain "aggressive" non-linearity
Afo = 10e12
Eaf = 90000 #J/mol
Aro = 10e10
Ear = 80000 #J/mol
R = 8.314 #J/mol


def equations(variables, T):
    Cc, Cb, Ca = variables
    kf = Afo * np.exp(-Eaf/(R*T)) #Arrhenius eqn for forward reaction
    kr = Aro * np.exp(-Ear/(R*T)) #arrhenius eqn for reverse reaction

    eq1 = Cao - Ca + -kf*Ca*(Cb**2)*tau + kr*(Cao-Ca+Cbo-Cb)*tau
    eq2 = Cbo - Cb + -2*kf*Ca*(Cb**2)*tau + 2*kr*(Cao-Ca+Cbo-Cb)*tau
    eq3=Cc-Cao+Ca-Cbo+Cb
    return [eq1, eq2, eq3]




#Load model for prediction
import torch
import numpy as np
import pickle
import os
from models import NN, NNOPT

device = "cuda" if torch.cuda.is_available() else "cpu"
# if the model is NN (line 294 of the code), set the A, B, b equal to None in the code below.
def load_saved_model(model_path, model_type, input_dim, hidden_dim, hidden_num, z0_dim, A, B, b):
    # Load the saved model
    if model_type == "NN":
        model = NN(input_dim, hidden_dim, hidden_num, z0_dim)
    elif model_type == "KKT":
        model = NNOPT(input_dim, hidden_dim, hidden_num, z0_dim, A, B, b)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.double().to(device)
    model.eval()
    return model

def make_prediction(model, scaler, temperature):
    # Create a dummy array with the same structure as the original dataset (input + outputs)
    # Assuming the original dataset had 1 input and 3 outputs (total 4 features)
    if isinstance(temperature, (int, float)):
        temperature = np.array([[temperature]])
    elif isinstance(temperature, list) or isinstance(temperature, np.ndarray):
        temperature = np.array(temperature).reshape(-1, 1)

    # Create a dummy array with 4 features (input + outputs)
    num_samples = temperature.shape[0]
    dummy_full_data = np.zeros((num_samples, 4))
    dummy_full_data[:, 0] = temperature[:, 0]  # Set the temperature value as the first feature

    # Transform the entire dummy dataset with the scaler
    transformed_data = scaler.transform(dummy_full_data)

    # Extract the transformed input (temperature)
    temperature_normalized = transformed_data[:, :1]  # Only use the first feature for input

    # Convert to tensor and make prediction
    temperature_tensor = torch.tensor(temperature_normalized, dtype=torch.float64).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(temperature_tensor)

    # Convert predictions to numpy and inverse-transform the complete dataset
    predictions = output.cpu().numpy()
    dummy_full_data[:, 1:] = predictions  # Insert the predictions into the dummy array to inverse transform

    # Inverse transform to get the original scale
    predictions_original = scaler.inverse_transform(dummy_full_data)[:, 1:]  # Extract only the output features

    return predictions_original

if __name__ == "__main__":
    _, scaler = load_data("./data.csv")

    input_dim = 1
    hidden_dim = 32
    hidden_num = 2
    z0_dim = 3
    
    # A = torch.tensor([[0], [-0.0338064827931121]
    #                             ])  #changed
    # B = torch.tensor([[1, 1, 1], [-1.79760064435781, - 0.797600644357812, 0]
    #                             ])  #changed
    # b = torch.tensor([3, -13.3172033549155])    #changed
    
    
    
    # A = torch.tensor([[-19.455307300247398]
    #                             ])  #changed
    # B = torch.tensor([[ -11743.801825483246, -11742.801825486215, 0]
    #                             ])  #changed
    # b = torch.tensor([-22866.47311208096])
    
    A = torch.tensor([[- 198.802430563989]
                                ])  #changed
    B = torch.tensor([[ -168482.732203137, - 168481.732203863, 0]
                                ])  #changed
    b = torch.tensor([-286884.958823058])
    
    A = torch.tensor([[-198.802430563989]], dtype=torch.float64)
    B = torch.tensor([[-168482.732203137, -168481.732203863, 0]], dtype=torch.float64)
    b = torch.tensor([-286884.958823058], dtype=torch.float64)

    NNmodel_path = "./models/cstr/NN/0.2/MODELID_0.2_0.pth"
    KKTmodel_path = "./models/cstr/KKThPINN/0.2/MODELID_0.2_0.pth"
    model_type = "KKT" #change this to produce NN or KKT results
    if model_type == "NN":
        model = load_saved_model(NNmodel_path, "NN", input_dim, hidden_dim, hidden_num, z0_dim)
        print(f"Model loaded successfully from {NNmodel_path}")
    elif model_type =="KKT":
        A, B, b = get_scaledABb(A, B, b, scaler)
        A = A.double()  # Ensure A is float32
        B = B.double()  # Ensure B is float32
        b = b.double()  # Ensure b is float32

        model = load_saved_model(KKTmodel_path,"KKT",input_dim,hidden_dim,hidden_num,z0_dim,A,B,b)
    elif FileNotFoundError:
        print(f"Model file not found at {NNmodel_path}. Make sure the model is trained and the file path is correct.")
        exit(1)
    
    import inspect, models
    print("models.py file =", models.__file__)
    print("forward defined at", inspect.getsourcefile(type(model).forward))


    # ───────────── PREPARE INPUTS (scaled & unscaled) ──────────
    TEMPS = np.linspace(280, 600.0, 200)
    T_unscaled = TEMPS.reshape(-1, 1)   # (N,1) raw K
    # Build dummy rows to transform T consistently with training scaler
    dummy = np.zeros((len(T_unscaled), 4), dtype=np.float64)
    dummy[:, 0] = T_unscaled[:, 0]
    X_scaled_np = scaler.transform(dummy)[:, :1]           # scaled T only
    X_scaled    = torch.tensor(X_scaled_np, dtype=torch.float64, device=device)  # (N,1)
        
        # ───────────── PREDICT (scaled outputs) ────────────────────
    with torch.no_grad():
        Z_scaled = model(X_scaled)  # (N,3), still in scaled space
        chunk = B.t() @ torch.inverse(B @ B.t())
        Astar_now = - chunk @ A
        Bstar_now = torch.eye(3, dtype=B.dtype, device=B.device) - chunk @ B
        bstar_now = (chunk @ b.view(-1,1)).view(-1)
    
    print("ΔBstar =", (model.fc_fixed1.weight - Bstar_now).abs().max().item())
    print("ΔAstar =", (model.fc_fixed2.weight - Astar_now).abs().max().item())
    print("Δbstar =", (model.fc_fixed2.bias   - bstar_now).abs().max().item())
        
    #     r_ext = (A @ X_scaled.T) + (B @ Z_scaled.T) - b.view(1,1)
    #      # the model's actual buffers (names assumed; adjust if different)
    #     A_m = model.A.detach()
    #     B_m = model.B.detach()
    #     b_m = model.b.detach()
    #     r_int = (A_m @ X_scaled.T) + (B_m @ Z_scaled.T) - b_m.view(1,1)
    # print("max |A_ext - A_model| =", (A - A_m).abs().max().item())
    # print("max |B_ext - B_model| =", (B - B_m).abs().max().item())
    # print("max |b_ext - b_model| =", (b - b_m).abs().max().item())
    # print("mean|r_ext| =", r_ext.abs().mean().item(), "   mean|r_int| =", r_int.abs().mean().item())
        

    Z_scaled_np = Z_scaled.detach().cpu().numpy()

    # 3) Residual in *scaled* space using the *scaled* (A,B,b) for the LAST segment
    A_s = A.cpu()   # (1,1)
    B_s = B.cpu()   # (1,3)
    b_s = b.cpu().reshape(1,)  # (1,)
    
    with torch.no_grad():
        r_scaled_t = (A_s @ X_scaled.T) + (B_s @ Z_scaled.T) - b_s.view(-1, 1)  # (1,N)
        r_scaled = r_scaled_t.squeeze(0).abs().detach().cpu().numpy()  # (N,)

    # r_scaled = (X_scaled @ A_s.T + Z_scaled @ B_s.T - b_s).ravel()
    print(f"[scaled] last seg 503–600: max|r|={np.abs(r_scaled).max():.3e}, mean|r|={np.abs(r_scaled).mean():.3e}")
    violation_like_training = np.mean(np.abs(r_scaled))
    print(violation_like_training)



    # 5) Quick plot of both
    plt.figure()
    plt.scatter(TEMPS, np.abs(r_scaled), label="scaled residual")
    # plt.semilogy(T_eval, np.abs(r_raw), ".", label="raw residual")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Violation |A·x + B·z − b|")
    plt.yscale('log')
    plt.title("KKT residuals")
    plt.legend()
    plt.grid(True, which="both")
    plt.show()


    # Make predictions for new temperatures
    new_temperatures = np.linspace(280, 600, 200) #think about it
    #new_temperatures=np.array([450, 451])
    predictions = make_prediction(model, scaler, new_temperatures)
    print("Scaler Type:", type(scaler))
    if hasattr(scaler, 'scale_'):
        print("Scaler Factors (scale_):", scaler.scale_)

    # Print predictions
    # for temp, pred in zip(new_temperatures, predictions):
    #     print(f"Temperature: {temp} K, Predicted concentrations (Ca, Cb, Cc): {pred} (mol/L)")
        
    
    # Extract concentrations from predictions (assuming Ca is the first output column)
    Ca_values1 = predictions[:, 0]
    Cb_values1 = predictions[:, 1]
    Cc_values1 = predictions[:, 2]

    # Plot Concentration versus Temperature
    plt.figure()
    plt.plot(new_temperatures, Ca_values1, color='b',label="Predicted Ca")
    plt.plot(new_temperatures, Cb_values1, color='r', label="Predicted Cb")
    plt.plot(new_temperatures, Cc_values1, color='g', label="Predicted Cc")
    
        # Define the range of T values
    n = 200 #number of points
    T_values = np.linspace(360, 600, n)  # Adjust the range and number of points as needed


    # Initial guess for fsolve
    initial_guess = [Cco, Cbo, Cao]

    # Lists to store results
    Ca_values = np.ones(n)*Cao
    Cb_values = np.ones(n)*Cbo
    Cc_values = np.ones(n)*Cco
    i=0
    # Loop over each value of T and solve for Ca and Cb
    for T in T_values:
        solution, infodict, ier, mesg = fsolve(equations, initial_guess, args=(T,), full_output=True)
        #solution, mesg = fsolve(equations, initial_guess, args=(T,))
        if ier == 1:  # ier == 1 indicates successful convergence
            Cc_values[i], Cb_values[i], Ca_values[i] = solution[0], solution[1], solution[2]
        else:
            print(f"Solver did not converge for T = {T}. Message: {mesg}")
        i+=1
    
    plt.plot(T_values, Ca_values,'b--',label='Nonlinearized Ca')
    plt.plot(T_values, Cb_values,'r--',label='Nonlinearized Cb')
    plt.plot(T_values, Cc_values,'g--',label='Nonlinearized Cc')
    
    plt.xlabel('Temperature (K)')
    plt.ylabel('Concentration (mol/L)')
    plt.title('Predictions Over Original Data')
    plt.legend()
    plt.grid()
    plt.show()
    
import pandas as pd

# # Load the data_limited.csv file
# data_limited = pd.read_csv("./data_limited.csv")

# # Filter the data to include only temperatures from 280 to 400 and from 500 to 600
# filtered_data = data_limited[(data_limited['Temperature (T)'] >= 280) & (data_limited['Temperature (T)'] <= 300) |
#                              (data_limited['Temperature (T)'] >= 500) & (data_limited['Temperature (T)'] <= 600)]

# # Save the filtered data to a new CSV file
# filtered_data.to_csv("./filtered_data.csv", index=False)
# print("Filtered data saved to 'filtered_data.csv'")


# # Load the small noisy data CSV file
# #noisy_data = pd.read_csv('./large_noisy_data.csv')
# bias_data = pd.read_csv('./moderate_biased_data.csv')

# # Plot the prediction graphs over noisy data
# plt.figure()
# plt.plot(bias_data['Temperature (T)'], bias_data['Ca'], 'b--', label='Noisy Ca')
# plt.plot(bias_data['Temperature (T)'], bias_data['Cb'], 'r--', label='Noisy Cb')
# plt.plot(bias_data['Temperature (T)'], bias_data['Cc'], 'g--', label='Noisy Cc')

# plt.plot(new_temperatures, Ca_values1, color='k',label="Predicted Ca")
# plt.plot(new_temperatures, Cb_values1, color='k', label="Predicted Cb")
# plt.plot(new_temperatures, Cc_values1, color='k', label="Predicted Cc")
# plt.xlabel('Temperature (K)')
# plt.ylabel('Concentration (mol/L)')
# plt.legend()
# plt.title('Predictions Over Biased Data')
# plt.show()

