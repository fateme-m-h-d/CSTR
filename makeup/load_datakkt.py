# import torch
# import numpy as np
# from models import NN, NNOPT, ECNN
# import torch.nn as nn
# import matplotlib.pyplot as plt


# device = "cuda" if torch.cuda.is_available() else "cpu"


# def load_saved_model(model_path, model_type, input_dim, hidden_dim, hidden_num, z0_dim, A=None, B=None, b=None):
#     """
#     Load a saved model from path
#     """
#     # Initialize the correct model type
#     if model_type == "NN":
#         model = NN(input_dim, hidden_dim, hidden_num, z0_dim)
#     elif model_type == "NNOPT":
#         model = NNOPT(input_dim, hidden_dim, hidden_num, z0_dim, A, B, b)
#     # elif model_type == "ECNN":
#     #     model = ECNN(input_dim, hidden_dim, hidden_num, z0_dim, A, B_indep, B_dep, b)
    
#     # Load the saved weights
#     checkpoint = torch.load(model_path, map_location=device)
#     model.load_state_dict(checkpoint['state_dict'])
#     model.to(device)
#     model.eval()
#     return model

# def make_prediction(model, temperature):
#     """
#     Make predictions for new temperature values.
#     """
    
#     if isinstance(temperature, (int, float)):
#         temperature = torch.tensor([[temperature]], dtype=torch.float32)
#     elif isinstance(temperature, list) or isinstance(temperature, np.ndarray):
#         temperature = torch.tensor(temperature, dtype=torch.float32).unsqueeze(-1)
    
#     temperature = temperature.to(device)
    

#     with torch.no_grad():
#         output = model(temperature)
    
#     return output.cpu().numpy()


# # Example usage:
# if __name__ == "__main__":
    
#     input_dim = 1  # Temperature
#     hidden_dim = 32
#     hidden_num = 2
#     z0_dim = 3  # Number of output variables (Ca, Cb, Cc)
    
    
#     model_path = "./models/cstr/NN/0.2/MODELID_0.2_0.pth"
    
#     # Load the model
#     model = load_saved_model(model_path, "NN", input_dim, hidden_dim, hidden_num, z0_dim)
    
#     # Make predictions for new temperatures
#     new_temperatures = [300, 400, 500]
#     predictions = make_prediction(model, new_temperatures)
    
#     # Print predictions
#     for temp, pred in zip(new_temperatures, predictions):
#         print(f"Temperature: {temp}K")
#         print(f"Predicted concentrations (Ca, Cb, Cc): {pred}")
   
#     from matplotlib import pyplot as plt
        
#     plt.figure()
#     plt.plot(new_temperatures, predictions [0])
#     plt.show
    


# import torch
# import numpy as np
# from models import NN, NNOPT, ECNN
# import torch.nn as nn
# import pickle
# import os

# device = "cuda" if torch.cuda.is_available() else "cpu"

# def load_saved_model(model_path, model_type, input_dim, hidden_dim, hidden_num, z0_dim, A=None, B=None, b=None):
#     # Load the saved model similar to your existing code
#     if model_type == "NN":
#         model = NN(input_dim, hidden_dim, hidden_num, z0_dim)
#     elif model_type == "NNOPT":
#         model = NNOPT(input_dim, hidden_dim, hidden_num, z0_dim, A, B, b)
    
#     checkpoint = torch.load(model_path, map_location=device)
#     model.load_state_dict(checkpoint['state_dict'])
#     model.to(device)
#     model.eval()
#     return model

# def make_prediction(model, scaler, temperature):
#     # Normalize the new input data using the loaded scaler
#     if isinstance(temperature, (int, float)):
#         temperature = np.array([[temperature]])
#     elif isinstance(temperature, list) or isinstance(temperature, np.ndarray):
#         temperature = np.array(temperature).reshape(-1, 1)

#     # Normalize input using scaler
#     temperature_normalized = scaler.transform(temperature)
#     temperature_tensor = torch.tensor(temperature_normalized, dtype=torch.float32).to(device)

#     # Make prediction
#     with torch.no_grad():
#         output = model(temperature_tensor)

#     # Inverse transform to get original scale
#     predictions = output.cpu().numpy()
#     predictions_original = scaler.inverse_transform(predictions)
    
#     return predictions_original





# if __name__ == "__main__":
#     # Load the scaler
#     scaler_path = os.path.join(os.getcwd(), 'scaler.pkl')  # Construct the full path to the scaler.pkl file
#     try:
#         with open(scaler_path, 'rb') as f:
#             scaler = pickle.load(f)
#         print(f"Scaler loaded successfully from {scaler_path}")
#     except FileNotFoundError:
#         print(f"Scaler file not found at {scaler_path}. Make sure to save the scaler during training.")
#         exit(1)  # Exit the script as there's no way to proceed without the scaler



#     # Load the model
#     input_dim = 1
#     hidden_dim = 32
#     hidden_num = 2
#     z0_dim = 3
#     model_path = "./models/cstr/NN/0.2/MODELID_0.2_0.pth"
#     model = load_saved_model(model_path, "NN", input_dim, hidden_dim, hidden_num, z0_dim)

#     # Make predictions for new temperatures
#     new_temperatures = [300, 400, 500]
#     predictions = make_prediction(model, scaler, new_temperatures)

#     # Print predictions
#     for temp, pred in zip(new_temperatures, predictions):
#         print(f"Temperature: {temp}K, Predicted concentrations (Ca, Cb, Cc): {pred}")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sym
from sympy import symbols, Eq, solve
from scipy.optimize import fsolve

Cao = 1 #mol/L
Cbo = 2 #mol/L
V = 10 #L
Q = 1 #L/s
tau = V/Q #s

#Parameters to tuning to obtain "aggressive" non-linearity
Afo = 10e12
Eaf = 100000 #J/mol
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

# Define the range of T values
T_values = np.linspace(280, 800, 1000)  # Adjust the range and number of points as needed


# Initial guess for fsolve
initial_guess = [0, 2, 1]

# Lists to store results
Ca_values = []
Cb_values = []
Cc_values = []

# Loop over each value of T and solve for Ca and Cb
for T in T_values:
    solution, infodict, ier, mesg = fsolve(equations, initial_guess, args=(T,), full_output=True)
    #solution, mesg = fsolve(equations, initial_guess, args=(T,))
    if ier == 1:  # ier == 1 indicates successful convergence
       Cc, Cb, Ca = solution
       Ca_values.append(Ca)
       Cb_values.append(Cb)
       Cc_values.append(Cc)
    else:
        print(f"Solver did not converge for T = {T}. Message: {mesg}")

kf = Afo * np.exp(-Eaf/(R*T_values)) #Arrhenius eqn for forward reaction
kr = Aro * np.exp(-Ear/(R*T_values)) #arrhenius eqn for reverse reaction
ra=-kf*Ca*(Cb**2)*tau + kr*(Cao-Ca+Cbo-Cb)

# Print results or process them further
for T, Ca, Cb, Cc in zip(T_values, Ca_values, Cb_values, Cc_values):
    print(f"T: {T:.2f}, Ca: {Ca:.4f}, Cb: {Cb:.4f}, Cc: {Cc:.4f}")


# Create a DataFrame
data = pd.DataFrame({
    'Temperature (T)': T_values,
    'Ca': Ca_values,
    'Cb': Cb_values,
    'Cc': Cc_values
})

# Save to Excel
data.to_csv("C:/Users/Fateme/Desktop/Research/CSTR/makeup/data.csv", index=False)
data.to_excel("C:/Users/Fateme/Desktop/Research/CSTR/makeup/data.xlsx", index=False)
print("Data saved to 'data.csv'")


# plt.figure()
# plt.scatter(T_values, Ca_values)
# plt.scatter(T_values, Cb_values)
# plt.scatter(T_values, Cc_values)
# plt.show()





import torch
import numpy as np
import pickle
import os
from models import NN, NNOPT, ECNN
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_saved_model(model_path, model_type, input_dim, hidden_dim, hidden_num, z0_dim, A=None, B=None, b=None):
    # Load the saved model similar to your existing code
    if model_type == "NN":
        model = NN(input_dim, hidden_dim, hidden_num, z0_dim)
    elif model_type == "NNOPT":
        model = NNOPT(input_dim, hidden_dim, hidden_num, z0_dim, A, B, b)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
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
    temperature_tensor = torch.tensor(temperature_normalized, dtype=torch.float32).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(temperature_tensor)

    # Convert predictions to numpy and inverse-transform the complete dataset
    predictions = output.cpu().numpy()
    dummy_full_data[:, 1:] = predictions  # Insert the predictions into the dummy array to inverse transform

    # Inverse transform to get the original scale
    predictions_original = scaler.inverse_transform(dummy_full_data)[:, 1:]  # Extract only the output features

    return predictions_original

# if __name__ == "__main__":
if __name__ == "__main__":
    # Load the scaler used during training
    scaler_path = os.path.join(os.getcwd(), 'scaler.pkl')  # Construct the full path to the scaler.pkl file
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Scaler loaded successfully from {scaler_path}")
    except FileNotFoundError:
        print(f"Scaler file not found at {scaler_path}. Make sure to save the scaler during training.")
        exit(1)  # Exit the script as there's no way to proceed without the scaler

    # Load the model
    input_dim = 1
    hidden_dim = 32
    hidden_num = 2
    z0_dim = 3
    model_path = "./models/cstr/NN/0.2/MODELID_0.2_0.pth"
    try:
        model = load_saved_model(model_path, "NN", input_dim, hidden_dim, hidden_num, z0_dim)
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Model file not found at {model_path}. Make sure the model is trained and the file path is correct.")
        exit(1)

    # Make predictions for new temperatures
    new_temperatures = np.linspace(280, 800, 1000)
    predictions = make_prediction(model, scaler, new_temperatures)

    # Print predictions
    for temp, pred in zip(new_temperatures, predictions):
        print(f"Temperature: {temp}K, Predicted concentrations (Ca, Cb, Cc): {pred}")
        
    
    # Extract 'Ca' from predictions (assuming Ca is the first output column)
    Ca_values1 = predictions[:, 0]
    Cb_values1 = predictions[:, 1]
    Cc_values1 = predictions[:, 2]

    # Plot Ca based on temperature (T)
    plt.figure()
    plt.plot(new_temperatures, Ca_values1, label='Ca', color='b')
    plt.plot(new_temperatures, Cb_values1, label='Cb', color='r')
    plt.plot(new_temperatures, Cc_values1, label='Cc', color='g')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Concentration')
    plt.title('Concentration vs Temperature')
    plt.legend()
    plt.grid(True)
    
    
    plt.scatter(T_values, Ca_values)
    plt.scatter(T_values, Cb_values)
    plt.scatter(T_values, Cc_values)
    plt.show()