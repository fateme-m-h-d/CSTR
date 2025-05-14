# First, load and examine the dataset to understand its structure
import pandas as pd
import numpy as np

# Load dataset
dataset_path = './data.csv'
data = pd.read_csv(dataset_path)

# Display the first few rows to confirm loading and structure
data.head()

# Define the equations to check
def check_equations(data):
    # Extract required columns
    T = data['Temperature (T)']
    Ca = data['Ca']
    Cb = data['Cb']
    Cc = data['Cc']
    
    # Equation 1: x2 - x3 - z2 + z3 = 0
    eq1_diff = -102870.612474334*Ca -102869.61247398*Cb + 0*Cc -315.818584958539*T +362807.369750825
    
    # Equation 2: x2 - z1 - z2 = 0
    eq2_diff = -205739.224948669*Ca -205740.224947961*Cb + 0*Cc -631.637169917077*T +725614.739501651
    
    # Equation 1: x2 - x3 - z2 + z3 = 0
    eq1 = (-102870.612474334*Ca -102869.61247398*Cb + 0*Cc -315.818584958539*T +362807.369750825).abs() < 1e-6
    
    # Equation 2: x2 - z1 - z2 = 0
    eq2 = (-205739.224948669*Ca -205740.224947961*Cb + 0*Cc -631.637169917077*T +725614.739501651).abs() < 1e-6
    
    # Combine results
    # data['eq1_satisfied'] = eq1
    # data['eq2_satisfied'] = eq2
    # data['both_satisfied'] = eq1 & eq2
    
    # Summary of differences
    print("Summary of Differences for Equation 1:")
    print(eq1_diff.describe())
    print("\nSummary of Differences for Equation 2:")
    print(eq2_diff.describe())
    
    # Display the first few differences
    print("\nFirst few differences for Equation 1:")
    print(eq1_diff.head())
    print("\nFirst few differences for Equation 2:")
    print(eq2_diff.head())
    
    # Summary of results
    eq1_percentage = eq1.mean() * 100
    eq2_percentage = eq2.mean() * 100
    both_percentage = (eq1 & eq2).mean() * 100
    
    print(f"Equation 1 satisfied for {eq1_percentage:.2f}% of rows")
    print(f"Equation 2 satisfied for {eq2_percentage:.2f}% of rows")
    print(f"Both equations satisfied for {both_percentage:.2f}% of rows")
    
    return data

# Run the check
data = check_equations(data)

# Display a few rows with satisfaction flags
data.head()


# Add Gaussian noise to the first three columns and save to a new CSV file
def gaussian(data, noise_std=0.05, output_path='noisy_data.csv', multiplier=3):
    noisy_data = data.copy()
    noise_statistics = {}
    
    # Identify the first three numeric columns
    last_three_columns = data.select_dtypes(include=[np.number]).columns[-3:]
    print("Last three columns to modify:", last_three_columns)
    
    # Add Gaussian noise only to the last three numeric columns
    for col in last_three_columns:
        noise = np.random.normal(0, multiplier*noise_std * data[col].std(), size=len(data))
        noisy_data[col] += noise
        noise_statistics[col] = {'mean': noise.mean(), 'std': noise.std()}
    
    # Save the noisy dataset to a new CSV file
    noisy_data.to_csv(output_path, index=False)
    print(f"Noisy dataset saved to {output_path}")
    
    # Display noise statistics
    for col, stats in noise_statistics.items():
        print(f"Noise added to column '{col}': Mean = {stats['mean']:.5f}, Std = {stats['std']:.5f}")

# Run the check and compute differences
check_equations(data)

# Generate and save noisy dataset with noise added only to the last three columns
gaussian(data, noise_std=0.05, output_path='noisy_data.csv', multiplier=3)
