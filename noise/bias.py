import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
dataset_path = './data.csv'
data = pd.read_csv(dataset_path)

# Display the first few rows to confirm loading and structure
data.head()

# Add bias and disturbance to the first three columns and save to a new CSV file
def add_bias_and_disturbance(data, bias_value=0.5, disturbance_value=0.5, output_path='biased_data.csv', multiplier=3):
    biased_data = data.copy()
    bias_statistics = {}
    
    # Identify the first three numeric columns
    last_three_columns = data.select_dtypes(include=[np.number]).columns[-3:]
    print("Last three columns to modify:", last_three_columns)
    
    # Add bias and disturbance only to the last three numeric columns
    for col in last_three_columns:
        bias = multiplier * bias_value * data[col].mean()
        disturbance = disturbance_value * data[col].mean()
        biased_data[col] += bias
        # Add disturbance in the middle of the range
        middle_index = len(biased_data) // 2
        disturbance_range = 100  # Adjust the range as needed
        biased_data.loc[middle_index-disturbance_range//2:middle_index+disturbance_range//2, col] += disturbance
        bias_statistics[col] = {'mean_shift': bias, 'disturbance': disturbance}
    
    # Save the biased dataset to a new CSV file
    biased_data.to_csv(output_path, index=False)
    print(f"Biased dataset saved to {output_path}")
    
    # Display bias statistics
    for col, stats in bias_statistics.items():
        print(f"Bias added to column '{col}': Mean shift = {stats['mean_shift']:.5f}, Disturbance = {stats['disturbance']:.5f}")
    
    return biased_data

# Generate and save biased datasets with different bias levels and disturbances
small_bias_data = add_bias_and_disturbance(data, bias_value=0.01, disturbance_value=0.02, output_path='small_biased_data.csv', multiplier=3)
moderate_bias_data = add_bias_and_disturbance(data, bias_value=0.05, disturbance_value=0.1, output_path='moderate_biased_data.csv', multiplier=3)
large_bias_data = add_bias_and_disturbance(data, bias_value=0.1, disturbance_value=0.2, output_path='large_biased_data.csv', multiplier=3)

# Plot the original and biased datasets
def plot_data(data, biased_data, title):
    plt.figure(figsize=(10, 6))
    plt.plot(data['Temperature (T)'], data['Ca'], 'b-', label='Original Ca', alpha=0.7)
    plt.plot(data['Temperature (T)'], data['Cb'], 'r-', label='Original Cb', alpha=0.7)
    plt.plot(data['Temperature (T)'], data['Cc'], 'g-', label='Original Cc', alpha=0.7)
    plt.plot(biased_data['Temperature (T)'], biased_data['Ca'], 'b--', label='Biased Ca', alpha=0.7)
    plt.plot(biased_data['Temperature (T)'], biased_data['Cb'], 'r--', label='Biased Cb', alpha=0.7)
    plt.plot(biased_data['Temperature (T)'], biased_data['Cc'], 'g--', label='Biased Cc', alpha=0.7)
    plt.xlabel('Temperature (T)')
    plt.ylabel('Concentration (mol/L)')
    plt.title(title)
    plt.legend()
    plt.show()

plot_data(data, small_bias_data, 'Small Bias with Disturbance')
plot_data(data, moderate_bias_data, 'Moderate Bias with Disturbance')
plot_data(data, large_bias_data, 'Large Bias with Disturbance')