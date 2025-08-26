# First, load and examine the dataset to understand its structure
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
dataset_path = './data.csv'
data = pd.read_csv(dataset_path)

# Display the first few rows to confirm loading and structure
data.head()

# Add Gaussian noise to the last three columns and save to a new CSV file
def gaussian(data, noise_std, output_path='noisy_data.csv', multiplier=3):
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
    
    return noisy_data

# Generate and save noisy dataset with noise added only to the last three columns
# gaussian(data, noise_std=0.05, output_path='noisy_data.csv', multiplier=3)

# Generate and save noisy datasets with different noise levels
small_noise_data = gaussian(data, noise_std=0.01, output_path='small_noisy_data.csv', multiplier=3)
moderate_noise_data = gaussian(data, noise_std=0.05, output_path='moderate_noisy_data.csv', multiplier=3)
medium_noise_data = gaussian(data, noise_std=0.1, output_path='large_noisy_data.csv', multiplier=3)

# Plot the original and noisy datasets
def plot_data(data, noisy_data, title):
    plt.figure(figsize=(10, 6))
    plt.plot(data['Temperature (T)'], data['Ca'], label='Original Ca', alpha=0.7)
    plt.plot(noisy_data['Temperature (T)'], noisy_data['Ca'], label='Noisy Ca', alpha=0.7)
    plt.xlabel('Temperature (T)')
    plt.ylabel('Concentration (Ca)')
    plt.title(title)
    plt.legend()
    plt.show()

plot_data(data, small_noise_data, 'Small Noise')
plot_data(data, moderate_noise_data, 'Moderate Noise')
plot_data(data, medium_noise_data, 'Large Noise')
