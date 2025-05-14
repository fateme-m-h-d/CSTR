import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
dataset_path = './data.csv'
data = pd.read_csv(dataset_path)

# Display the first few rows to confirm loading and structure
data.head()

# Add Gaussian noise to the first column (Temperature) and save to a new CSV file
def gaussian_temperature(data, noise_std=0.5, output_path='noisy_data.csv', multiplier=3):
    noisy_data = data.copy()
    
    # Identify the first numeric column (assuming it's Temperature)
    first_numeric_column = data.select_dtypes(include=[np.number]).columns[0]
    print("Column to modify:", first_numeric_column)
    
    # Preserve original order by modifying values in-place
    noise = np.random.normal(0, multiplier * noise_std * data[first_numeric_column].std(), size=len(data))
    noisy_data[first_numeric_column] = data[first_numeric_column] + noise
    
    # Ensure the dataset maintains its original order
    noisy_data = noisy_data.sort_index()
    
    # Save without modifying row order
    noisy_data.to_csv(output_path, index=False)
    print(f"Noisy dataset saved to {output_path}")
    
    return noisy_data

# Generate and save noisy datasets with different noise levels
small_noise_data = gaussian_temperature(data, noise_std=0.01, output_path='small_noisy_data.csv', multiplier=3)
moderate_noise_data = gaussian_temperature(data, noise_std=0.05, output_path='moderate_noisy_data.csv', multiplier=3)
large_noise_data = gaussian_temperature(data, noise_std=0.1, output_path='large_noisy_data.csv', multiplier=3)


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
plot_data(data, large_noise_data, 'Large Noise')
