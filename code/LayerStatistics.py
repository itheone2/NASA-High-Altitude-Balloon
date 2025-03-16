#Layer Statistics:
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# File name
file_name = 'cleaned_gps_data.csv'

# Check if the file exists
if not os.path.exists(file_name):
    print(f"The file '{file_name}' does not exist in the current directory.")
    print("Please make sure the file is in the same directory as this script.")
    exit()

# Load the data
try:
    df = pd.read_csv(file_name)
except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    exit()

# Function to calculate potential temperature
def potential_temperature(T, P):
    return (T + 273.15) * (1000 / P) ** 0.286

# Calculate potential temperature
df['potential_temperature'] = potential_temperature(df['temperature_c_celsius'], df['pressure_hpa'])

# Calculate lapse rate
df['lapse_rate'] = df['temperature_c_celsius'].diff() / df['gps_alt_m'].diff() * 1000  # °C/km

# Perform KMeans clustering
features = ['temperature_c_celsius', 'pressure_hpa', 'humidity_%.1', 'potential_temperature']
X = df[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42)  # Assuming 5 layers, adjust as needed
df.loc[X.index, 'layer'] = kmeans.fit_predict(X_scaled)

# Analyze layers
layers = df.groupby('layer')
layer_stats = layers.agg({
    'gps_alt_m': ['min', 'max'],
    'temperature_c_celsius': ['mean', 'std'],
    'pressure_hpa': ['mean', 'std'],
    'humidity_%.1': ['mean', 'std'],
    'potential_temperature': ['mean', 'std'],
    'lapse_rate': ['mean', 'std']
})

# Plot results
fig, axs = plt.subplots(2, 3, figsize=(20, 15))

# Temperature profile
axs[0, 0].plot(df['temperature_c_celsius'], df['gps_alt_m'])
axs[0, 0].set_xlabel('Temperature (°C)')
axs[0, 0].set_ylabel('Altitude (m)')
axs[0, 0].set_title('Temperature Profile')

# Lapse rate
axs[0, 1].plot(df['lapse_rate'], df['gps_alt_m'])
axs[0, 1].axvline(x=-9.8, color='r', linestyle='--', label='Dry Adiabatic Lapse Rate')
axs[0, 1].set_xlabel('Lapse Rate (°C/km)')
axs[0, 1].set_ylabel('Altitude (m)')
axs[0, 1].set_title('Lapse Rate Profile')
axs[0, 1].legend()

# Potential temperature
axs[0, 2].plot(df['potential_temperature'], df['gps_alt_m'])
axs[0, 2].set_xlabel('Potential Temperature (K)')
axs[0, 2].set_ylabel('Altitude (m)')
axs[0, 2].set_title('Potential Temperature Profile')

# Humidity and Dew Point
axs[1, 0].plot(df['humidity_%.1'], df['gps_alt_m'], label='Relative Humidity')
axs[1, 0].plot(df['dew_point'], df['gps_alt_m'], label='Dew Point')
axs[1, 0].set_xlabel('Relative Humidity (%) / Dew Point (°C)')
axs[1, 0].set_ylabel('Altitude (m)')
axs[1, 0].set_title('Humidity and Dew Point Profile')
axs[1, 0].legend()

# Layer visualization
scatter = axs[1, 1].scatter(df['temperature_c_celsius'], df['gps_alt_m'], c=df['layer'], cmap='viridis')
axs[1, 1].set_xlabel('Temperature (°C)')
axs[1, 1].set_ylabel('Altitude (m)')
axs[1, 1].set_title('Atmospheric Layers')
plt.colorbar(scatter, ax=axs[1, 1], label='Layer')

# Temperature inversion detection
temp_diff = df['temperature_c_celsius'].diff()
inversion_mask = temp_diff > 0
axs[1, 2].plot(df['temperature_c_celsius'], df['gps_alt_m'], label='Temperature')
axs[1, 2].scatter(df.loc[inversion_mask, 'temperature_c_celsius'], df.loc[inversion_mask, 'gps_alt_m'],
                  color='red', label='Inversion')
axs[1, 2].set_xlabel('Temperature (°C)')
axs[1, 2].set_ylabel('Altitude (m)')
axs[1, 2].set_title('Temperature Inversions')
axs[1, 2].legend()

plt.tight_layout()
plt.savefig('atmospheric_analysis.png')
plt.close()

# Print layer statistics
print("Layer Statistics:")
print(layer_stats)

# Identify tropopause (simplistic approach - where lapse rate becomes positive)
tropopause_idx = (df['lapse_rate'] > 0).idxmax()
print(f"\nEstimated tropopause altitude: {df.loc[tropopause_idx, 'gps_alt_m']:.2f} m")
print(f"Tropopause temperature: {df.loc[tropopause_idx, 'temperature_c_celsius']:.2f} °C")

# Stability analysis
df['stability'] = np.where(df['lapse_rate'] < -9.8, 'Unstable',
                           np.where(df['lapse_rate'] > 0, 'Very Stable', 'Stable'))

stability_stats = df.groupby('stability').agg({
    'gps_alt_m': ['min', 'max'],
    'lapse_rate': ['mean', 'std']
})
print("\nStability analysis:")
print(stability_stats)

# Save processed data
df.to_csv('processed_balloon_data.csv', index=False)

print("\nAnalysis complete. Check 'atmospheric_analysis.png' for plots and 'processed_balloon_data.csv' for processed data.")