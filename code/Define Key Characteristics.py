#Define Key Characteristics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import stats

# Load the data
flight_data = pd.read_csv('cleaned_flight_data.csv')
gps_data = pd.read_csv('cleaned_gps_data.csv')

# Convert 'Time(ms)' and 'timems' to datetime
flight_data['timestamp'] = pd.to_datetime(flight_data['Time(ms)'], unit='ms')
gps_data['timestamp'] = pd.to_datetime(gps_data['timems'], unit='ms')

# Set timestamp as index for both dataframes
flight_data.set_index('timestamp', inplace=True)
gps_data.set_index('timestamp', inplace=True)

# Merge the datasets based on the timestamp index
merged_data = pd.merge_asof(gps_data, flight_data, left_index=True, right_index=True, direction='nearest', tolerance=pd.Timedelta('1s'))

# Function to calculate lapse rate
def calculate_lapse_rate(temp, altitude):
    return np.polyfit(altitude, temp, 1)[0] * 1000  # Convert to °C/km

# Function to calculate layer statistics
def calculate_layer_stats(data):
    return {
        'avg_temp': np.mean(data['temperature_c_celsius']),
        'temp_std': np.std(data['temperature_c_celsius']),
        'lapse_rate': calculate_lapse_rate(data['temperature_c_celsius'], data['gps_alt_m']),
        'avg_humidity': np.mean(data['humidity_%']),
        'humidity_std': np.std(data['humidity_%']),
        'avg_pressure': np.mean(data['pressure_pa']),
        'pressure_std': np.std(data['pressure_pa']),
        'avg_co2': np.mean(data['co2_ppm']),
        'co2_std': np.std(data['co2_ppm']),
        'avg_tvoc': np.mean(data['tvoc_ppb']),
        'tvoc_std': np.std(data['tvoc_ppb']),
    }

# Perform K-means clustering
X = merged_data[['gps_alt_m', 'temperature_c_celsius', 'humidity_%', 'pressure_pa', 'co2_ppm', 'tvoc_ppb']].dropna()
kmeans = KMeans(n_clusters=5, random_state=42)  # Assuming 5 layers
X['layer'] = kmeans.fit_predict(X)

# Calculate statistics for each layer
layer_stats = X.groupby('layer').apply(calculate_layer_stats).reset_index()

# Create a single figure with all plots
fig = plt.figure(figsize=(30, 25))
fig.suptitle('Atmospheric Layer Analysis', fontsize=24)

# Box plots
variables = ['temperature_c_celsius', 'humidity_%', 'pressure_pa', 'co2_ppm', 'tvoc_ppb']
for i, var in enumerate(variables):
    ax = fig.add_subplot(3, 5, i+1)
    X.boxplot(column=var, by='layer', ax=ax)
    ax.set_title(f'{var.capitalize()} by Layer', fontsize=12)
    ax.set_xlabel('Layer', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=8)

# Line graphs
for i, var in enumerate(variables):
    ax = fig.add_subplot(3, 5, i+6)
    for layer in X['layer'].unique():
        layer_data = X[X['layer'] == layer]
        ax.scatter(layer_data['gps_alt_m'], layer_data[var], alpha=0.1, s=1, label=f'Layer {layer}')
    ax.set_xlabel('Altitude (m)', fontsize=10)
    ax.set_ylabel(var.capitalize(), fontsize=10)
    ax.set_title(f'{var.capitalize()} vs Altitude', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(fontsize=6)

# Vertical profiles
for i, var in enumerate(['temperature_c_celsius', 'humidity_%', 'pressure_pa']):
    ax = fig.add_subplot(3, 5, i+11)
    scatter = ax.scatter(X[var], X['gps_alt_m'], c=X['layer'], cmap='viridis', alpha=0.5, s=1)
    ax.set_xlabel(var.capitalize(), fontsize=10)
    ax.set_ylabel('Altitude (m)', fontsize=10)
    ax.set_title(f'{var.capitalize()} Profile', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=8)

# Add colorbar for the vertical profiles
cbar_ax = fig.add_subplot(3, 5, 15)
cbar = plt.colorbar(scatter, cax=cbar_ax)
cbar.set_label('Layer', fontsize=10)
cbar.ax.tick_params(labelsize=8)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('atmospheric_layer_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Analysis complete. The graph has been saved as 'atmospheric_layer_analysis.png'.")

# Print layer statistics
print("\nLayer Statistics:")
print(layer_stats)