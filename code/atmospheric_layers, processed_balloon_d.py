#atmospheric_layers, processed_balloon_data
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

# Perform KMeans clustering
features = ['temperature_c_celsius', 'pressure_hpa', 'humidity_%.1', 'potential_temperature']
X = df[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42)  # Assuming 5 layers, adjust as needed
df.loc[X.index, 'layer'] = kmeans.fit_predict(X_scaled)

# Create a single large figure
plt.figure(figsize=(20, 30))

# Create a custom colormap with purple for Layer 1
colors = ['purple'] + plt.cm.viridis(np.linspace(0, 1, 4)).tolist()
cmap = mcolors.ListedColormap(colors)

# Sort the dataframe by altitude to ensure proper color gradient
df_sorted = df.sort_values('gps_alt_m')

# Create the scatter plot
scatter = plt.scatter(df_sorted['temperature_c_celsius'], df_sorted['gps_alt_m'],
                      c=df_sorted['layer'], cmap=cmap, s=50, alpha=0.7)

# Create a custom colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=4))
sm.set_array([])
cbar = plt.colorbar(sm, label='Layer', pad=0.01, ticks=[0, 1, 2, 3, 4])
cbar.set_ticklabels(['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5'])

plt.xlabel('Temperature (°C)', fontsize=16)
plt.ylabel('Altitude (m)', fontsize=16)
plt.title('Atmospheric Layers', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=14)

# Annotate the layers
for layer in range(5):
    layer_data = df[df['layer'] == layer]
    mid_temp = layer_data['temperature_c_celsius'].median()
    mid_alt = layer_data['gps_alt_m'].median()
    plt.annotate(f'Layer {layer+1}', (mid_temp, mid_alt), fontsize=14,
                 xytext=(5, 5), textcoords='offset points',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('atmospheric_layers.png', dpi=300, bbox_inches='tight')
plt.close()

# Analyze layers
layers = df.groupby('layer')
layer_stats = layers.agg({
    'gps_alt_m': ['min', 'max'],
    'temperature_c_celsius': ['mean', 'std'],
    'pressure_hpa': ['mean', 'std'],
    'humidity_%.1': ['mean', 'std'],
    'potential_temperature': ['mean', 'std']
})

# Print layer statistics
print("Layer Statistics:")
print(layer_stats)

# Save processed data
df.to_csv('processed_balloon_data.csv', index=False)

print("\nAnalysis complete. Check 'atmospheric_layers.png' for the plot and 'processed_balloon_data.csv' for processed data.")