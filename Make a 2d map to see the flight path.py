#Make a 2d map to see the flight path
import pandas as pd
import folium
from folium import plugins
from geopy.distance import geodesic
import numpy as np
from IPython.display import display, HTML

# File name
file_name = 'cleaned_gps_data.csv'

# Read the CSV file
df = pd.read_csv(file_name)

# Remove rows with NaN values in lat or lon
df_clean = df.dropna(subset=['lat', 'lon'])

# Ensure longitude is negative for western hemisphere
df_clean['lon'] = -df_clean['lon'].abs()

# Create a map centered on the mean coordinates
center_lat = df_clean['lat'].mean()
center_lon = df_clean['lon'].mean()
flight_map = folium.Map(location=[center_lat, center_lon], zoom_start=8)

# Create a list of coordinates
coordinates = df_clean[['lat', 'lon']].values.tolist()

# Add the flight path to the map
folium.PolyLine(coordinates, color="red", weight=2, opacity=0.8).add_to(flight_map)

# Add markers for start and end points
start_point = df_clean.iloc[0]
end_point = df_clean.iloc[-1]

folium.Marker(
    [start_point['lat'], start_point['lon']],
    popup=f"Start: {start_point['time_actual']}",
    icon=folium.Icon(color='green', icon='play')
).add_to(flight_map)

folium.Marker(
    [end_point['lat'], end_point['lon']],
    popup=f"End: {end_point['time_actual']}",
    icon=folium.Icon(color='red', icon='stop')
).add_to(flight_map)

# Add a minimap
minimap = plugins.MiniMap()
flight_map.add_child(minimap)

# Add layer control
folium.LayerControl().add_to(flight_map)

# Display the map in Colab
display(flight_map)

# Print some statistics
print(f"\nTotal data points: {len(df)}")
print(f"Valid GPS data points: {len(df_clean)}")
print(f"\nStart point:")
print(f"Time: {start_point['time_actual']}")
print(f"GPS: LAT {start_point['lat']:.4f}, LON {start_point['lon']:.4f}")
print(f"Altitude: {start_point['gps_alt_m']:.2f} m")

print(f"\nEnd point:")
print(f"Time: {end_point['time_actual']}")
print(f"GPS: LAT {end_point['lat']:.4f}, LON {end_point['lon']:.4f}")
print(f"Altitude: {end_point['gps_alt_m']:.2f} m")

duration = (pd.to_datetime(end_point['time_actual']) - pd.to_datetime(start_point['time_actual'])).total_seconds()
print(f"\nFlight duration: {duration/60:.2f} minutes")

# Calculate total distance (this is a rough approximation)
total_distance = sum(geodesic(coord1, coord2).kilometers for coord1, coord2 in zip(coordinates[:-1], coordinates[1:]))
print(f"Approximate total distance traveled: {total_distance:.2f} km")

# Calculate altitude change
altitude_change = end_point['gps_alt_m'] - start_point['gps_alt_m']
print(f"Total altitude change: {altitude_change:.2f} m")

# Find maximum altitude
max_altitude = df_clean['gps_alt_m'].max()
max_altitude_time = df_clean.loc[df_clean['gps_alt_m'].idxmax(), 'time_actual']
print(f"Maximum altitude: {max_altitude:.2f} m (reached at {max_altitude_time})")