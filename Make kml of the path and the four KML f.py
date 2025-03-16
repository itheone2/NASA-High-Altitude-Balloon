#Make kml of the path and the four KML file co2, ethanol, h2, tvoc with a color scale.s
!pip install simplekml
import pandas as pd
import simplekml
import os
from google.colab import files
import numpy as np
import colorsys

# File name
file_name = 'cleaned_gps_data.csv'

# Check if the file exists
if not os.path.exists(file_name):
    print(f"The file '{file_name}' does not exist in the current directory.")
    print("Please upload your CSV file:")
    uploaded = files.upload()
    file_name = list(uploaded.keys())[0]

# Read the CSV file
df = pd.read_csv(file_name)

# Remove rows with NaN values in lat, lon, or any of the parameters
parameters = ['co2_ppm', 'ethanol_raw', 'h2_raw', 'tvoc_ppb']
df_clean = df.dropna(subset=['lat', 'lon'] + parameters)

# Ensure longitude is negative for western hemisphere
df_clean.loc[:, 'lon'] = -df_clean['lon'].abs()

# Function to get color based on value
def get_rainbow_color(value, min_value, max_value):
    hue = (1 - (value - min_value) / (max_value - min_value)) * 240 / 360
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, 1.0, 1.0)]
    return simplekml.Color.rgb(r, g, b)

# Function to create KML for a specific parameter
def create_kml_for_parameter(parameter, unit):
    kml = simplekml.Kml()

    min_val = df_clean[parameter].min()
    max_val = df_clean[parameter].max()

    # Create a folder for the colored line segments
    line_folder = kml.newfolder(name=f"{parameter} Colored Flight Path")

    # Create colored line segments
    for i in range(len(df_clean) - 1):
        start = df_clean.iloc[i]
        end = df_clean.iloc[i+1]

        line = line_folder.newlinestring(name="")
        line.coords = [(start['lon'], start['lat'], start['gps_alt_m']),
                       (end['lon'], end['lat'], end['gps_alt_m'])]
        line.extrude = 1
        line.altitudemode = simplekml.AltitudeMode.absolute

        # Use average value for the segment color
        avg_val = (start[parameter] + end[parameter]) / 2
        line.style.linestyle.color = get_rainbow_color(avg_val, min_val, max_val)
        line.style.linestyle.width = 4

    # Create a folder for individual points
    points_folder = kml.newfolder(name="GPS Points")

    # Add individual points
    for idx, row in df_clean.iterrows():
        point = points_folder.newpoint(name="")
        point.coords = [(row['lon'], row['lat'], row['gps_alt_m'])]
        point.description = (f"Time: {row['time_actual']}\n"
                             f"Altitude: {row['gps_alt_m']:.2f}m\n"
                             f"{parameter}: {row[parameter]} {unit}")
        point.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
        point.style.iconstyle.color = get_rainbow_color(row[parameter], min_val, max_val)
        point.style.iconstyle.scale = 0.5  # Make the icons smaller

    # Create color scale
    scale_folder = kml.newfolder(name=f"{parameter} Color Scale")
    scale_width = 0.1  # degrees
    scale_height = 1.0  # degrees
    num_segments = 20

    lat_start = df_clean['lat'].min() - 0.5
    lon_start = df_clean['lon'].min() - 0.5

    for i in range(num_segments):
        value = min_val + (max_val - min_val) * i / (num_segments - 1)
        segment = scale_folder.newpolygon(name="")
        segment.outerboundaryis = [(lon_start, lat_start + i * scale_height / num_segments),
                                   (lon_start + scale_width, lat_start + i * scale_height / num_segments),
                                   (lon_start + scale_width, lat_start + (i + 1) * scale_height / num_segments),
                                   (lon_start, lat_start + (i + 1) * scale_height / num_segments),
                                   (lon_start, lat_start + i * scale_height / num_segments)]
        segment.style.polystyle.color = get_rainbow_color(value, min_val, max_val)
        segment.style.polystyle.outline = 0

        label = scale_folder.newpoint(name=f"{value:.0f} {unit}")
        label.coords = [(lon_start + scale_width * 1.5, lat_start + (i + 0.5) * scale_height / num_segments)]
        label.style.iconstyle.icon.href = ''
        label.style.labelstyle.color = simplekml.Color.white
        label.style.labelstyle.scale = 0.8

    # Save the KML file
    kml.save(f"flight_data_{parameter}.kml")
    print(f"KML file 'flight_data_{parameter}.kml' has been created.")

# Create KML files for each parameter
create_kml_for_parameter('co2_ppm', 'ppm')
create_kml_for_parameter('ethanol_raw', 'raw')
create_kml_for_parameter('h2_raw', 'raw')
create_kml_for_parameter('tvoc_ppb', 'ppb')

# Print some statistics
print(f"\nTotal data points: {len(df)}")
print(f"Valid data points: {len(df_clean)}")

start_point = df_clean.iloc[0]
end_point = df_clean.iloc[-1]

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

# Calculate altitude change
altitude_change = end_point['gps_alt_m'] - start_point['gps_alt_m']
print(f"Total altitude change: {altitude_change:.2f} m")

# Find maximum altitude
max_altitude = df_clean['gps_alt_m'].max()
max_altitude_time = df_clean.loc[df_clean['gps_alt_m'].idxmax(), 'time_actual']
print(f"Maximum altitude: {max_altitude:.2f} m (reached at {max_altitude_time})")

# Print statistics for each parameter
for param in parameters:
    min_val = df_clean[param].min()
    max_val = df_clean[param].max()
    avg_val = df_clean[param].mean()
    print(f"\n{param} range: {min_val:.2f} - {max_val:.2f}")
    print(f"Average {param}: {avg_val:.2f}")