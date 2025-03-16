#Make clean data into a table
import pandas as pd
from tabulate import tabulate

# File name
file_name = 'cleaned_gps_data.csv'

# Read the CSV file
df = pd.read_csv(file_name)

# Convert timems to seconds
df['Time(s)'] = df['timems'] / 1000

# Select the first 10 rows and relevant columns
table_data = df[['Time(s)', 'lat', 'lon', 'gps_alt_m', 'co2_ppm', 'ethanol_raw', 'h2_raw', 'tvoc_ppb']].head(10)

# Rename columns for clarity
table_data.columns = ['Time (s)', 'Latitude', 'Longitude', 'Altitude (m)', 'CO2 (ppm)', 'Ethanol (raw)', 'H2 (raw)', 'TVOC (ppb)']

# Round Time(s) to 2 decimal places
table_data['Time (s)'] = table_data['Time (s)'].round(2)

# Display the table
print(tabulate(table_data, headers='keys', tablefmt='pretty', showindex=False))

# Print some statistics
print(f"\nTotal data points: {len(df)}")
print(f"\nFirst data point:")
print(f"Time: {df['Time(s)'].iloc[0]:.2f}s")
print(f"GPS: LAT {df['lat'].iloc[0]}, LON {df['lon'].iloc[0]}")
print(f"Altitude: {df['gps_alt_m'].iloc[0]} m")
print(f"CO2: {df['co2_ppm'].iloc[0]} ppm")
print(f"Ethanol: {df['ethanol_raw'].iloc[0]} raw")
print(f"H2: {df['h2_raw'].iloc[0]} raw")
print(f"TVOC: {df['tvoc_ppb'].iloc[0]} ppb")

print(f"\nLast data point:")
print(f"Time: {df['Time(s)'].iloc[-1]:.2f}s")
print(f"GPS: LAT {df['lat'].iloc[-1]}, LON {df['lon'].iloc[-1]}")
print(f"Altitude: {df['gps_alt_m'].iloc[-1]} m")
print(f"CO2: {df['co2_ppm'].iloc[-1]} ppm")
print(f"Ethanol: {df['ethanol_raw'].iloc[-1]} raw")
print(f"H2: {df['h2_raw'].iloc[-1]} raw")
print(f"TVOC: {df['tvoc_ppb'].iloc[-1]} ppb")

start_time = df['Time(s)'].iloc[0]
end_time = df['Time(s)'].iloc[-1]
duration = end_time - start_time
print(f"\nTime range: {start_time:.2f}s - {end_time:.2f}s")
print(f"Total duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")

# Function to calculate and print range statistics
def print_range_stats(column, label, unit):
    min_val = df[column].min()
    max_val = df[column].max()
    range_val = max_val - min_val
    print(f"\n{label} range: {min_val:.2f}{unit} - {max_val:.2f}{unit}")
    print(f"{label} variation: {range_val:.2f}{unit}")

print_range_stats('gps_alt_m', 'Altitude', 'm')
print_range_stats('co2_ppm', 'CO2', 'ppm')
print_range_stats('ethanol_raw', 'Ethanol', ' raw')
print_range_stats('h2_raw', 'H2', ' raw')
print_range_stats('tvoc_ppb', 'TVOC', 'ppb')