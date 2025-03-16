#Data fusion
import pandas as pd
import numpy as np

def find_column(df, target_column):
    # Simple matching function
    for col in df.columns:
        if target_column.lower() in col.lower():
            return col
    return None

# Load the cleaned data
df = pd.read_csv('cleaned_flight_data.csv')

# Find relevant columns
temp_col = find_column(df, 'temperature')
pressure_col = find_column(df, 'pressure')
humidity_col = find_column(df, 'humidity')

# Air Density Calculation
if temp_col and pressure_col and humidity_col:
    def calculate_air_density(temperature, pressure, humidity):
        R = 287.05  # Specific gas constant for dry air (J/(kg·K))
        return pressure / (R * (temperature + 273.15)) * (1 - 0.378 * humidity / 100)

    df['air_density'] = calculate_air_density(df[temp_col], df[pressure_col], df[humidity_col])
    print("Air density calculated.")
else:
    print("Couldn't calculate air density. Missing required columns.")

# Wind Speed Estimation
accel_x_col = find_column(df, 'acceleration_x')
accel_y_col = find_column(df, 'acceleration_y')
if accel_x_col and accel_y_col:
    def estimate_wind_speed(acceleration_x, acceleration_y):
        return np.sqrt(acceleration_x**2 + acceleration_y**2)

    df['estimated_wind_speed'] = estimate_wind_speed(df[accel_x_col], df[accel_y_col])
    print("Wind speed estimated.")
else:
    print("Couldn't estimate wind speed. Missing required columns.")

# Dew Point Calculation
if temp_col and humidity_col:
    def calculate_dew_point(temperature, humidity):
        a = 17.27
        b = 237.7
        alpha = ((a * temperature) / (b + temperature)) + np.log(humidity / 100.0)
        return (b * alpha) / (a - alpha)

    df['dew_point'] = calculate_dew_point(df[temp_col], df[humidity_col])
    print("Dew point calculated.")
else:
    print("Couldn't calculate dew point. Missing required columns.")

# Save the updated dataframe
df.to_csv('flight_data_fused_robust.csv', index=False)

print("\nData fusion complete. New columns added where possible.")
print("Columns in the final dataset:")
for column in df.columns:
    print(f"- {column}")

# Print the first few rows of the updated dataset
print("\nFirst few rows of the updated dataset:")
print(df.head())