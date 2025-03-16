#processed_flight_data_with_indices
import pandas as pd
import numpy as np
from scipy import signal

def load_data(file_path):
    df = pd.read_csv(file_path)
    print("Columns in the dataset:", df.columns.tolist())
    print("\nSample data:")
    print(df.head())
    return df

def clean_and_prepare_data(df):
    # Convert 'timems' to datetime
    df['timestamp'] = pd.to_datetime(df['timems'], unit='ms')

    # Sort by timestamp
    df = df.sort_values('timestamp')

    return df

def synchronize_data(df):
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove 'timems' from numeric columns if present
    if 'timems' in numeric_cols:
        numeric_cols.remove('timems')

    # Set timestamp as index and resample only numeric columns
    resampled = df.set_index('timestamp')[numeric_cols].resample('1s').mean()

    # Merge back with non-numeric columns
    non_numeric_cols = [col for col in df.columns if col not in numeric_cols and col != 'timestamp']
    merged = pd.merge_asof(resampled.reset_index(), df[['timestamp'] + non_numeric_cols], on='timestamp')

    return merged

def calculate_air_density(temperature, pressure):
    # Temperature in Celsius, pressure in hPa
    return pressure * 100 / (287.05 * (temperature + 273.15))

def estimate_wind_speed(accel_x, accel_y, accel_z):
    # This is a simplistic model and may need refinement
    total_accel = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    return total_accel * 0.1  # Scaling factor, adjust as needed

def calculate_dew_point(temperature, relative_humidity):
    a = 17.27
    b = 237.7
    alpha = ((a * temperature) / (b + temperature)) + np.log(relative_humidity/100.0)
    return (b * alpha) / (a - alpha)

def estimate_cloud_base_height(temperature, dew_point, altitude):
    # Assuming a lapse rate of 6.5°C per 1000m
    dew_point_depression = temperature - dew_point
    return altitude + (dew_point_depression / 6.5) * 1000

def calculate_air_quality_index(co2, tvoc):
    # This is a simplified index and should be adjusted based on specific sensor ranges and air quality standards
    return (co2 / 1000) + (tvoc / 100)  # Normalize and combine

def calculate_stability_index(accel_z, pressure):
    # This is a simplified index. You might want to refine this based on your specific requirements.
    accel_variability = np.abs(np.diff(accel_z))
    pressure_variability = np.abs(np.diff(pressure))
    return np.mean(accel_variability) / np.mean(pressure_variability)

def process_data(df):
    # Synchronize data
    df = synchronize_data(df)

    # Calculate air density
    df['air_density'] = calculate_air_density(df['temperature_c_celsius'], df['pressure_hpa'])

    # Estimate wind speed
    df['wind_speed'] = estimate_wind_speed(df['accel_x_mg'], df['accel_y_mg'], df['accel_z_mg'])

    # Calculate dew point (using existing dew_point column if available)
    if 'dew_point' not in df.columns:
        df['dew_point'] = calculate_dew_point(df['temperature_c_celsius'], df['humidity_%.1'])

    # Estimate cloud base height
    df['cloud_base_height'] = estimate_cloud_base_height(df['temperature_c_celsius'], df['dew_point'], df['gps_alt_m'])

    # Calculate air quality index
    df['air_quality_index'] = calculate_air_quality_index(df['co2_ppm'], df['tvoc_ppb'])

    # Calculate stability index
    df['stability_index'] = calculate_stability_index(df['accel_z_mg'], df['pressure_hpa'])

    return df

def main():
    file_path = 'cleaned_gps_data.csv'
    df = load_data(file_path)

    print("Original data shape:", df.shape)

    df_cleaned = clean_and_prepare_data(df)
    print("\nCleaned data shape:", df_cleaned.shape)

    df_processed = process_data(df_cleaned)

    print("\nProcessed data shape:", df_processed.shape)
    print("\nNew columns added:")
    new_columns = ['air_density', 'wind_speed', 'cloud_base_height', 'air_quality_index', 'stability_index']
    for col in new_columns:
        if col in df_processed.columns:
            print(f"{col}:")
            print(df_processed[col].describe())
            print()
        else:
            print(f"{col} not found in processed data.")

    # Save processed data
    df_processed.to_csv('processed_flight_data_with_indices.csv', index=False)
    print("\nProcessed data saved to 'processed_flight_data_with_indices.csv'")

if __name__ == "__main__":
    main()