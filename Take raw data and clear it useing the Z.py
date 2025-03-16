#Take raw data and clear it useing the Z-score Method

import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats

def load_data(file_path):
    return pd.read_csv(file_path)

def convert_time(df):
    df['Timestamp'] = pd.to_datetime(df['Time(ms)'], unit='ms')
    df['TIME Actual'] = pd.to_datetime(df['TIME Actual'], format='%H:%M:%S', errors='coerce')
    return df

def convert_gps_coordinates(coord):
    if isinstance(coord, str):
        parts = coord.replace('-', ' ').split()
        if len(parts) == 2:
            degrees = float(parts[0])
            minutes = float(parts[1])
            return degrees + (minutes / 60) * (-1 if degrees < 0 else 1)
    return np.nan

def clean_gps_data(df):
    df['LAT'] = df['LAT'].apply(convert_gps_coordinates)
    df['LON'] = df['LON'].apply(convert_gps_coordinates)
    df.loc[df['LAT'] == 0, 'LAT'] = np.nan
    df.loc[df['LON'] == 0, 'LON'] = np.nan
    return df

def handle_missing_data(df):
    # Replace '0-00.0' with NaN for LAT and LON
    df['LAT'] = df['LAT'].replace('0-00.0', np.nan)
    df['LON'] = df['LON'].replace('0-00.0', np.nan)

    # For other columns, replace 0 with NaN where it doesn't make sense
    zero_to_nan_columns = ['BAR ALT', 'GPS ALT (m)', 'AIRT', 'Humidity (%)', 'Temperature (C)']
    for col in zero_to_nan_columns:
        df[col] = df[col].replace(0, np.nan)

    return df

def calculate_z_score(data):
    return np.abs(stats.zscore(data))

def handle_outliers(df, threshold=3):
    numerical_columns = ['BAR ALT', 'GPS ALT (m)', 'AIRT', 'Temperature (C)', 'Pressure (Pa)', 'Humidity (%)', 'MBS']
    for col in numerical_columns:
        if df[col].dtype in ['int64', 'float64']:
            z_scores = calculate_z_score(df[col])
            df[f'{col}_outlier'] = z_scores > threshold
            print(f"Outliers detected in {col}: {df[f'{col}_outlier'].sum()}")
    return df

def consistency_checks(df):
    # Check altitude consistency
    df['altitude_diff'] = df['GPS ALT (m)'] - df['BAR ALT']
    df['altitude_inconsistent'] = abs(df['altitude_diff']) > 1000  # Flag if difference is more than 1000m

    # Check temperature consistency
    df['temp_diff'] = df['AIRT'] - df['Temperature (C)']
    df['temp_inconsistent'] = abs(df['temp_diff']) > 10  # Flag if difference is more than 10°C

    return df

def standardize_units(df):
    # Convert pressure from Pa to hPa
    df['Pressure (hPa)'] = df['Pressure (Pa)'] / 100

    # Ensure all temperature columns are in Celsius (assuming they already are)
    temp_columns = ['AIRT', 'Temperature (C)']
    for col in temp_columns:
        df[f'{col}_celsius'] = df[col]

    return df

def create_derived_variables(df):
    # Calculate vertical speed (m/s)
    df['vertical_speed'] = df['GPS ALT (m)'].diff() / df['Timestamp'].diff().dt.total_seconds()

    # Calculate dew point (simplified formula)
    df['dew_point'] = df['Temperature (C)'] - ((100 - df['Humidity (%)']) / 5)

    return df

def rename_columns(df):
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    return df

def main():
    file_path = 'gps_with_our_trimmed_data - Sheet1.csv'
    df = load_data(file_path)

    print("Original data shape:", df.shape)
    print("\nOriginal data types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())

    # Apply cleaning steps
    df = convert_time(df)
    df = clean_gps_data(df)
    df = handle_missing_data(df)
    df = handle_outliers(df)
    df = consistency_checks(df)
    df = standardize_units(df)
    df = create_derived_variables(df)
    df = rename_columns(df)

    print("\nCleaned data shape:", df.shape)
    print("\nCleaned data types:")
    print(df.dtypes)
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

    # Save cleaned data
    df.to_csv('cleaned_gps_data.csv', index=False)
    print("\nCleaned data saved to 'cleaned_gps_data.csv'")

if __name__ == "__main__":
    main()