#UV and Light Analysis Visualization Script
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import pytz

def load_data(file_path):
    return pd.read_csv(file_path)

def convert_to_mountain_time(df):
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['time_actual'])

    # Assume the original time is in UTC
    utc = pytz.UTC
    mountain = pytz.timezone('US/Mountain')

    # Convert to Mountain Time
    df['timestamp'] = df['timestamp'].dt.tz_localize(utc).dt.tz_convert(mountain)

    return df

def calculate_solar_elevation(df):
    # Extract day of year and hour
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['hour'] = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60

    # Calculate solar declination
    df['solar_declination'] = 23.45 * np.sin(np.radians(360/365 * (df['day_of_year'] - 81)))

    # Calculate hour angle
    df['hour_angle'] = 15 * (df['hour'] - 12)

    # Calculate solar elevation
    df['solar_elevation'] = np.degrees(np.arcsin(
        np.sin(np.radians(df['lat'])) * np.sin(np.radians(df['solar_declination'])) +
        np.cos(np.radians(df['lat'])) * np.cos(np.radians(df['solar_declination'])) * np.cos(np.radians(df['hour_angle']))
    ))

    return df['solar_elevation']

def create_visualizations(df):
    # Calculate solar elevation
    df['solar_elevation'] = calculate_solar_elevation(df)

    # Set up the plot
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle('UV and Light Analysis (Mountain Time)', fontsize=16)

    # 1. UV Intensity vs. Altitude
    axs[0, 0].plot(df['gps_alt_m'], df['uv_index'])
    axs[0, 0].set_xlabel('Altitude (meters)')
    axs[0, 0].set_ylabel('UV Index')
    axs[0, 0].set_title('UV Intensity vs. Altitude')

    # 2. Ambient Light vs. Altitude with Solar Elevation
    ax2 = axs[0, 1].twinx()
    axs[0, 1].plot(df['gps_alt_m'], df['light_lux'], color='blue', label='Ambient Light')
    ax2.plot(df['gps_alt_m'], df['solar_elevation'], color='red', label='Solar Elevation')
    axs[0, 1].set_xlabel('Altitude (meters)')
    axs[0, 1].set_ylabel('Ambient Light (lux)', color='blue')
    ax2.set_ylabel('Solar Elevation Angle (degrees)', color='red')
    axs[0, 1].set_title('Ambient Light vs. Altitude with Solar Elevation')
    lines1, labels1 = axs[0, 1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # 3. UV Intensity vs. Ambient Light
    axs[1, 0].scatter(df['uv_index'], df['light_lux'])
    axs[1, 0].set_xlabel('UV Index')
    axs[1, 0].set_ylabel('Ambient Light (lux)')
    axs[1, 0].set_title('UV Intensity vs. Ambient Light')

    # 4. Diurnal Variations of UV and Ambient Light with Solar Elevation
    ax4 = axs[1, 1].twinx()
    axs[1, 1].plot(df['timestamp'], df['uv_index'], color='green', label='UV Index')
    axs[1, 1].plot(df['timestamp'], df['light_lux'], color='blue', label='Ambient Light')
    ax4.plot(df['timestamp'], df['solar_elevation'], color='red', label='Solar Elevation')
    axs[1, 1].set_xlabel('Time of Day (Mountain Time)')
    axs[1, 1].set_ylabel('UV Index / Ambient Light (lux)')
    ax4.set_ylabel('Solar Elevation Angle (degrees)')
    axs[1, 1].set_title('Diurnal Variations of UV and Ambient Light with Solar Elevation')
    lines1, labels1 = axs[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax4.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Format x-axis for the time plot
    axs[1, 1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M', tz=pytz.timezone('US/Mountain')))
    plt.setp(axs[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('uv_light_analysis_mountain_time.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'uv_light_analysis_mountain_time.png'")

def main():
    file_path = 'cleaned_gps_data.csv'
    df = load_data(file_path)
    df = convert_to_mountain_time(df)
    create_visualizations(df)

if __name__ == "__main__":
    main()