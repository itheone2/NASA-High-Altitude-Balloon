#2d flight path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyproj import Geod
import pandas as pd

def reconstruct_flight_path(acc_x, acc_y, initial_lat, initial_lon, initial_alt,
                            time_interval=1.019, g=9.81):
    # Convert acceleration from mg to m/s^2
    acc_x = acc_x * g / 1000
    acc_y = acc_y * g / 1000
    # Number of data points
    n = len(acc_x)
    # Initialize arrays
    vel_x, vel_y = np.zeros(n), np.zeros(n)
    pos_x, pos_y = np.zeros(n), np.zeros(n)
    # Integrate acceleration to get velocity
    for i in range(1, n):
        vel_x[i] = vel_x[i-1] + acc_x[i-1] * time_interval
        vel_y[i] = vel_y[i-1] + acc_y[i-1] * time_interval
    # Integrate velocity to get position
    for i in range(1, n):
        pos_x[i] = pos_x[i-1] + vel_x[i-1] * time_interval + 0.5 * acc_x[i-1] * time_interval**2
        pos_y[i] = pos_y[i-1] + vel_y[i-1] * time_interval + 0.5 * acc_y[i-1] * time_interval**2
    # Calculate latitude, longitude, and altitude
    geod = Geod(ellps='WGS84')
    lats, lons = np.zeros(n), np.zeros(n)
    lats[0], lons[0] = initial_lat, initial_lon
    alts = initial_alt + pos_y
    for i in range(1, n):
        # Calculate new lat/lon based on horizontal displacement
        lons[i], lats[i], _ = geod.fwd(lons[i-1], lats[i-1],
                                       np.degrees(np.arctan2(pos_x[i]-pos_x[i-1], pos_y[i]-pos_y[i-1])),
                                       np.hypot(pos_x[i]-pos_x[i-1], pos_y[i]-pos_y[i-1]))
    return lats, lons, alts

def plot_flight_path(lats, lons, alts):
    fig = plt.figure(figsize=(15, 5))

    # 2D plot
    ax1 = fig.add_subplot(121)
    ax1.plot(lons, lats)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('2D Flight Path')

    # 3D plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(lons, lats, alts)
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_zlabel('Altitude (m)')
    ax2.set_title('3D Flight Path')

    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load data from CSV
    df = pd.read_csv('cleaned_flight_data.csv')

    # Print column names
    print("Columns in the CSV file:")
    print(df.columns)

    # Set column names based on the actual data
    acc_x_col = 'Accel X (mg)'
    acc_y_col = 'Accel Y (mg)'

    # Extract accelerometer data
    acc_x = df[acc_x_col].values
    acc_y = df[acc_y_col].values

    # Ask user for initial conditions
    initial_lat = float(input("Enter initial latitude (e.g., 34.49): "))
    initial_lon = float(input("Enter initial longitude (e.g., 104.225): "))
    initial_alt = float(input("Enter initial altitude in meters (e.g., 4045.0): "))

    # Reconstruct flight path
    lats, lons, alts = reconstruct_flight_path(acc_x, acc_y, initial_lat, initial_lon, initial_alt)

    # Plot results
    plot_flight_path(lats, lons, alts)