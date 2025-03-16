#KML file
import pandas as pd
import numpy as np
from pyproj import Geod
import simplekml

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

def create_kml(lats, lons, alts, output_file='flight_path.kml'):
    kml = simplekml.Kml()

    # Create a new line string
    linestring = kml.newlinestring(name="Flight Path")

    # Add coordinates to the line string
    for lat, lon, alt in zip(lats, lons, alts):
        linestring.coords.addcoordinates([(lon, lat, alt)])

    # Set the altitude mode to absolute (altitude is relative to sea level)
    linestring.altitudemode = simplekml.AltitudeMode.absolute

    # Set line style
    linestring.style.linestyle.width = 4
    linestring.style.linestyle.color = simplekml.Color.blue

    # Save the KML file
    kml.save(output_file)
    print(f"KML file has been saved as {output_file}")

def validate_coordinates(lat, lon):
    if not (-90 <= lat <= 90):
        raise ValueError("Latitude must be between -90 and 90 degrees.")
    if not (-180 <= lon <= 180):
        raise ValueError("Longitude must be between -180 and 180 degrees.")

# Main execution
if __name__ == "__main__":
    # Load data from CSV
    df = pd.read_csv('cleaned_flight_data.csv')

    # Set column names based on the actual data
    acc_x_col = 'Accel X (mg)'
    acc_y_col = 'Accel Y (mg)'

    # Extract accelerometer data
    acc_x = df[acc_x_col].values
    acc_y = df[acc_y_col].values

    while True:
        try:
            # Ask user for initial conditions
            initial_lat = float(input("Enter initial latitude (-90 to 90, e.g., 34.49): "))
            initial_lon = float(input("Enter initial longitude (-180 to 180, e.g., -104.225): "))
            validate_coordinates(initial_lat, initial_lon)
            initial_alt = float(input("Enter initial altitude in meters (e.g., 4045.0): "))
            break
        except ValueError as e:
            print(f"Error: {e}")
            print("Please try again.")

    # Reconstruct flight path
    lats, lons, alts = reconstruct_flight_path(acc_x, acc_y, initial_lat, initial_lon, initial_alt)

    # Create KML file
    create_kml(lats, lons, alts)

    print("\nImportant note:")
    print("If the flight path still appears on the wrong side of the Earth,")
    print("try running the script again and entering the longitude as a negative number.")
    print("For example, if you entered 104.225, try -104.225 instead.")