Atmospheric Data Analysis Project
Overview
This repository contains a collection of Python scripts for analyzing, visualizing, and processing atmospheric data collected during high-altitude balloon flights. The project focuses on various aspects of atmospheric data analysis, including flight path reconstruction, atmospheric layer identification, gas concentration analysis, and data visualization.
Features
- **Flight Path Reconstruction**: Convert accelerometer data into a 3D flight path
- **Atmospheric Layer Analysis**: Identify and analyze different atmospheric layers using clustering techniques
- **Gas Concentration Analysis**: Analyze CO2, TVOC, and other gas concentrations at different altitudes
- **Data Fusion**: Combine multiple data sources and calculate derived metrics
- **Visualization Tools**: Create 2D/3D plots, KML files for Google Earth, and comprehensive heatmaps
Scripts
Data Processing
- `clean_the_adjusted_data.py`: Remove outliers from flight data
- `cleaned_flight_data.py`: Process flight data and display column information
- `Data_fusion.py`: Combine data sources and calculate derived parameters like air density
- `processed_flight_data_with_indices.py`: Add calculated indices to processed data
- `Take raw data and clear it useing the Z.py`: Clean raw data using Z-score method for outlier detection
Flight Path Analysis
- `2d_flight_path.py`: Reconstruct flight path from accelerometer data
- `KML_file.py`: Generate KML files for visualization in Google Earth
- `Make_a_2d_map_to_see_the_flight_path.py`: Create an interactive map to visualize the flight path
- `Make_kml_of_the_path_and_the_four_KML_f.py`: Generate KML files with color-coded gas concentration data
Atmospheric Analysis
- `atmospheric_layers_processed_balloon_d.py`: Cluster and analyze atmospheric layers
- `Define_Key_Characteristics.py`: Identify key characteristics of atmospheric layers
- `gas_concentration_analysis.py`: Analyze gas concentration patterns
- `LayerStatistics.py`: Calculate and visualize statistics for each atmospheric layer
- `PCA_analysis.py`: Apply principal component analysis to atmospheric variables
- `temaperature analysis.py`: Analyze temperature data with anomaly detection and clustering
Regression Analysis
- `Regression analysis.py`: Perform regression analysis on atmospheric data
- `Regression analysis and plot.py`: Create plots for regression analysis results
- `UV Index regression analysis.py`: Analyze UV index using regression models
- `UV and Light Analysis Visualization Scr.py`: Visualize UV and light data with solar elevation
Visualization and Data Preparation
- `Available_columns_in_the_dataset.py`: Display available data columns and create sample plots
- `Heatmap.py`: Generate comprehensive correlation heatmaps
- `Make_clean_data_into_a_table.py`: Create clean tabular data for reporting
- `Merged_Data_Info.py`: Display information about merged datasets

Requirements
- Python 3.7+
- pandas
- numpy
- matplotlib
- scikit-learn
- tensorflow
- folium
- simplekml
- pyproj
- seaborn
- scipy

Installation
Clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/atmospheric-data-analysis.git
cd atmospheric-data-analysis
pip install -r requirements.txt
```

Usage
Most scripts are designed to work with CSV data files. The primary input files expected are:
- `cleaned_flight_data.csv`: Processed flight sensor data
- `cleaned_gps_data.csv`: GPS and atmospheric measurement data
Example usage:

```bash
# Reconstruct flight path
python 2d_flight_path.py

# Generate KML file for Google Earth visualization
python KML_file.py

# Analyze atmospheric layers
python atmospheric_layers_processed_balloon_d.py
```

Data Format
The scripts expect CSV files with specific columns. Key columns include:
- Time data: `Time(ms)`, `timems`
- Location: `lat`, `lon`, `gps_alt_m`
- Atmospheric: `temperature_c_celsius`, `pressure_hpa`, `humidity_%.1`
- Gas concentrations: `co2_ppm`, `tvoc_ppb`, `ethanol_raw`, `h2_raw`
- Accelerometer: `Accel X (mg)`, `Accel Y (mg)`, `Accel Z (mg)`
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
License
This project is licensed under the MIT License - see the LICENSE file for details.
