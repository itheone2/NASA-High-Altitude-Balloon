#Available columns in the dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
df = pd.read_csv('flight_data_fused_robust.csv')

# Print available columns
print("Available columns in the dataset:")
for col in df.columns:
    print(f"- {col}")

# Set the style for better-looking plots
plt.style.use('seaborn-v0_8')

# 1. Time Series Plot
time_col = 'Time(ms)'
y_col = 'Temperature (C)'  # You can change this to any other column you're interested in

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df[time_col], df[y_col], label=y_col)
ax.set_xlabel('Time (ms)')
ax.set_ylabel(y_col)
ax.set_title(f'{y_col} over Time')
ax.legend()
plt.savefig('time_series_plot.png')
plt.close()

# 2. Scatter Plot
x_col = 'Temperature (C)'
y_col = 'Pressure (Pa)'
z_col = 'Humidity (%)'

fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(df[x_col], df[y_col], c=df[z_col], cmap='viridis')
ax.set_xlabel(x_col)
ax.set_ylabel(y_col)
ax.set_title(f'{x_col} vs {y_col} (colored by {z_col})')
plt.colorbar(scatter, label=z_col)
plt.savefig('scatter_plot.png')
plt.close()

# 3. Histogram
hist_col = 'air_density'

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df[hist_col], bins=30, edgecolor='black')
ax.set_xlabel(hist_col)
ax.set_ylabel('Frequency')
ax.set_title(f'Distribution of {hist_col}')
plt.savefig('histogram.png')
plt.close()

# 4. Box Plot
numeric_cols = ['Temperature (C)', 'Humidity (%)', 'Pressure (Pa)', 'CO2 (ppm)', 'TVOC (ppb)']
fig, ax = plt.subplots(figsize=(12, 6))
df.boxplot(column=numeric_cols, ax=ax)
ax.set_title('Box Plots of Key Variables')
ax.set_ylabel('Value')
plt.xticks(rotation=45)
plt.savefig('boxplot.png', bbox_inches='tight')
plt.close()

# 5. Correlation Heatmap
numeric_columns = df[['Temperature (C)', 'Humidity (%)', 'Pressure (Pa)', 'CO2 (ppm)', 'TVOC (ppb)', 'air_density', 'dew_point']].columns
correlation_matrix = df[numeric_columns].corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
ax.set_title('Correlation Heatmap of Key Variables')
plt.savefig('correlation_heatmap.png', bbox_inches='tight')
plt.close()

print("\nCharts have been created and saved as PNG files in the current directory.")
print("Created charts:")
print("1. time_series_plot.png")
print("2. scatter_plot.png")
print("3. histogram.png")
print("4. boxplot.png")
print("5. correlation_heatmap.png")