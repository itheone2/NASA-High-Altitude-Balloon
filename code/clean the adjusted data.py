#clean the adjusted data
import pandas as pd
import numpy as np
from scipy import stats

# Step 1: Load the data
df = pd.read_csv('adjusted_flight_data.csv')

# Step 2: Identify numerical columns
numeric_columns = df.select_dtypes(include=[np.number]).columns

# Step 3: Calculate Z-scores for each numeric column
z_scores = np.abs(stats.zscore(df[numeric_columns]))

# Step 4: Set threshold (usually 3 or 2.5)
threshold = 3

# Step 5: Create a mask of all the outliers
outliers = (z_scores > threshold).any(axis=1)

# Step 6: Remove the outliers
df_cleaned = df[~outliers]

# Step 7: Save the cleaned data
df_cleaned.to_csv('cleaned_flight_data.csv', index=False)

print(f"Original data shape: {df.shape}")
print(f"Cleaned data shape: {df_cleaned.shape}")
print(f"Number of outliers removed: {df.shape[0] - df_cleaned.shape[0]}")