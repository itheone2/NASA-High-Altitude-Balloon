#PCA analysis
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('flight_data_fused_robust.csv')

# Select variables for PCA
variables = [
    'Temperature (C)', 'Pressure (Pa)', 'air_density',
    'UV Index', 'Light (lux)', 'Humidity (%)'
]

# Check if all variables exist in the dataframe
missing_vars = [var for var in variables if var not in df.columns]
if missing_vars:
    print(f"Warning: The following variables are missing from the dataset: {missing_vars}")
    print("Available columns:")
    print(df.columns.tolist())
    variables = [var for var in variables if var in df.columns]

# Prepare the data
X = df[variables]

# Standardize the variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA()
pca_result = pca.fit_transform(X_scaled)

# Calculate explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Select components that explain 90% of the variance
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
n_components = np.argmax(cumulative_variance_ratio >= 0.9) + 1

print(f"Number of components explaining 90% of variance: {n_components}")

# Create a DataFrame with PCA results
pca_df = pd.DataFrame(
    data=pca_result[:, :n_components],
    columns=[f'PC{i+1}' for i in range(n_components)]
)

# Visualization
plt.figure(figsize=(20, 15))

# 1. Scree plot
plt.subplot(2, 2, 1)
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'bo-')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')

# 2. Cumulative explained variance plot
plt.subplot(2, 2, 2)
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'ro-')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Proportion of Variance Explained')
plt.axhline(y=0.9, color='k', linestyle='--')
plt.text(0.5, 0.85, '90% explained variance', fontsize=12)

# 3. Biplot of PC1 and PC2
plt.subplot(2, 2, 3)
for i, var in enumerate(variables):
    plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], color='r', alpha=0.5)
    plt.text(pca.components_[0, i] * 1.15, pca.components_[1, i] * 1.15, var, color='g', ha='center', va='center')
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.2)
plt.title('Biplot of PC1 and PC2')
plt.xlabel('PC1')
plt.ylabel('PC2')

# 4. Heatmap of component loadings
plt.subplot(2, 2, 4)
loadings = pd.DataFrame(
    pca.components_[:n_components, :].T,
    columns=[f'PC{i+1}' for i in range(n_components)],
    index=variables
)
sns.heatmap(loadings, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Component Loadings Heatmap')

plt.tight_layout()
plt.savefig('PCA_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("PCA analysis complete. Results saved as 'PCA_analysis.png'")

# Additional textual output
print("\nExplained variance ratio for each component:")
for i, ratio in enumerate(explained_variance_ratio):
    print(f"PC{i+1}: {ratio:.4f}")

print("\nComponent loadings:")
print(loadings)