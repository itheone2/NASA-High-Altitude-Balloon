#gas_concentration_analysis
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('cleaned_gps_data.csv')

# Filter data for the specified altitude range
df_filtered = df[(df['gps_alt_m'] >= 26870.0) & (df['gps_alt_m'] <= 38674.0)].copy()

# Ensure the data is sorted by timestamp
df_filtered['timestamp'] = pd.to_datetime(df_filtered['timems'], unit='ms')
df_filtered = df_filtered.sort_values('timestamp')

# 1. Time Series Analysis

def lstm_analysis(data, feature, lookback=10):
    # Prepare data for LSTM
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[[feature]])

    X, y = [], []
    for i in range(len(scaled_data) - lookback):
        X.append(scaled_data[i:(i + lookback), 0])
        y.append(scaled_data[i + lookback, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the LSTM model
    model = Sequential([
        Input(shape=(lookback, 1)),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=0)

    return model, history

def moving_average(data, feature, window=5):
    return data[feature].rolling(window=window).mean()

# Perform time series analysis for CO2 and TVOC
lstm_co2, history_co2 = lstm_analysis(df_filtered, 'co2_ppm')
lstm_tvoc, history_tvoc = lstm_analysis(df_filtered, 'tvoc_ppb')

ma_co2 = moving_average(df_filtered, 'co2_ppm')
ma_tvoc = moving_average(df_filtered, 'tvoc_ppb')

# 2. Correlation Analysis

def random_forest_importance(data, target):
    features = ['gps_alt_m', 'temperature_c_celsius', 'pressure_hpa', 'humidity_%.1', 'gas_resistance_ohms', 'gas_index']
    X = data[features]
    y = data[target]

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importance = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    return importance

# Get feature importance for CO2 and TVOC
importance_co2 = random_forest_importance(df_filtered, 'co2_ppm')
importance_tvoc = random_forest_importance(df_filtered, 'tvoc_ppb')

# 3. Anomaly Detection

def isolation_forest_anomalies(data, features):
    X = data[features]
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomalies = iso_forest.fit_predict(X)
    return anomalies

# Detect anomalies in CO2 and TVOC data
anomalies_co2 = isolation_forest_anomalies(df_filtered, ['co2_ppm', 'gps_alt_m', 'temperature_c_celsius', 'pressure_hpa', 'humidity_%.1', 'gas_resistance_ohms', 'gas_index'])
anomalies_tvoc = isolation_forest_anomalies(df_filtered, ['tvoc_ppb', 'gps_alt_m', 'temperature_c_celsius', 'pressure_hpa', 'humidity_%.1', 'gas_resistance_ohms', 'gas_index'])

# Visualizations
plt.figure(figsize=(20, 20))

# Time Series Analysis - LSTM
plt.subplot(3, 2, 1)
plt.plot(history_co2.history['loss'], label='Training Loss (CO2)')
plt.plot(history_co2.history['val_loss'], label='Validation Loss (CO2)')
plt.plot(history_tvoc.history['loss'], label='Training Loss (TVOC)')
plt.plot(history_tvoc.history['val_loss'], label='Validation Loss (TVOC)')
plt.title('LSTM Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Time Series Analysis - Moving Average
plt.subplot(3, 2, 2)
plt.plot(df_filtered['timestamp'], df_filtered['co2_ppm'], label='Actual CO2', alpha=0.5)
plt.plot(df_filtered['timestamp'], ma_co2, label='MA CO2')
plt.plot(df_filtered['timestamp'], df_filtered['tvoc_ppb'], label='Actual TVOC', alpha=0.5)
plt.plot(df_filtered['timestamp'], ma_tvoc, label='MA TVOC')
plt.title('Moving Average')
plt.xlabel('Timestamp')
plt.ylabel('Concentration')
plt.legend()

# Correlation Analysis
plt.subplot(3, 2, 3)
importance_co2.plot(x='feature', y='importance', kind='bar', ax=plt.gca())
plt.title('Feature Importance for CO2')
plt.xlabel('Features')
plt.ylabel('Importance')

plt.subplot(3, 2, 4)
importance_tvoc.plot(x='feature', y='importance', kind='bar', ax=plt.gca())
plt.title('Feature Importance for TVOC')
plt.xlabel('Features')
plt.ylabel('Importance')

# Anomaly Detection
plt.subplot(3, 2, 5)
plt.scatter(df_filtered['timestamp'], df_filtered['co2_ppm'], c='black', alpha=0.5)
plt.scatter(df_filtered['timestamp'][anomalies_co2 == -1], df_filtered['co2_ppm'][anomalies_co2 == -1], c='red')
plt.title('CO2 Anomalies')
plt.xlabel('Timestamp')
plt.ylabel('CO2 Concentration (ppm)')

plt.subplot(3, 2, 6)
plt.scatter(df_filtered['timestamp'], df_filtered['tvoc_ppb'], c='black', alpha=0.5)
plt.scatter(df_filtered['timestamp'][anomalies_tvoc == -1], df_filtered['tvoc_ppb'][anomalies_tvoc == -1], c='red')
plt.title('TVOC Anomalies')
plt.xlabel('Timestamp')
plt.ylabel('TVOC Concentration (ppb)')

plt.tight_layout()
plt.savefig('gas_concentration_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Print summary statistics
print("Analysis complete. Check 'gas_concentration_analysis.png' for visualizations.")
print("\nFeature Importance for CO2:")
print(importance_co2)
print("\nFeature Importance for TVOC:")
print(importance_tvoc)
print(f"\nNumber of CO2 anomalies detected: {sum(anomalies_co2 == -1)}")
print(f"Number of TVOC anomalies detected: {sum(anomalies_tvoc == -1)}")

# Data Quality Check
print("\nData Quality Summary:")
print(df_filtered[['co2_ppm', 'tvoc_ppb']].describe())
print("\nMissing Values:")
print(df_filtered[['co2_ppm', 'tvoc_ppb']].isnull().sum())
print("\nUnique Values:")
print(df_filtered[['co2_ppm', 'tvoc_ppb']].nunique())

# Correlation matrix
correlation_matrix = df_filtered[['co2_ppm', 'tvoc_ppb', 'gps_alt_m', 'temperature_c_celsius', 'pressure_hpa', 'humidity_%.1', 'gas_resistance_ohms', 'gas_index']].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nCorrelation matrix saved as 'correlation_matrix.png'")