#temaperature analysis
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('cleaned_gps_data.csv')

# Filter data for the specified altitude range
df_filtered = df[(df['gps_alt_m'] >= 26870.0) & (df['gps_alt_m'] <= 38674.0)].copy()

# Prepare features and target
X = df_filtered[['gps_alt_m', 'pressure_hpa', 'humidity_%.1']]
y = df_filtered['temperature_c_celsius']

# 1. Anomaly Detection
def anomaly_detection(X, contamination=0.1):
    # Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    if_anomalies = iso_forest.fit_predict(X)

    # One-Class SVM
    ocsvm = OneClassSVM(nu=contamination)
    ocsvm_anomalies = ocsvm.fit_predict(X)

    return if_anomalies, ocsvm_anomalies

if_anomalies, ocsvm_anomalies = anomaly_detection(X)
df_filtered.loc[:, 'if_anomaly'] = if_anomalies
df_filtered.loc[:, 'ocsvm_anomaly'] = ocsvm_anomalies

# 2. Regression Analysis
def polynomial_regression(X, y, degree=3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    return model, mse

poly_model, poly_mse = polynomial_regression(X['gps_alt_m'].values.reshape(-1, 1), y)

# 3. Time Series Analysis
def lstm_analysis(X, y, lookback=10):
    # Prepare data for LSTM
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create sequences
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - lookback):
        X_seq.append(X_scaled[i:i+lookback])
        y_seq.append(y.iloc[i+lookback])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(lookback, X.shape[1])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=0)

    return model, history

lstm_model, lstm_history = lstm_analysis(X, y)

# 4. Clustering
def dbscan_clustering(X, eps=0.5, min_samples=5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_scaled)

    return clusters

clusters = dbscan_clustering(X)
df_filtered.loc[:, 'cluster'] = clusters

# 5. Feature Importance
def feature_importance(X, y):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    return importance

feature_imp = feature_importance(X, y)

# Visualizations
plt.figure(figsize=(20, 20))

# Anomaly Detection
plt.subplot(3, 2, 1)
plt.scatter(df_filtered['gps_alt_m'], df_filtered['temperature_c_celsius'], c='black', alpha=0.5)
plt.scatter(df_filtered['gps_alt_m'][if_anomalies == -1], df_filtered['temperature_c_celsius'][if_anomalies == -1], c='red')
plt.title('Isolation Forest Anomalies')
plt.xlabel('Altitude (m)')
plt.ylabel('Temperature (°C)')

plt.subplot(3, 2, 2)
plt.scatter(df_filtered['gps_alt_m'], df_filtered['temperature_c_celsius'], c='black', alpha=0.5)
plt.scatter(df_filtered['gps_alt_m'][ocsvm_anomalies == -1], df_filtered['temperature_c_celsius'][ocsvm_anomalies == -1], c='red')
plt.title('One-Class SVM Anomalies')
plt.xlabel('Altitude (m)')
plt.ylabel('Temperature (°C)')

# Regression Analysis
plt.subplot(3, 2, 3)
X_plot = np.linspace(26870, 38674, 100).reshape(-1, 1)
plt.scatter(X['gps_alt_m'], y, c='black', alpha=0.5)
plt.plot(X_plot, poly_model.predict(X_plot), color='r')
plt.title(f'Polynomial Regression (MSE: {poly_mse:.2f})')
plt.xlabel('Altitude (m)')
plt.ylabel('Temperature (°C)')

# Time Series Analysis
plt.subplot(3, 2, 4)
plt.plot(lstm_history.history['loss'], label='Training Loss')
plt.plot(lstm_history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Clustering
plt.subplot(3, 2, 5)
scatter = plt.scatter(df_filtered['gps_alt_m'], df_filtered['temperature_c_celsius'], c=df_filtered['cluster'], cmap='viridis')
plt.colorbar(scatter)
plt.title('DBSCAN Clustering')
plt.xlabel('Altitude (m)')
plt.ylabel('Temperature (°C)')

# Feature Importance
plt.subplot(3, 2, 6)
feature_imp.plot(x='feature', y='importance', kind='bar', ax=plt.gca())
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')

plt.tight_layout()
plt.savefig('temperature_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Analysis complete. Check 'temperature_analysis.png' for visualizations.")
print("\nFeature Importance:")
print(feature_imp)
print(f"\nPolynomial Regression MSE: {poly_mse:.4f}")
print(f"Number of anomalies detected (Isolation Forest): {sum(if_anomalies == -1)}")
print(f"Number of anomalies detected (One-Class SVM): {sum(ocsvm_anomalies == -1)}")
print(f"Number of clusters detected (DBSCAN): {len(set(clusters))}")