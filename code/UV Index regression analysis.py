#UV Index regression analysis
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('flight_data_fused_robust.csv')

# Function to prepare data for regression
def prepare_data(df, target, features):
    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate regression results
def evaluate_regression(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{model_name} Results for UV Index:")
    print(f"R-squared: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return model, r2, rmse, y_test, y_pred

# Identify available relevant columns
uv_features = [col for col in ['Temperature (C)', 'Pressure (Pa)', 'Humidity (%)', 'Accel X (mg)', 'Accel Y (mg)', 'Accel Z (mg)'] if col in df.columns]
uv_target = 'UV Index'

if uv_target not in df.columns:
    raise ValueError("'UV Index' column is not available in the dataset.")

# Prepare data for UV Index prediction
X_train_uv, X_test_uv, y_train_uv, y_test_uv = prepare_data(df, uv_target, uv_features)

# Create a figure to hold all subplots
fig, axs = plt.subplots(2, 3, figsize=(20, 15))
fig.suptitle('UV Index Prediction Analysis', fontsize=16)

# Linear Regression for UV
lr_uv, lr_r2_uv, lr_rmse_uv, y_test_lr_uv, y_pred_lr_uv = evaluate_regression(LinearRegression(), X_train_uv, X_test_uv, y_train_uv, y_test_uv, "Linear Regression")
axs[0, 0].scatter(y_test_lr_uv, y_pred_lr_uv, alpha=0.5)
axs[0, 0].plot([y_test_lr_uv.min(), y_test_lr_uv.max()], [y_test_lr_uv.min(), y_test_lr_uv.max()], 'r--', lw=2)
axs[0, 0].set_title(f"Linear Regression: UV Index\nR2: {lr_r2_uv:.4f}, RMSE: {lr_rmse_uv:.4f}")
axs[0, 0].set_xlabel("Actual UV Index")
axs[0, 0].set_ylabel("Predicted UV Index")

# Polynomial Regression for UV
poly = PolynomialFeatures(degree=2)
X_train_poly_uv = poly.fit_transform(X_train_uv)
X_test_poly_uv = poly.transform(X_test_uv)
pr_uv, pr_r2_uv, pr_rmse_uv, y_test_pr_uv, y_pred_pr_uv = evaluate_regression(LinearRegression(), X_train_poly_uv, X_test_poly_uv, y_train_uv, y_test_uv, "Polynomial Regression")
axs[0, 1].scatter(y_test_pr_uv, y_pred_pr_uv, alpha=0.5)
axs[0, 1].plot([y_test_pr_uv.min(), y_test_pr_uv.max()], [y_test_pr_uv.min(), y_test_pr_uv.max()], 'r--', lw=2)
axs[0, 1].set_title(f"Polynomial Regression: UV Index\nR2: {pr_r2_uv:.4f}, RMSE: {pr_rmse_uv:.4f}")
axs[0, 1].set_xlabel("Actual UV Index")
axs[0, 1].set_ylabel("Predicted UV Index")

# Random Forest Regression for UV
rf_uv, rf_r2_uv, rf_rmse_uv, y_test_rf_uv, y_pred_rf_uv = evaluate_regression(RandomForestRegressor(n_estimators=100, random_state=42), X_train_uv, X_test_uv, y_train_uv, y_test_uv, "Random Forest")
axs[0, 2].scatter(y_test_rf_uv, y_pred_rf_uv, alpha=0.5)
axs[0, 2].plot([y_test_rf_uv.min(), y_test_rf_uv.max()], [y_test_rf_uv.min(), y_test_rf_uv.max()], 'r--', lw=2)
axs[0, 2].set_title(f"Random Forest: UV Index\nR2: {rf_r2_uv:.4f}, RMSE: {rf_rmse_uv:.4f}")
axs[0, 2].set_xlabel("Actual UV Index")
axs[0, 2].set_ylabel("Predicted UV Index")

# Feature importance for Random Forest model
importance = rf_uv.feature_importances_
indices = np.argsort(importance)[::-1]
axs[1, 0].bar(range(len(importance)), importance[indices])
axs[1, 0].set_title("Feature Importance for UV Index")
axs[1, 0].set_xlabel("Features")
axs[1, 0].set_ylabel("Importance")
axs[1, 0].set_xticks(range(len(importance)))
axs[1, 0].set_xticklabels([uv_features[i] for i in indices], rotation=45, ha='right')

# Residual analysis for Random Forest
residuals = y_test_rf_uv - y_pred_rf_uv
axs[1, 1].scatter(y_pred_rf_uv, residuals, alpha=0.5)
axs[1, 1].set_xlabel('Predicted UV Index')
axs[1, 1].set_ylabel('Residuals')
axs[1, 1].set_title('Residual Plot for Random Forest on UV Index')
axs[1, 1].axhline(y=0, color='r', linestyle='--')

# Remove unused subplot
fig.delaxes(axs[1, 2])

# Adjust layout and save the figure
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("uv_index_analysis_results.png", dpi=300, bbox_inches='tight')
plt.close()

print("\nUV Index regression analysis complete. Results have been saved in a single image: uv_index_analysis_results.png")