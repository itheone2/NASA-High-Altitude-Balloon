#Regression analysis and plot
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('flight_data_fused_robust.csv')

# Print available columns
print("Available columns in the dataset:")
for col in df.columns:
    print(f"- {col}")

# Function to prepare data for regression
def prepare_data(df, target, features):
    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate and plot regression results
def evaluate_regression(model, X_train, X_test, y_train, y_test, model_name, target):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{model_name} Results for {target}:")
    print(f"R-squared: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel(f"Actual {target}")
    plt.ylabel(f"Predicted {target}")
    plt.title(f"{model_name}: Actual vs Predicted {target}")
    plt.tight_layout()
    plt.savefig(f"{target.replace(' ', '_')}_{model_name.replace(' ', '_')}_prediction.png")
    plt.close()

    return model, r2, rmse

# Identify available relevant columns
available_columns = df.columns
uv_features = [col for col in ['Temperature (C)', 'Pressure (Pa)', 'Humidity (%)', 'Accel X (mg)', 'Accel Y (mg)', 'Accel Z (mg)'] if col in available_columns]
light_features = [col for col in ['Temperature (C)', 'Pressure (Pa)', 'Humidity (%)', 'Accel X (mg)', 'Accel Y (mg)', 'Accel Z (mg)'] if col in available_columns]

uv_target = 'UV Index' if 'UV Index' in available_columns else None
light_target = 'Light (lux)' if 'Light (lux)' in available_columns else None

if uv_target is None and light_target is None:
    raise ValueError("Neither 'UV Index' nor 'Light (lux)' columns are available in the dataset.")

# UV Intensity Prediction
if uv_target:
    X_train_uv, X_test_uv, y_train_uv, y_test_uv = prepare_data(df, uv_target, uv_features)

    # Linear Regression for UV
    lr_uv, lr_r2_uv, lr_rmse_uv = evaluate_regression(LinearRegression(), X_train_uv, X_test_uv, y_train_uv, y_test_uv, "Linear Regression", uv_target)

    # Polynomial Regression for UV
    poly = PolynomialFeatures(degree=2)
    X_train_poly_uv = poly.fit_transform(X_train_uv)
    X_test_poly_uv = poly.transform(X_test_uv)
    pr_uv, pr_r2_uv, pr_rmse_uv = evaluate_regression(LinearRegression(), X_train_poly_uv, X_test_poly_uv, y_train_uv, y_test_uv, "Polynomial Regression", uv_target)

    # Random Forest Regression for UV
    rf_uv, rf_r2_uv, rf_rmse_uv = evaluate_regression(RandomForestRegressor(n_estimators=100, random_state=42), X_train_uv, X_test_uv, y_train_uv, y_test_uv, "Random Forest", uv_target)

# Ambient Light Modeling
if light_target:
    X_train_light, X_test_light, y_train_light, y_test_light = prepare_data(df, light_target, light_features)

    # Linear Regression for Light
    lr_light, lr_r2_light, lr_rmse_light = evaluate_regression(LinearRegression(), X_train_light, X_test_light, y_train_light, y_test_light, "Linear Regression", light_target)

    # Polynomial Regression for Light
    X_train_poly_light = poly.fit_transform(X_train_light)
    X_test_poly_light = poly.transform(X_test_light)
    pr_light, pr_r2_light, pr_rmse_light = evaluate_regression(LinearRegression(), X_train_poly_light, X_test_poly_light, y_train_light, y_test_light, "Polynomial Regression", light_target)

    # Random Forest Regression for Light
    rf_light, rf_r2_light, rf_rmse_light = evaluate_regression(RandomForestRegressor(n_estimators=100, random_state=42), X_train_light, X_test_light, y_train_light, y_test_light, "Random Forest", light_target)

# Cross-validation
def cross_validate(model, X, y, model_name, target):
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"\nCross-validation results for {model_name} on {target}:")
    print(f"Mean R-squared: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

if uv_target:
    cross_validate(lr_uv, X_train_uv, y_train_uv, "Linear Regression", uv_target)
    cross_validate(rf_uv, X_train_uv, y_train_uv, "Random Forest", uv_target)

if light_target:
    cross_validate(lr_light, X_train_light, y_train_light, "Linear Regression", light_target)
    cross_validate(rf_light, X_train_light, y_train_light, "Random Forest", light_target)

# Feature importance for Random Forest models
def plot_feature_importance(model, features, target):
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importance for {target}")
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), [features[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{target.replace(' ', '_')}_feature_importance.png")
    plt.close()

if uv_target:
    plot_feature_importance(rf_uv, uv_features, uv_target)
if light_target:
    plot_feature_importance(rf_light, light_features, light_target)

# Residual analysis
def plot_residuals(model, X_test, y_test, model_name, target):
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot for {model_name} on {target}')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig(f"{target.replace(' ', '_')}_{model_name.replace(' ', '_')}_residuals.png")
    plt.close()

if uv_target:
    plot_residuals(lr_uv, X_test_uv, y_test_uv, "Linear Regression", uv_target)
    plot_residuals(rf_uv, X_test_uv, y_test_uv, "Random Forest", uv_target)

if light_target:
    plot_residuals(lr_light, X_test_light, y_test_light, "Linear Regression", light_target)
    plot_residuals(rf_light, X_test_light, y_test_light, "Random Forest", light_target)

print("\nRegression analysis complete. Results and plots have been saved.")