#Regression analysis
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ... [Keep all the existing code up to the print statement at the end] ...

print("\nRegression analysis complete. Results and plots have been saved.")

# New function to create the regression analysis summary plot
def plot_regression_summary(model_comparison, feature_importance, residuals):
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle('Regression Analysis Summary', fontsize=16)

    # Model Comparison
    models = list(model_comparison.keys())
    x = np.arange(len(models))
    width = 0.35

    axs[0, 0].bar(x - width/2, [model_comparison[m]['UV']['R2'] for m in models], width, label='UV R²', color='b', alpha=0.7)
    axs[0, 0].bar(x + width/2, [model_comparison[m]['Light']['R2'] for m in models], width, label='Light R²', color='g', alpha=0.7)
    axs[0, 0].set_ylabel('R² Score')
    axs[0, 0].set_title('Model Comparison (R²)')
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(models)
    axs[0, 0].legend()

    axs[0, 1].bar(x - width/2, [model_comparison[m]['UV']['RMSE'] for m in models], width, label='UV RMSE', color='b', alpha=0.7)
    axs[0, 1].bar(x + width/2, [model_comparison[m]['Light']['RMSE'] for m in models], width, label='Light RMSE', color='g', alpha=0.7)
    axs[0, 1].set_ylabel('RMSE')
    axs[0, 1].set_title('Model Comparison (RMSE)')
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(models)
    axs[0, 1].legend()

    # Feature Importance
    features = list(feature_importance['UV'].keys())
    axs[1, 0].barh(features, [feature_importance['UV'][f] for f in features], label='UV', color='b', alpha=0.7)
    axs[1, 0].barh(features, [feature_importance['Light'][f] for f in features], left=[feature_importance['UV'][f] for f in features], label='Light', color='g', alpha=0.7)
    axs[1, 0].set_xlabel('Importance')
    axs[1, 0].set_title('Feature Importance')
    axs[1, 0].legend()

    # Residual Analysis
    axs[1, 1].scatter(residuals['UV']['predicted'], residuals['UV']['residuals'], label='UV', color='b', alpha=0.7)
    axs[1, 1].scatter(residuals['Light']['predicted'], residuals['Light']['residuals'], label='Light', color='g', alpha=0.7)
    axs[1, 1].axhline(y=0, color='r', linestyle='--')
    axs[1, 1].set_xlabel('Predicted Values')
    axs[1, 1].set_ylabel('Residuals')
    axs[1, 1].set_title('Residual Analysis')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig('regression_analysis_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

# Prepare data for the summary plot
model_comparison = {
    'Linear Regression': {
        'UV': {'R2': lr_r2_uv, 'RMSE': lr_rmse_uv},
        'Light': {'R2': lr_r2_light, 'RMSE': lr_rmse_light}
    },
    'Polynomial Regression': {
        'UV': {'R2': pr_r2_uv, 'RMSE': pr_rmse_uv},
        'Light': {'R2': pr_r2_light, 'RMSE': pr_rmse_light}
    },
    'Random Forest': {
        'UV': {'R2': rf_r2_uv, 'RMSE': rf_rmse_uv},
        'Light': {'R2': rf_r2_light, 'RMSE': rf_rmse_light}
    }
}

feature_importance = {
    'UV': dict(zip(uv_features, rf_uv.feature_importances_)),
    'Light': dict(zip(light_features, rf_light.feature_importances_))
}

residuals = {
    'UV': {
        'predicted': rf_uv.predict(X_test_uv),
        'residuals': y_test_uv - rf_uv.predict(X_test_uv)
    },
    'Light': {
        'predicted': rf_light.predict(X_test_light),
        'residuals': y_test_light - rf_light.predict(X_test_light)
    }
}

# Generate the summary plot
plot_regression_summary(model_comparison, feature_importance, residuals)

print("Regression analysis summary plot has been saved as 'regression_analysis_summary.png'")