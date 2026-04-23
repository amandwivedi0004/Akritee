import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_generation import generate_real_estate_data
from src.preprocessing import StandardScaler, SimpleImputer, remove_outliers_iqr
from src.models.simple_linear_regression import SimpleLinearRegression
from src.models.multiple_linear_regression import MultipleLinearRegression, forward_selection, calculate_vif
from src.models.polynomial_regression import PolynomialRegression
from src.models.ensemble import BlendingRegressor
from src.evaluation import r_squared, bootstrap_metric_ci, permutation_test

def main():
    print("Generating data...")
    df, clean_df = generate_real_estate_data(n_samples=5000, seed=42)
    
    print("\n--- Part 1: Simple Linear Regression ---")
    df_imputed = SimpleImputer(strategy='mean').fit_transform(df)
    df_clean = remove_outliers_iqr(df_imputed, 'price')
    print(f"Data shape after imputation and IQR outlier removal: {df_clean.shape}")
    
    X_simple = df_clean['area'].values
    y_simple = df_clean['price'].values
    
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_simple_scaled = scaler_x.fit_transform(X_simple)
    y_simple_scaled = scaler_y.fit_transform(y_simple)
    
    slr = SimpleLinearRegression(learning_rate=0.05, max_iter=2000, early_stopping=True)
    slr.fit(X_simple_scaled, y_simple_scaled)
    
    preds_scaled, lower_ci, upper_ci, _, _ = slr.predict_with_intervals(X_simple_scaled, confidence=0.95)
    
    print(f"Simple LR R-squared: {r_squared(y_simple_scaled, preds_scaled):.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X_simple_scaled, y_simple_scaled, alpha=0.1, label='Data')
    plt.plot(X_simple_scaled, preds_scaled, color='red', label='Regression Line')
    
    sort_idx = np.argsort(X_simple_scaled)
    plt.fill_between(X_simple_scaled[sort_idx], lower_ci[sort_idx], upper_ci[sort_idx], color='red', alpha=0.2, label='95% CI')
    plt.legend()
    plt.title("Simple Linear Regression: Area vs Price (Scaled)")
    plt.savefig('slr_result.png')
    plt.close()

    print("\n--- Part 2: Multiple Linear Regression ---")
    features = ['area', 'bedrooms', 'bathrooms', 'age', 'distance_city', 'crime_rate', 'school_rating', 'garage', 'basement']
    X_mult = df_clean[features]
    y_mult = df_clean['price'].values
    
    X_mult_imputed = SimpleImputer(strategy='median').fit_transform(X_mult)
    y_mult_scaled = scaler_y.fit_transform(y_mult)
    
    scaler_mult = StandardScaler()
    X_mult_scaled = scaler_mult.fit_transform(X_mult_imputed)
    
    mlr = MultipleLinearRegression(solver='normal', penalty='l2', alpha=5.0)
    mlr.fit(X_mult_scaled, y_mult_scaled)
    mlr_preds = mlr.predict(X_mult_scaled)
    print(f"Multiple LR (Ridge, Normal Eq) R-squared: {r_squared(y_mult_scaled, mlr_preds):.4f}")
    
    vif_data = calculate_vif(pd.DataFrame(X_mult_scaled, columns=features))
    print("VIF Analysis (Multicollinearity):")
    print(vif_data)

    print("\n--- Part 3: Polynomial Regression ---")
    top_features = forward_selection(pd.DataFrame(X_mult_scaled, columns=features), y_mult_scaled, significance_level=0.01)
    print(f"Selected top features via Stepwise: {top_features}")
    
    X_top = pd.DataFrame(X_mult_scaled, columns=features)[top_features].values
    
    for degree in range(1, 4):
        pr = PolynomialRegression(degree=degree, solver='normal', penalty='l2', alpha=1.0)
        pr.fit(X_top, y_mult_scaled)
        preds = pr.predict(X_top)
        r2 = r_squared(y_mult_scaled, preds)
        print(f"Polynomial Reg Degree {degree} R-squared: {r2:.4f}")
            
    print("\n--- Part 4: Ensemble Analysis ---")
    pr1 = PolynomialRegression(degree=1, solver='normal', penalty='l2', alpha=1.0)
    pr2 = PolynomialRegression(degree=2, solver='normal', penalty='l2', alpha=1.0)
    ensemble = BlendingRegressor(base_models=[pr1, pr2], weights=[0.4, 0.6])
    ensemble.fit(X_top, y_mult_scaled)
    ensemble_preds = ensemble.predict(X_top)
    
    print(f"Blending Regressor (Deg 1 & 2) R-squared: {r_squared(y_mult_scaled, ensemble_preds):.4f}")
    
    lower, upper = bootstrap_metric_ci(y_mult_scaled, ensemble_preds, r_squared, n_bootstraps=50)
    print(f"Blending Regressor R-squared 95% CI: [{lower:.4f}, {upper:.4f}]")
    
    print("\nPipeline complete. Visualizations saved as '*.png'.")

if __name__ == "__main__":
    main()
