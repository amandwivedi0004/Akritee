import numpy as np
import pandas as pd
from src.preprocessing import StandardScaler, SimpleImputer, train_test_split
from src.models.simple_linear_regression import SimpleLinearRegression
from src.models.polynomial_regression import PolynomialRegression
from src.models.ensemble import BlendingRegressor
from src.evaluation import r_squared, mean_absolute_error, root_mean_squared_error

def run_extended_examples():
    np.random.seed(42)
    
    print("=== 1. Preprocessing Example ===")
    raw_data = pd.DataFrame({
        'A': [1.0, 2.0, np.nan, 4.0, 5.0],
        'B': [10.0, np.nan, 30.0, 40.0, 50.0]
    })
    print("Original Data with NaNs:")
    print(raw_data)
    
    # Impute missing values with mean
    imputer = SimpleImputer(strategy='mean')
    imputed_data = imputer.fit_transform(raw_data)
    print("\nAfter Mean Imputation:")
    print(imputed_data)
    
    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(imputed_data)
    print("\nAfter StandardScaler (mean~0, std~1):")
    print(scaled_data.round(4))
    
    print("\n=== 2. Evaluation Metrics Example ===")
    y_true = np.array([2.5, 0.0, 2.0, 8.0])
    y_pred = np.array([3.0, -0.5, 2.0, 7.5])
    
    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    print(f"R-squared: {r_squared(y_true, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"RMSE: {root_mean_squared_error(y_true, y_pred):.4f}")
    
    print("\n=== 3. Ensemble (BlendingRegressor) Example ===")
    # Generate non-linear data
    X = np.linspace(-3, 3, 50)
    y = np.sin(X) + np.random.randn(50) * 0.1
    
    # We will blend a Polynomial Degree 1 (Linear) and Degree 3 (Cubic) model
    pr1 = PolynomialRegression(degree=1, solver='normal')
    pr3 = PolynomialRegression(degree=3, solver='normal')
    
    # Blend them: 20% weight to linear, 80% weight to cubic
    ensemble = BlendingRegressor(base_models=[pr1, pr3], weights=[0.2, 0.8])
    
    # Train test split natively
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit ensemble
    ensemble.fit(X_train, y_train)
    
    # Evaluate
    preds = ensemble.predict(X_test)
    print(f"Ensemble blended weights: linear=0.2, cubic=0.8")
    print(f"Ensemble R-squared on test set: {r_squared(y_test, preds):.4f}")
    print(f"Test X: {X_test[:3]}")
    print(f"True y: {y_test[:3]}")
    print(f"Pred y: {preds[:3]}")

if __name__ == '__main__':
    run_extended_examples()
