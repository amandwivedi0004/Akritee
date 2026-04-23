import unittest
import numpy as np
from src.models.simple_linear_regression import SimpleLinearRegression
from src.models.multiple_linear_regression import MultipleLinearRegression
from src.models.polynomial_regression import PolynomialRegression
from src.evaluation import r_squared, root_mean_squared_error

class TestRegression(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Simple dataset
        self.X_sim = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        self.y_sim = np.array([2, 4, 5, 4, 5])
        
        # Multiple dataset
        self.X_mult = np.random.rand(100, 3)
        self.y_mult = 3 * self.X_mult[:, 0] + 1.5 * self.X_mult[:, 1] - 2 * self.X_mult[:, 2] + np.random.randn(100) * 0.1

    def test_simple_linear_regression(self):
        model = SimpleLinearRegression(learning_rate=0.01, max_iter=2000, tol=1e-6)
        model.fit(self.X_sim, self.y_sim)
        preds = model.predict(self.X_sim)
        r2 = r_squared(self.y_sim, preds)
        self.assertGreater(r2, 0.4) # Should correlate
        
        preds_int, lower_ci, upper_ci, lower_pi, upper_pi = model.predict_with_intervals(self.X_sim)
        self.assertEqual(len(lower_ci), len(self.X_sim))
        
    def test_multiple_linear_regression_normal(self):
        model = MultipleLinearRegression(solver='normal')
        model.fit(self.X_mult, self.y_mult)
        preds = model.predict(self.X_mult)
        r2 = r_squared(self.y_mult, preds)
        self.assertGreater(r2, 0.9)
        
    def test_multiple_linear_regression_gd(self):
        model = MultipleLinearRegression(solver='gd', learning_rate=0.1, max_iter=3000)
        model.fit(self.X_mult, self.y_mult)
        preds = model.predict(self.X_mult)
        r2 = r_squared(self.y_mult, preds)
        self.assertGreater(r2, 0.8)
        
    def test_polynomial_regression(self):
        X_poly = np.linspace(-3, 3, 100).reshape(-1, 1)
        y_poly = X_poly[:, 0]**2 - 2*X_poly[:, 0] + 1 + np.random.randn(100)*0.1
        
        model = PolynomialRegression(degree=2, solver='normal')
        model.fit(X_poly, y_poly)
        preds = model.predict(X_poly)
        
        r2 = r_squared(y_poly, preds)
        self.assertGreater(r2, 0.9)
        
if __name__ == '__main__':
    unittest.main()
