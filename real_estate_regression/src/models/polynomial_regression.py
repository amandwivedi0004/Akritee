import numpy as np
from .multiple_linear_regression import MultipleLinearRegression

class PolynomialRegression:
    def __init__(self, degree=2, solver='normal', penalty=None, alpha=1.0, 
                 learning_rate=0.01, max_iter=1000):
        self.degree = degree
        self.model = MultipleLinearRegression(solver=solver, penalty=penalty, alpha=alpha, 
                                            learning_rate=learning_rate, max_iter=max_iter)
        
    def _create_polynomial_features(self, X):
        X = np.asarray(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        m, n = X.shape
        # This creates purely polynomial features up to degree (X1, X1^2, X1^3...)
        # We also need interaction terms but standard poly regression per requirement 
        # "Create polynomial features up to degree 5" usually implies raising
        # each numerical feature to a power. If interactions are strictly needed,
        # we can implement combinations. Let's do simple raised powers for ease.
        poly_cols = []
        for d in range(1, self.degree + 1):
            poly_cols.append(X ** d)
            
        return np.concatenate(poly_cols, axis=1)

    def fit(self, X, y):
        X_poly = self._create_polynomial_features(X)
        self.model.fit(X_poly, y)
        return self
        
    def predict(self, X):
        X_poly = self._create_polynomial_features(X)
        return self.model.predict(X_poly)
        
    @property
    def cost_history_(self):
        return self.model.cost_history_
        
    @property
    def weights_(self):
        return self.model.weights_

def generate_spline_features(X, knots):
    """
    Piecewise linear splines
    X: 1D array of feature
    knots: list of knot points
    """
    X = np.asarray(X).flatten()
    features = [X]
    for knot in knots:
        features.append(np.maximum(0, X - knot))
    return np.column_stack(features)
