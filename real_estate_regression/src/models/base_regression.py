import numpy as np

class BaseRegressor:
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4, 
                 random_state=None, early_stopping=False, validation_fraction=0.1):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        
        self.weights_ = None
        self.bias_ = None
        self.cost_history_ = []
        self.val_cost_history_ = []
        
    def _initialize_weights(self, n_features):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        # Using a small random initialization
        self.weights_ = np.random.randn(n_features) * 0.01
        self.bias_ = 0.0
        
    def _compute_cost(self, X, y):
        """Mean Squared Error"""
        predictions = self.predict(X)
        errors = predictions - y
        return np.mean(errors ** 2) / 2
        
    def _compute_gradients(self, X, y, predictions):
        m = len(y)
        errors = predictions - y
        dw = (1 / m) * np.dot(X.T, errors)
        db = (1 / m) * np.sum(errors)
        return dw, db
        
    def predict(self, X):
        X = np.asarray(X)
        return np.dot(X, self.weights_) + self.bias_
        
    def fit(self, X, y):
        raise NotImplementedError("Subclasses must implement the fit method.")
