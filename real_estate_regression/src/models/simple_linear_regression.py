import numpy as np
from .base_regression import BaseRegressor

class SimpleLinearRegression(BaseRegressor):
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4, 
                 random_state=None, early_stopping=False, validation_fraction=0.1,
                 lr_decay=0.0):
        super().__init__(learning_rate, max_iter, tol, random_state, early_stopping, validation_fraction)
        self.lr_decay = lr_decay
        self.std_error_ = None
        self.n_ = None
        self.mse_ = None
        self.X_mean_ = None
        self.ss_x_ = None
        
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).flatten()
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        m, n = X.shape
        if n > 1:
            raise ValueError("SimpleLinearRegression only accepts 1D features.")
            
        self._initialize_weights(n)
        
        X_train, y_train = X, y
        if self.early_stopping:
            val_size = int(m * self.validation_fraction)
            if val_size > 0:
                if self.random_state is not None:
                    np.random.seed(self.random_state)
                indices = np.random.permutation(m)
                train_idx, val_idx = indices[val_size:], indices[:val_size]
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]
            else:
                self.early_stopping = False
                
        best_val_cost = float('inf')
        patience = 10
        patience_counter = 0
        
        for i in range(self.max_iter):
            current_lr = self.learning_rate / (1 + self.lr_decay * i)
            
            predictions = np.dot(X_train, self.weights_) + self.bias_
            dw, db = self._compute_gradients(X_train, y_train, predictions)
            
            self.weights_ -= current_lr * dw
            self.bias_ -= current_lr * db
            
            cost = self._compute_cost(X_train, y_train)
            self.cost_history_.append(cost)
            
            if self.early_stopping:
                val_cost = self._compute_cost(X_val, y_val)
                self.val_cost_history_.append(val_cost)
                
                if val_cost < best_val_cost - self.tol:
                    best_val_cost = val_cost
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    break
            else:
                if i > 0 and abs(self.cost_history_[-2] - cost) < self.tol:
                    break
                    
        self._calculate_statistics(X, y)
        return self
        
    def predict(self, X):
        X = np.asarray(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return super().predict(X).flatten()
        
    def _calculate_statistics(self, X, y):
        predictions = self.predict(X)
        residuals = y - predictions
        self.n_ = len(y)
        
        self.mse_ = np.sum(residuals**2) / max(1, (self.n_ - 2))
        self.X_mean_ = np.mean(X)
        self.ss_x_ = np.sum((X - self.X_mean_)**2)
        
        if self.ss_x_ > 0:
            self.std_error_ = np.sqrt(self.mse_ / self.ss_x_)
        else:
            self.std_error_ = float('inf')

    def predict_with_intervals(self, X, confidence=0.95):
        X = np.asarray(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        predictions = self.predict(X)
        
        alpha = 1.0 - confidence
        try:
            from scipy import stats
            t_value = stats.t.ppf(1.0 - alpha/2.0, self.n_ - 2)
        except ImportError:
            t_value = 1.96
            
        term = (X - self.X_mean_)**2 / self.ss_x_
        se_mean = np.sqrt(self.mse_ * (1/self.n_ + term))
        se_pred = np.sqrt(self.mse_ * (1 + 1/self.n_ + term))
        
        se_mean = se_mean.flatten()
        se_pred = se_pred.flatten()
        predictions = predictions.flatten()
        
        lower_ci = predictions - t_value * se_mean
        upper_ci = predictions + t_value * se_mean
        
        lower_pi = predictions - t_value * se_pred
        upper_pi = predictions + t_value * se_pred
        
        return predictions, lower_ci, upper_ci, lower_pi, upper_pi
