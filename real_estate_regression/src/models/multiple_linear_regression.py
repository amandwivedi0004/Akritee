import numpy as np
from .base_regression import BaseRegressor

class MultipleLinearRegression(BaseRegressor):
    def __init__(self, solver='gd', learning_rate=0.01, max_iter=1000, tol=1e-4, 
                 random_state=None, early_stopping=False, validation_fraction=0.1,
                 penalty=None, alpha=1.0):
        """
        solver: 'gd' (gradient descent) or 'normal' (normal equation)
        penalty: None, 'l1' (Lasso), or 'l2' (Ridge)
        alpha: regularization strength
        """
        super().__init__(learning_rate, max_iter, tol, random_state, early_stopping, validation_fraction)
        self.solver = solver
        self.penalty = penalty
        self.alpha = alpha
        
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).flatten()
        
        m, n = X.shape
        
        if self.solver == 'normal':
            # Add bias term (column of 1s)
            X_b = np.c_[np.ones((m, 1)), X]
            
            if self.penalty == 'l2':
                # Normal Equation with Ridge
                A = np.eye(n + 1)
                A[0, 0] = 0 # Do not regularize the bias term
                theta = np.linalg.inv(X_b.T.dot(X_b) + self.alpha * A).dot(X_b.T).dot(y)
            elif self.penalty == 'l1':
                raise ValueError("Normal equation not supported for L1 (Lasso). Use solver='gd'")
            else:
                # Standard Normal Equation
                theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
                
            self.bias_ = theta[0]
            self.weights_ = theta[1:]
            return self
            
        elif self.solver == 'gd':
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
            patience = 20
            patience_counter = 0
            
            for i in range(self.max_iter):
                predictions = np.dot(X_train, self.weights_) + self.bias_
                errors = predictions - y_train
                
                # Gradients without regularization
                dw = (1 / len(y_train)) * np.dot(X_train.T, errors)
                db = (1 / len(y_train)) * np.sum(errors)
                
                # Regularization
                if self.penalty == 'l2':
                    dw += (self.alpha / len(y_train)) * self.weights_
                elif self.penalty == 'l1':
                    dw += (self.alpha / len(y_train)) * np.sign(self.weights_)
                    
                self.weights_ -= self.learning_rate * dw
                self.bias_ -= self.learning_rate * db
                
                # Calculate cost mapping regularization
                cost = self._compute_cost_reg(X_train, y_train)
                self.cost_history_.append(cost)
                
                if self.early_stopping:
                    val_cost = self._compute_cost_reg(X_val, y_val)
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
            return self
        else:
            raise ValueError("solver must be 'gd' or 'normal'")

    def _compute_cost_reg(self, X, y):
        m = len(y)
        predictions = self.predict(X)
        base_cost = np.mean((predictions - y)**2) / 2
        
        if self.penalty == 'l2':
            return base_cost + (self.alpha / (2 * m)) * np.sum(self.weights_**2)
        elif self.penalty == 'l1':
            return base_cost + (self.alpha / m) * np.sum(np.abs(self.weights_))
        return base_cost

# Feature Selection Function (Forward Stepwise)
def forward_selection(X, y, significance_level=0.05):
    from scipy import stats
    import pandas as pd
    
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
        
    initial_features = X.columns.tolist()
    best_features = []
    
    while (len(initial_features) > 0):
        remaining_features = list(set(initial_features) - set(best_features))
        new_pval = pd.Series(index=remaining_features, dtype=float)
        
        for new_column in remaining_features:
            current_X = X[best_features + [new_column]].values
            
            # fit normal equation
            m = len(current_X)
            X_b = np.c_[np.ones((m, 1)), current_X]
            try:
                theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
                preds = X_b.dot(theta)
                
                n_params = X_b.shape[1]
                mse = np.sum((preds - y)**2) / max(1, (m - n_params))
                var_b = mse * np.linalg.inv(X_b.T.dot(X_b))
                se_b = np.sqrt(np.diag(var_b))
                
                t_stats = theta / se_b
                p_values = [2 * (1 - stats.t.cdf(np.abs(t), df=m-n_params)) for t in t_stats]
                
                # new_column is the last one (index -1)
                new_pval[new_column] = p_values[-1]
            except Exception:
                new_pval[new_column] = 1.0 # singular matrix
            
        min_p_value = new_pval.min()
        if min_p_value < significance_level:
            best_features.append(new_pval.idxmin())
        else:
            break
            
    return best_features

def calculate_vif(X):
    """
    Calculate Variance Inflation Factor to check multicollinearity.
    """
    import pandas as pd
    from scipy.stats import linregress
    
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
        
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    
    vifs = []
    for i in range(X.shape[1]):
        y_target = X.iloc[:, i].values
        X_data = X.drop(X.columns[i], axis=1).values
        
        # We can use our MultipleLinearRegression (Normal Equation) to find R^2
        model = MultipleLinearRegression(solver='normal')
        model.fit(X_data, y_target)
        preds = model.predict(X_data)
        
        ss_res = np.sum((y_target - preds)**2)
        ss_tot = np.sum((y_target - np.mean(y_target))**2)
        r_squared = 1 - (ss_res / max(ss_tot, 1e-10))
        
        vif = 1 / (1 - r_squared + 1e-10)
        vifs.append(vif)
        
    vif_data["VIF"] = vifs
    return vif_data
