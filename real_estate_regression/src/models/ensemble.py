import numpy as np
from .multiple_linear_regression import MultipleLinearRegression

class StackingRegressor:
    def __init__(self, base_models, meta_model=None):
        self.base_models = base_models
        # Default meta-model is simple linear regression over predictions
        self.meta_model = meta_model if meta_model else MultipleLinearRegression(solver='normal', penalty='l2', alpha=1.0)
        
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).flatten()
        
        # We need validation splits to train the meta model natively without target leakage
        m = len(y)
        indices = np.random.permutation(m)
        train_size = int(m * 0.7)
        
        train_idx, val_idx = indices[:train_size], indices[train_size:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        meta_features_val = []
        
        for model in self.base_models:
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            meta_features_val.append(preds)
            
        # Refit base models on all data for future predictions
        for model in self.base_models:
            model.fit(X, y)
            
        meta_features_val = np.column_stack(meta_features_val)
        self.meta_model.fit(meta_features_val, y_val)
        return self
        
    def predict(self, X):
        X = np.asarray(X)
        meta_features = []
        
        for model in self.base_models:
            meta_features.append(model.predict(X))
            
        meta_features = np.column_stack(meta_features)
        return self.meta_model.predict(meta_features)

class BlendingRegressor:
    def __init__(self, base_models, weights=None):
        self.base_models = base_models
        if weights is None:
            self.weights = np.ones(len(base_models)) / len(base_models)
        else:
            self.weights = np.asarray(weights) / np.sum(weights)
            
    def fit(self, X, y):
        for model in self.base_models:
            model.fit(X, y)
        return self
        
    def predict(self, X):
        X = np.asarray(X)
        preds = np.zeros(len(X))
        for weight, model in zip(self.weights, self.base_models):
            preds += weight * model.predict(X)
        return preds
