import numpy as np
import pandas as pd

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        
    def fit(self, X):
        X = np.asarray(X)
        self.mean_ = np.nanmean(X, axis=0)
        self.scale_ = np.nanstd(X, axis=0)
        if np.isscalar(self.scale_):
            self.scale_ = 1.0 if self.scale_ == 0.0 else self.scale_
        else:
            self.scale_[self.scale_ == 0.0] = 1.0
        return self
        
    def transform(self, X):
        X = np.asarray(X)
        return (X - self.mean_) / self.scale_
        
    def fit_transform(self, X):
        return self.fit(X).transform(X)

class MinMaxScaler:
    def __init__(self):
        self.min_ = None
        self.scale_ = None
        
    def fit(self, X):
        X = np.asarray(X)
        self.min_ = np.nanmin(X, axis=0)
        data_max = np.nanmax(X, axis=0)
        self.scale_ = data_max - self.min_
        if np.isscalar(self.scale_):
            self.scale_ = 1.0 if self.scale_ == 0.0 else self.scale_
        else:
            self.scale_[self.scale_ == 0.0] = 1.0
        return self
        
    def transform(self, X):
        X = np.asarray(X)
        return (X - self.min_) / self.scale_
        
    def fit_transform(self, X):
        return self.fit(X).transform(X)

class SimpleImputer:
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.statistics_ = {}
        
    def fit(self, df):
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
            
        for col in df.columns:
            if self.strategy == 'mean':
                self.statistics_[col] = df[col].mean()
            elif self.strategy == 'median':
                self.statistics_[col] = df[col].median()
            else:
                raise ValueError("Strategy must be 'mean' or 'median'")
        return self
        
    def transform(self, df):
        is_numpy = False
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
            is_numpy = True
            
        df_out = df.copy()
        for col in self.statistics_:
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna(self.statistics_[col])
                
        if is_numpy:
            return df_out.values
        return df_out
        
    def fit_transform(self, df):
        return self.fit(df).transform(df)

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Return a mask or filtered dataframe. Let's return the filtered dataframe.
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()

def train_test_split(*arrays, test_size=0.2, random_state=42):
    n_samples = len(arrays[0])
    np.random.seed(random_state)
    indices = np.random.permutation(n_samples)
    
    test_samples = int(n_samples * test_size)
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]
    
    result = []
    for array in arrays:
        if isinstance(array, pd.DataFrame) or isinstance(array, pd.Series):
            result.append(array.iloc[train_indices])
            result.append(array.iloc[test_indices])
        else:
            result.append(array[train_indices])
            result.append(array[test_indices])
            
    return result

def k_fold_split(n_samples, k=5, shuffle=True, random_state=42):
    indices = np.arange(n_samples)
    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(indices)
    
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1
    
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_indices = indices[start:stop]
        train_indices = np.concatenate((indices[:start], indices[stop:]))
        folds.append((train_indices, val_indices))
        current = stop
    return folds
