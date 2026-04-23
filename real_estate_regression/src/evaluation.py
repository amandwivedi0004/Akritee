import numpy as np

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / max(ss_tot, 1e-10))

def adjusted_r_squared(y_true, y_pred, n_features):
    n = len(y_true)
    r2 = r_squared(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / max(1, (n - n_features - 1))

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_absolute_percentage_error(y_true, y_pred):
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def aic(y_true, y_pred, n_features):
    n = len(y_true)
    mse = np.mean((y_true - y_pred)**2)
    return n * np.log(max(mse, 1e-10)) + 2 * n_features

def bic(y_true, y_pred, n_features):
    n = len(y_true)
    mse = np.mean((y_true - y_pred)**2)
    return n * np.log(max(mse, 1e-10)) + n_features * np.log(n)

def bootstrap_metric_ci(y_true, y_pred, metric_func, n_bootstraps=1000, confidence_level=0.95):
    """
    Calculates bootstrap confidence intervals for a given metric.
    """
    n = len(y_true)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    bootstrapped_scores = []
    
    for _ in range(n_bootstraps):
        # Sample with replacement
        indices = np.random.randint(0, n, n)
        if len(np.unique(y_true[indices])) < 2:
            continue
            
        score = metric_func(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
        
    alpha = 1.0 - confidence_level
    lower_bound = np.percentile(bootstrapped_scores, alpha / 2.0 * 100)
    upper_bound = np.percentile(bootstrapped_scores, (1.0 - alpha / 2.0) * 100)
    
    return lower_bound, upper_bound

def permutation_test(y_true, y_pred1, y_pred2, metric_func, n_permutations=1000):
    """
    Checks if model 1 is significantly better than model 2.
    Tests the null hypothesis that there is no difference in their performance.
    """
    score1 = metric_func(y_true, y_pred1)
    score2 = metric_func(y_true, y_pred2)
    observed_diff = abs(score1 - score2)
    
    count = 0
    n = len(y_true)
    
    for _ in range(n_permutations):
        # Randomly swap predictions
        mask = np.random.random(n) > 0.5
        perm_pred1 = np.where(mask, y_pred1, y_pred2)
        perm_pred2 = np.where(mask, y_pred2, y_pred1)
        
        perm_diff = abs(metric_func(y_true, perm_pred1) - metric_func(y_true, perm_pred2))
        
        if perm_diff >= observed_diff:
            count += 1
            
    p_value = count / n_permutations
    return p_value
