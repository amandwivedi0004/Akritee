import numpy as np
import pandas as pd

def generate_real_estate_data(n_samples=10000, seed=42):
    """
    Generates synthetic real estate data based on the provided formula and distribution.
    Adds 5% missing values (MCAR) and 2% outliers in price data.
    """
    np.random.seed(seed)
    
    # Generate correlated features
    area = np.random.normal(2000, 500, n_samples)
    bedrooms = np.random.poisson(3, n_samples) + 1
    bathrooms = bedrooms * 0.8 + np.random.normal(0, 0.5, n_samples)
    age = np.random.exponential(15, n_samples)
    distance_city = np.random.gamma(2, 3, n_samples)
    crime_rate = np.random.exponential(5, n_samples)
    school_rating = np.random.beta(2, 1, n_samples) * 9 + 1
    garage = np.random.binomial(3, 0.6, n_samples)
    basement = area * 0.3 + np.random.normal(0, 200, n_samples)
    
    # Complex price relationship with non-linearities
    price = (150 * area +  
             10000 * bedrooms +  
             8000 * bathrooms -  
             300 * age -  
             2000 * distance_city -  
             1000 * crime_rate +  
             5000 * school_rating +  
             3000 * garage +  
             50 * basement + 
             0.01 * area**2 -  # Non-linear term 
             100 * age * distance_city +  # Interaction 
             np.random.normal(0, 20000, n_samples))  # Noise
             
    # Create DataFrame
    df = pd.DataFrame({
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age': age,
        'distance_city': distance_city,
        'crime_rate': crime_rate,
        'school_rating': school_rating,
        'garage': garage,
        'basement': basement,
        'price': price
    })
    
    # Keep a copy of clean data for validation if needed
    clean_df = df.copy()
    
    # Add 5% missing values randomly distributed across ALL features (excluding price)
    # The requirement says "5% missing values randomly distributed", let's assume it means 
    # overall 5% of the entries in features.
    n_features = len(df.columns) - 1 # price is target
    n_missing = int((n_samples * n_features) * 0.05)
    
    # Randomly choose locations for missing values
    rows = np.random.randint(0, n_samples, n_missing)
    cols = np.random.randint(0, n_features, n_missing)
    
    # Convert feature columns list
    feature_cols = df.columns[:-1]
    
    for r, c in zip(rows, cols):
        col_name = feature_cols[c]
        df.loc[r, col_name] = np.nan
        
    # Add 2% outliers in price data
    # 2% outliers implies 200 samples
    n_outliers = int(n_samples * 0.02)
    outlier_indices = np.random.choice(df.index, n_outliers, replace=False)
    
    # Make outliers extremely high or extremely low
    # Using 5 to 10 standard deviations from the mean
    price_mean = df['price'].mean()
    price_std = df['price'].std()
    
    for idx in outlier_indices:
        if np.random.random() > 0.5:
            # high outlier
            df.loc[idx, 'price'] = df.loc[idx, 'price'] + np.random.uniform(5, 10) * price_std
        else:
            # low outlier (make sure it doesn't go below say 10k or negative)
            new_price = df.loc[idx, 'price'] - np.random.uniform(5, 10) * price_std
            # Optional: ensure price > 0, cap at random small positive
            df.loc[idx, 'price'] = max(new_price, np.random.uniform(10000, 50000))
            
    return df, clean_df

if __name__ == "__main__":
    print("Generating data...")
    df, clean_df = generate_real_estate_data()
    print(f"Data shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Summary statistics:\n{df.describe()}")
