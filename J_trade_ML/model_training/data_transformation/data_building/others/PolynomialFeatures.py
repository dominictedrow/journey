import sqlite3
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# Connect to the SQLite database
db_path = 'C://Users//crgon//OneDrive//Desktop//trading//data//test//XAUUSD_15m_OHLC_Val_2.db'
conn = sqlite3.connect(db_path)

# Load data from database
query = "SELECT * FROM XAUUSD"  # Assuming table name is stock_data
data = pd.read_sql_query(query, conn)

# Drop the index and target columns to avoid data leakage
data = data.drop(columns=['index', 'Target1', 'Target2', 'Target3'])

# Fill any potential NaNs in the data
data.fillna(method='ffill', inplace=True)

# Create rolling and expanding features for more robust time-series features
def add_features(df):
    # Identify groups for rolling and expanding calculations
    minute_cols = [f'1m_{attr}_{i}' for i in range(15) for attr in ['Open', 'High', 'Low', 'Close']]
    five_min_cols = [f'5m_{attr}_{i}' for i in range(3) for attr in ['Open', 'High', 'Low', 'Close']]

    new_cols = []  # List to collect new column dataframes

    # Rolling windows for 1-minute data
    window_size = 5  # Example window size
    for col in minute_cols:
        roll_mean = df[col].rolling(window=window_size, min_periods=1).mean().rename(f'rol_mean_{col}')
        roll_std = df[col].rolling(window=window_size, min_periods=1).std().fillna(0).rename(f'rol_std_{col}')
        new_cols.append(roll_mean)
        new_cols.append(roll_std)

    # Expanding windows for 5-minute data
    for col in five_min_cols:
        exp_mean = df[col].expanding(min_periods=1).mean().rename(f'exp_mean_{col}')
        exp_std = df[col].expanding(min_periods=1).std().fillna(0).rename(f'exp_std_{col}')
        new_cols.append(exp_mean)
        new_cols.append(exp_std)

    # Concatenate all new columns to the original dataframe
    df = pd.concat([df] + new_cols, axis=1)
    return df

# Apply feature engineering
data = add_features(data)

# Polynomial and Interaction Features
def create_polynomial_features(data, degree=2, interaction_only=True):
    feature_columns = [col for col in data.columns if 'Minute' not in col and 'Day' not in col and 'Month' not in col and 'Hour' not in col]
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    poly_features = poly.fit_transform(data[feature_columns])
    feature_names = poly.get_feature_names_out(feature_columns)
    return pd.DataFrame(poly_features, columns=feature_names, index=data.index)

polynomial_data = create_polynomial_features(data)

# Saving the enhanced dataset back to a new SQLite table
polynomial_data.to_sql('enhanced_stock_data', conn, if_exists='replace', index=False)

# Close the database connection
conn.close()