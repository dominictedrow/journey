import pandas as pd
import sqlite3
import numpy as np

def preprocess_data_for_ml(db_path, table_name, rounding_increment=0.25):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    
    # Remove the index column (assumed to be the first column)
    df = df.iloc[:, 1:]

    # Identify non-price columns and separate them
    non_price_columns = ['Minute', 'Day', 'Month', 'Hour']
    feature_columns = [col for col in df.columns if col not in non_price_columns and col not in df.columns[-3:]]
    target_columns = df.columns[-1:]

    # Function to round values to the nearest increment, handling NaN values
    def round_to_increment(x, increment):
        if pd.isna(x):
            return x
        return round(x / increment) * increment

    # Check for non-vectorized data (assuming any non-integer values in price-related columns)
    def is_non_vectorized(x):
        return not pd.isna(x) and not x.is_integer()

    # Apply rounding to non-vectorized data
    df[feature_columns] = df[feature_columns].applymap(lambda x: round_to_increment(x, rounding_increment) if is_non_vectorized(x) else x)
    df[target_columns] = df[target_columns].applymap(lambda x: round_to_increment(x, rounding_increment) if is_non_vectorized(x) else x)
    
    # Calculate the unique classes for price-related feature and target values
    combined_features = df[feature_columns].values.flatten()
    combined_targets = df[target_columns].values.flatten()
    
    min_val = min(combined_features[~np.isnan(combined_features)].min(), combined_targets[~np.isnan(combined_targets)].min())
    max_val = max(combined_features[~np.isnan(combined_features)].max(), combined_targets[~np.isnan(combined_targets)].max())
    
    # Create a class for every value in the range
    class_labels = np.arange(min_val, max_val + rounding_increment, rounding_increment)
    class_map = {value: idx for idx, value in enumerate(class_labels)}
    
    # Map feature and target columns to class labels
    df[feature_columns] = df[feature_columns].applymap(lambda x: class_map[round_to_increment(x, rounding_increment)] if not pd.isna(x) else x)
    df[target_columns] = df[target_columns].applymap(lambda x: class_map[round_to_increment(x, rounding_increment)] if not pd.isna(x) else x)
    
    # Handle non-price columns separately by assigning unique class for each value
    for col in non_price_columns:
        df[col] = df[col].astype(str).astype('category').cat.codes
    
    # Overwrite the original table with the processed data
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    
    # Print the number of classes
    num_classes = len(class_labels)
    print(f"Processed data saved to table '{table_name}' in database '{db_path}'")
    print(f"Number of classes for price-related features and targets: {num_classes}")
    
    # Print unique target classes and their counts
    for target in target_columns:
        unique_classes = df[target].unique()
        num_unique_classes = len(unique_classes)
        print(f"Unique classes for target '{target}': {sorted(unique_classes)}")
        print(f"Total number of unique classes for target '{target}': {num_unique_classes}")

# Example usage:
db_path = 'C://Users//crgon//OneDrive//Desktop//trading//data//XAUUSD_15m_OHLC_Val_classification.db'
table_name = 'XAUUSD'
preprocess_data_for_ml(db_path, table_name, rounding_increment=0.25)


"""
Explanation:
Remove the index column: The first column, assumed to be the index, is removed by slicing df.iloc[:, 1:].

Identify non-price columns: The function separates the first four columns (Minute, Day, Month, Hour) and identifies the feature and target columns.

Round and handle NaN values: The round_to_increment function now checks if the value is NaN and leaves it unchanged if it is. The rounding operation is only applied to non-integer, non-NaN values.

Calculate unique classes: The unique classes for price-related features and targets are calculated and mapped.

Handle non-price columns: Non-price columns are handled separately by converting their values to categorical codes.

Save processed data: The processed DataFrame is saved back to the SQLite database, overwriting the original table.

Print the number of classes: The function prints the number of unique classes created for the price-related features and targets.
"""