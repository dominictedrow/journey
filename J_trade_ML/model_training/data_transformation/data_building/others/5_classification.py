import pandas as pd
import sqlite3
import numpy as np

def process_pricing_data(db_path, table_name, new_db_path, rounding_increment=0.25, multiple=2):
    # Connect to the original SQLite database
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Remove the index column (assumed to be the first column)
    df = df.iloc[:, 1:]

    # Identify feature and target columns
    non_price_columns = ['Minute', 'Day', 'Month', 'Hour']
    feature_columns = [col for col in df.columns[:-3] if col not in non_price_columns]
    target_columns = df.columns[-3:]

    # Function to round values to the nearest increment, handling NaN values
    def round_to_increment(x, increment):
        if pd.isna(x):
            return x
        return round(x / increment) * increment

    # Apply rounding to price-related feature columns
    df[feature_columns] = df[feature_columns].applymap(lambda x: round_to_increment(x, rounding_increment))
    
    # Augment dataset by creating two additional copies
    df_multiplied = df.copy()
    df_multiplied[feature_columns] = df_multiplied[feature_columns].applymap(lambda x: x * multiple if not pd.isna(x) else x)
    
    df_divided = df.copy()
    df_divided[feature_columns] = df_divided[feature_columns].applymap(lambda x: x / multiple if not pd.isna(x) else x)
    
    # Combine original, multiplied, and divided datasets
    df_combined = pd.concat([df, df_multiplied, df_divided], ignore_index=True)
    
    # Calculate the unique classes for price-related feature and target values
    combined_features = df_combined[feature_columns].values.flatten()
    combined_targets = df_combined[target_columns].values.flatten()
    
    min_val = min(combined_features[~np.isnan(combined_features)].min(), combined_targets[~np.isnan(combined_targets)].min())
    max_val = max(combined_features[~np.isnan(combined_features)].max(), combined_targets[~np.isnan(combined_targets)].max())
    
    # Create a class for every value in the range
    class_labels = np.arange(min_val, max_val + rounding_increment, rounding_increment)
    class_map = {value: idx for idx, value in enumerate(class_labels)}
    
    # Map feature and target columns to class labels
    df_combined[feature_columns] = df_combined[feature_columns].applymap(lambda x: class_map[round_to_increment(x, rounding_increment)] if not pd.isna(x) else x)
    df_combined[target_columns] = df_combined[target_columns].applymap(lambda x: class_map[round_to_increment(x, rounding_increment)] if not pd.isna(x) else x)
    
    # Handle non-price columns separately by assigning unique class for each value
    for col in non_price_columns:
        df_combined[col] = df_combined[col].astype(str).astype('category').cat.codes
    
    # Save the processed data to the new database
    conn = sqlite3.connect(new_db_path)
    processed_table_name = f"{table_name}"
    df_combined.to_sql(processed_table_name, conn, if_exists='replace', index=False)
    conn.close()
    
    # Print the number of classes
    num_classes = len(class_labels)
    print(f"Processed data saved to table '{processed_table_name}' in database '{new_db_path}'")
    print(f"Number of classes for price-related features and targets: {num_classes}")
    
    # Print unique target classes and their counts
    for target in target_columns:
        unique_classes = df_combined[target].unique()
        num_unique_classes = len(unique_classes)
        print(f"Unique classes for target '{target}': {sorted(unique_classes)}")
        print(f"Total number of unique classes for target '{target}': {num_unique_classes}")

# Example usage:
db_path = 'C://Users//crgon//OneDrive//Desktop//trading//data//XAUUSD_15m_OHLC_Val.db'
new_db_path = 'C://Users//crgon//OneDrive//Desktop//trading//data//XAUUSD_15m_OHLC_Val_classification.db'
table_name = 'XAUUSD'
process_pricing_data(db_path, table_name, new_db_path, rounding_increment=0.25, multiple=2)




"""
Explanation:

Identify non-price columns: The script separates the first four columns (Minute, Day, Month, Hour) that should not be included in the rounding and class calculation for targets.

Apply rounding to price-related feature columns: The price-related features are rounded to the nearest specified increment (default is 0.25).

Augment dataset: Two additional copies of the dataset are created: one where price-related feature values are multiplied by the specified multiple (default is 2) and another where they are divided by the same multiple.

Combine datasets: The original, multiplied, and divided datasets are combined.

Calculate unique classes: Unique classes for all price-related feature and target values are calculated within the min-max range, considering the specified rounding increment.

Map values to classes: Both price-related feature and target columns are mapped to their respective class labels.

Handle non-price columns: Non-price columns are handled separately by converting their values to categorical codes.

Save processed data to new database: The processed DataFrame is saved to a new SQLite database at the specified location.

Print the number of classes: The script prints the number of unique classes created for the price-related features and targets.
"""