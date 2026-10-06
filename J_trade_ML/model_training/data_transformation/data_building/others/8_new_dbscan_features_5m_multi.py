import sqlite3
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import joblib
import os
from scipy.spatial.distance import cdist

# Function to load all table names from SQLite database
def get_table_names(db_path):
    conn = sqlite3.connect(db_path)
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql_query(query, conn)
    conn.close()
    return tables['name'].tolist()

# Function to load data from SQLite database
def load_data_from_sqlite(db_path, table_name):
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    df.drop(df.columns[0], axis=1, inplace=True)  # Drop the first column if it's an index
    return df

# Reshape the data into the required groups
def reshape_data(df):
    group_1min = df.iloc[:, 4:24]  # Columns 5 through 24 (indices 4 through 23)
    group_5min = df.iloc[:, 24:28]  # Columns 25 through 28 (indices 24 through 27)
    return group_1min, group_5min

# Analyze each row to identify patterns
def identify_patterns(group, group_size):
    patterns = []
    num_groups = group.shape[1] // group_size

    for i in range(group.shape[0]):
        row_patterns = []
        for col in range(group_size):
            group_values = group.iloc[i, col::group_size].values
            diff = np.diff(group_values)
            pattern = (diff > 0).astype(int)
            row_patterns.append(pattern)
        row_patterns = np.hstack(row_patterns)
        patterns.append(row_patterns)

    return np.array(patterns)

# Classify each row into one of the clusters
def classify_patterns(patterns, model_path=None, is_new_model=False):
    if is_new_model:
        dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust parameters as needed
        dbscan.fit(patterns)
        joblib.dump(dbscan, model_path)
        labels = dbscan.labels_
        # Save core samples
        core_samples = patterns[dbscan.core_sample_indices_]
        joblib.dump(core_samples, model_path.replace('.pkl', '_cores.pkl'))
    else:
        if model_path and os.path.exists(model_path) and os.path.exists(model_path.replace('.pkl', '_cores.pkl')):
            dbscan = joblib.load(model_path)
            core_samples = joblib.load(model_path.replace('.pkl', '_cores.pkl'))
            # Assign clusters based on nearest core sample
            distances = cdist(patterns, core_samples, 'euclidean')
            labels = np.argmin(distances, axis=1)
            labels = dbscan.labels_[dbscan.core_sample_indices_][labels]
        else:
            raise FileNotFoundError(f"The model file at {model_path} or its corresponding cores file does not exist.")
    
    # Print unique labels to debug
    print(f"Unique labels from clustering: {np.unique(labels)}")
    return labels

# Save results to a new SQLite database
def save_to_new_db(df, new_db_path, table_name):
    conn = sqlite3.connect(new_db_path)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

# Ensure unique column names
def ensure_unique_column_names(df):
    columns = df.columns
    unique_columns = []
    seen = set()
    for col in columns:
        new_col = col
        count = 1
        while new_col in seen:
            new_col = f"{col}_{count}"
            count += 1
        unique_columns.append(new_col)
        seen.add(new_col)
    df.columns = unique_columns
    return df

# Main function to execute the steps
def main(db_path, new_db_path, model_path, is_new_model):
    table_names = get_table_names(db_path)
    combined_patterns = []
    df_dict = {}
    
    for table_name in table_names:
        df = load_data_from_sqlite(db_path, table_name)
        df_dict[table_name] = df

        original_first_4_cols = df.iloc[:, :4]
        original_cols_to_keep = df.iloc[:, 4:-3]
        original_last_3_cols = df.iloc[:, -3:]

        # Reshape data
        group_1min, group_5min = reshape_data(df)

        # Identify patterns
        patterns_1min = identify_patterns(group_1min, 4)
        patterns_5min = identify_patterns(group_5min, 4)

        table_patterns = np.hstack([patterns_1min, patterns_5min])
        combined_patterns.append(table_patterns)
    
    combined_patterns = np.vstack(combined_patterns)

    # Classify patterns
    classes = classify_patterns(combined_patterns, model_path=model_path, is_new_model=is_new_model)
    
    current_index = 0
    for table_name, df in df_dict.items():
        num_rows = df.shape[0]
        table_classes = classes[current_index:current_index + num_rows]
        current_index += num_rows

        original_first_4_cols = df.iloc[:, :4]
        original_cols_to_keep = df.iloc[:, 4:-3]
        original_last_3_cols = df.iloc[:, -3:]

        # Reshape data
        group_1min, group_5min = reshape_data(df)

        # Identify patterns
        patterns_1min = identify_patterns(group_1min, 4)
        patterns_5min = identify_patterns(group_5min, 4)

        table_patterns = np.hstack([patterns_1min, patterns_5min])

        # Create a DataFrame for patterns
        patterns_df = pd.DataFrame(table_patterns, columns=[f'Pattern_{i+1}' for i in range(table_patterns.shape[1])])

        # Combine all columns into the final DataFrame, removing the original last 3 columns
        final_df = pd.concat([original_first_4_cols, original_cols_to_keep, patterns_df], axis=1)
        final_df['Pattern_Class'] = table_classes
        final_df = pd.concat([final_df, original_last_3_cols], axis=1)

        # Ensure unique column names
        final_df = ensure_unique_column_names(final_df)

        # Save to new database
        save_to_new_db(final_df, new_db_path, table_name)

# Define the paths to your SQLite databases and model path
db_path = 'C://Users//crgon//OneDrive//Desktop//trading//data//XAUUSD_15m_OHLC_Val_1.db'
new_db_path = 'C://Users//crgon//OneDrive//Desktop//trading//data//XAUUSD_15m_OHLC_Val_4_4_new.db'
model_path = 'C://Users//crgon//OneDrive//Desktop//trading//data//data_scalers//dbscan_model_5m.pkl'

# Ask the user whether to create a new model or load an existing one
is_new_model = input("Do you want to create a new model? (yes/no): ").strip().lower() == 'yes'

# Run the main function
main(db_path, new_db_path, model_path, is_new_model)

print("Data processing and saving to new database complete.")
