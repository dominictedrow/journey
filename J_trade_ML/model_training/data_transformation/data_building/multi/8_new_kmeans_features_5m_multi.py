import sqlite3
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import joblib
import os
from tqdm import tqdm
import multiprocessing

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
    return df

# Reshape the data into the required groups
def reshape_data(df):
    group_1min = df.iloc[:, 4:24]  # Columns 5 through 24 (indices 4 through 23)
    group_5min = df.iloc[:, 24:28]  # Columns 25 through 28 (indices 24 through 27)
    return group_1min, group_5min

# Analyze each row to identify patterns for group_1min
def identify_patterns_group_1min(group):
    num_rows, num_cols = group.shape
    patterns = np.zeros((num_rows, 12), dtype=int)
    
    for i in range(num_rows):
        row_patterns = []
        for j in range(0, num_cols - 4, 4):
            # Compare the last 3 columns of the current group of 4 to the next group's last 3 columns
            current_group = group.iloc[i, j+1:j+4].values
            next_group = group.iloc[i, j+5:j+8].values
            diff = next_group - current_group
            pattern = (diff > 0).astype(int)
            row_patterns.append(pattern)
        patterns[i] = np.hstack(row_patterns)

    return patterns

# Classify each row into one of the clusters with MiniBatchKMeans
def classify_patterns(patterns, model_path=None, is_new_model=False, batch_size=10000):
    if is_new_model:
        kmeans = MiniBatchKMeans(n_clusters=25000, batch_size=batch_size, random_state=42, n_init='auto')
        kmeans.fit(patterns)
        joblib.dump(kmeans, model_path)
        labels = kmeans.labels_
    else:
        if model_path and os.path.exists(model_path):
            kmeans = joblib.load(model_path)
            labels = kmeans.predict(patterns)
        else:
            raise FileNotFoundError(f"The model file at {model_path} does not exist.")
    
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

# Load table function for multiprocessing
def load_table(args):
    db_path, table_name = args
    return table_name, load_data_from_sqlite(db_path, table_name)

# Function to parallelize data loading
def parallel_load_data(db_path, table_names, n_jobs):
    with multiprocessing.Pool(n_jobs) as pool:
        results = list(tqdm(pool.imap(load_table, [(db_path, table_name) for table_name in table_names]), total=len(table_names), desc="Loading data from tables"))

    return dict(results)

# Process table function for multiprocessing
def process_table(args):
    table_name, df = args
    if df.empty:
        print(f"Table {table_name} is empty, skipping.")
        return table_name, np.array([])

    # Reshape data
    group_1min, group_5min = reshape_data(df)

    # Identify patterns
    patterns_1min = identify_patterns_group_1min(group_1min)
    table_patterns = np.hstack([patterns_1min]) 

    return table_name, table_patterns

# Main function to execute the steps
def main(db_path, new_db_path, model_path, is_new_model):
    table_names = get_table_names(db_path)
    
    # Parallelize data loading
    n_jobs = multiprocessing.cpu_count()  # Use all available CPUs
    df_dict = parallel_load_data(db_path, table_names, n_jobs)
    
    # Parallelize table processing
    with multiprocessing.Pool(n_jobs) as pool:
        processed_tables = list(tqdm(pool.imap(process_table, df_dict.items()), total=len(df_dict), desc="Processing tables"))
    
    # Filter out empty results
    processed_tables = [(name, patterns) for name, patterns in processed_tables if patterns.size > 0]

    if not processed_tables:
        print("No tables were processed successfully.")
        return

    combined_patterns = np.vstack([table_patterns for _, table_patterns in processed_tables])

    # Classify patterns
    print("Classifying patterns")
    classes = classify_patterns(combined_patterns, model_path=model_path, is_new_model=is_new_model)
    
    current_index = 0
    for table_name, df in tqdm(df_dict.items(), desc="Saving results to new DB"):
        num_rows = df.shape[0]
        table_classes = classes[current_index:current_index + num_rows]
        current_index += num_rows

        original_first_4_cols = df.iloc[:, :4]
        original_cols_to_keep = df.iloc[:, 4:-3]
        original_last_3_cols = df.iloc[:, -3:]

        # Reshape data
        group_1min, group_5min = reshape_data(df)

        # Identify patterns
        patterns_1min = identify_patterns_group_1min(group_1min)

        table_patterns = np.hstack([patterns_1min]) 

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

if __name__ == '__main__':
    # Define the paths to your SQLite databases and model path
    db_path = 'C://Users//crgon//OneDrive//Desktop//trading//data//multi_5m_OHLC_Val_2.db'
    new_db_path = 'C://Users//crgon//OneDrive//Desktop//trading//data//multi_5m_OHLC_Val_5_5_new.db'
    model_path = 'C://Users//crgon//OneDrive//Desktop//trading//data//data_scalers//kmeans_model_5m_multi_2.pkl'

    # Ask the user whether to create a new model or load an existing one
    is_new_model = input("Do you want to create a new model? (yes/no): ").strip().lower() == 'yes'

    # Run the main function
    main(db_path, new_db_path, model_path, is_new_model)

    print("Data processing and saving to new database complete.")
