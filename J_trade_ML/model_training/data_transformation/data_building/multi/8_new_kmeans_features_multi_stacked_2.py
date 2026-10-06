import numpy as np
import pandas as pd
import sqlite3
import os
from sklearn.cluster import MiniBatchKMeans
import joblib
from tqdm import tqdm
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
from numba import njit, prange  # Import njit and prange for Numba parallelization

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Number of CPU cores
num_cores = psutil.cpu_count(logical=False)

def get_table_names(db_path):
    """
    Retrieves the list of table names from the SQLite database.

    Parameters:
        db_path (str): Path to the SQLite database.

    Returns:
        list: A list of table names.
    """
    with sqlite3.connect(db_path) as conn:
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = pd.read_sql_query(query, conn)
    return tables['name'].tolist()

def load_data_from_sqlite(db_path, table_name):
    """
    Loads data from a specified table in the SQLite database.

    Parameters:
        db_path (str): Path to the SQLite database.
        table_name (str): Name of the table to load.

    Returns:
        pd.DataFrame: DataFrame containing the table data.
    """
    with sqlite3.connect(db_path, timeout=30) as conn:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)
    return df

def ensure_unique_column_names(df):
    """
    Ensures that the DataFrame has unique column names by appending suffixes to duplicates.

    Parameters:
        df (pd.DataFrame): DataFrame to process.

    Returns:
        pd.DataFrame: DataFrame with unique column names.
    """
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

@njit
def calculate_pattern_value(diff):
    """
    Calculates the pattern value based on the difference.

    Parameters:
        diff (float): The calculated difference.

    Returns:
        int32: The pattern value.
    """
    if diff > 0:
        if diff < 0.33:
            return 1
        elif diff < 0.66:
            return 2
        else:
            return 3
    else:
        if diff > -0.33:
            return 4
        elif diff > -0.66:
            return 5
        else:
            return 6

@njit
def compute_patterns_numba(current, previous):
    """
    Computes the patterns between current and previous data arrays.

    Parameters:
        current (np.array): Current data array.
        previous (np.array): Previous data array.

    Returns:
        np.array: Array of pattern values.
    """
    patterns = np.zeros(4, dtype=np.int32)
    for i in range(4):
        diff = (current[i] - previous[i]) / max(abs(previous[i]), abs(current[i]), 1e-5)
        patterns[i] = calculate_pattern_value(diff)
    return patterns

@njit
def compute_last_patterns_numba(data):
    """
    Computes the last set of patterns for the given data array.

    Parameters:
        data (np.array): Data array.

    Returns:
        np.array: Array of pattern values.
    """
    patterns = np.zeros(3, dtype=np.int32)
    open_price = data[3]
    close_price = data[0]
    high_price = data[1]
    low_price = data[2]

    # Open to Close
    diff = (close_price - open_price) / max(abs(open_price), abs(close_price), 1e-5)
    patterns[0] = calculate_pattern_value(diff)

    # Open to High
    diff = (high_price - open_price) / max(abs(open_price), abs(high_price), 1e-5)
    patterns[1] = calculate_pattern_value(diff)

    # Open to Low
    diff = (low_price - open_price) / max(abs(open_price), abs(low_price), 1e-5)
    patterns[2] = calculate_pattern_value(diff)

    return patterns

def identify_patterns(df, group_col_name):
    """
    Identifies patterns in the DataFrame and returns a DataFrame of pattern features.

    Parameters:
        df (pd.DataFrame): DataFrame to process.
        group_col_name (str): Name of the group column.

    Returns:
        pd.DataFrame: DataFrame containing pattern features.
    """
    all_timeframes = [1, 5, 15, 30, 60]
    max_pattern_count = 0

    # Collect columns and calculate max_pattern_count
    col_indices = {}
    for timeframe in all_timeframes:
        tf_prefix = f'{timeframe}m'
        tf_cols = {'Close': [], 'High': [], 'Low': [], 'Open': []}
        for col in df.columns:
            if col.startswith(tf_prefix):
                parts = col.split('_')
                if len(parts) == 3 and parts[2].isdigit():
                    price_type = parts[1]
                    group_num = int(parts[2])
                    if price_type in tf_cols:
                        tf_cols[price_type].append(col)
        col_indices[tf_prefix] = tf_cols

    # Determine max_pattern_count
    for tf_prefix, tf_cols in col_indices.items():
        num_groups = len(tf_cols['Close'])
        max_pattern_count = max(max_pattern_count, num_groups * 4 + 3)

    logging.info(f"Maximum pattern feature count determined: {max_pattern_count}")

    # Initialize pattern data array
    pattern_data = np.zeros((len(df), max_pattern_count), dtype=np.int32)

    # Process each row
    for idx in range(len(df)):
        row_patterns = []
        group = df.iloc[idx][group_col_name]
        if group == 1:
            # Modified Group 1 handling
            past_group1_rows = []
            current_idx = idx - 1
            while len(past_group1_rows) < 4 and current_idx >= 0:
                if df.iloc[current_idx][group_col_name] == 1:
                    past_group1_rows.append(df.iloc[current_idx])
                current_idx -= 1
            
            if len(past_group1_rows) == 4:
                current_data = df.iloc[idx][['1m_Close_0', '1m_High_0', '1m_Low_0', '1m_Open_0']].values.astype(np.float64)
                for past_row in past_group1_rows:
                    previous_data = past_row[['1m_Close_0', '1m_High_0', '1m_Low_0', '1m_Open_0']].values.astype(np.float64)
                    patterns = compute_patterns_numba(current_data, previous_data)
                    row_patterns.extend(patterns)
                last_patterns = compute_last_patterns_numba(current_data)
                row_patterns.extend(last_patterns)
            else:
                # Not enough past Group 1 data
                row_patterns.extend([0] * max_pattern_count)
        else:
            # Handle other groups
            timeframes = []
            if group == 5:
                timeframes = [5, 1]
            elif group == 15:
                timeframes = [15, 5]
            elif group == 30:
                timeframes = [30, 15]
            elif group == 60:
                timeframes = [60, 30]
            else:
                timeframes = [group]

            for timeframe in timeframes:
                tf_prefix = f'{timeframe}m'
                tf_cols = col_indices.get(tf_prefix, {})
                num_groups = len(tf_cols.get('Close', [])) - 1
                for i in range(num_groups + 1):
                    cols = [f'{tf_prefix}_{ptype}_{i}' for ptype in ['Close', 'High', 'Low', 'Open']]
                    if all(col in df.columns for col in cols):
                        current_data = df.iloc[idx][cols].values.astype(np.float64)
                        if i < num_groups:
                            next_cols = [f'{tf_prefix}_{ptype}_{i+1}' for ptype in ['Close', 'High', 'Low', 'Open']]
                            next_data = df.iloc[idx][next_cols].values.astype(np.float64)
                            patterns = compute_patterns_numba(current_data, next_data)
                            row_patterns.extend(patterns)
                        else:
                            last_patterns = compute_last_patterns_numba(current_data)
                            row_patterns.extend(last_patterns)
        # Pad with zeros if necessary
        if len(row_patterns) < max_pattern_count:
            row_patterns.extend([0] * (max_pattern_count - len(row_patterns)))
        pattern_data[idx, :max_pattern_count] = row_patterns[:max_pattern_count]

    pattern_df = pd.DataFrame(pattern_data, columns=[f'Pattern_{i+1}' for i in range(max_pattern_count)])
    return pattern_df

def classify_patterns(patterns, model_path=None, is_new_model=False, batch_size=500000):
    """
    Classifies the patterns using MiniBatchKMeans.

    Parameters:
        patterns (pd.DataFrame): DataFrame containing pattern features.
        model_path (str): Path to save or load the model.
        is_new_model (bool): Whether to create a new model.
        batch_size (int): Batch size for MiniBatchKMeans.

    Returns:
        np.array: Array of cluster labels.
    """
    if is_new_model:
        kmeans = MiniBatchKMeans(n_clusters=360, batch_size=batch_size, random_state=42)
        kmeans.fit(patterns)
        joblib.dump(kmeans, model_path)
        labels = kmeans.labels_
    else:
        if model_path and os.path.exists(model_path):
            kmeans = joblib.load(model_path)
            labels = kmeans.predict(patterns)
        else:
            raise FileNotFoundError(f"The model file at {model_path} does not exist.")
    return labels

def save_to_new_db(df, new_db_path, table_name):
    """
    Saves the DataFrame to a new SQLite database.

    Parameters:
        df (pd.DataFrame): DataFrame to save.
        new_db_path (str): Path to the new SQLite database.
        table_name (str): Name of the table to save.
    """
    with sqlite3.connect(new_db_path) as conn:
        df.to_sql(table_name, conn, if_exists='replace', index=False)

def process_table(args):
    """
    Processes a single table: loads data, identifies patterns, and returns pattern DataFrame.

    Parameters:
        args (tuple): Tuple containing (db_path, table_name, num_targets, group_col_name).

    Returns:
        tuple: (table_name, pattern_df, original_index, target_columns)
    """
    db_path, table_name, num_targets, group_col_name = args
    try:
        df = load_data_from_sqlite(db_path, table_name)
        df = ensure_unique_column_names(df)
        original_index = df.index
        target_columns = df.columns[-num_targets:].tolist()
        df_without_targets = df.drop(columns=target_columns)
        pattern_df = identify_patterns(df_without_targets, group_col_name)
        return table_name, pattern_df, original_index, target_columns
    except Exception as e:
        logging.error(f"Error processing table {table_name}: {str(e)}")
        raise

def collect_and_classify_patterns(db_path, table_names, model_path, is_new_model, num_targets, group_col_name):
    """
    Collects patterns from all tables and classifies them collectively to ensure consistent classes.

    Parameters:
        db_path (str): Path to the SQLite database.
        table_names (list): List of table names to process.
        model_path (str): Path to save or load the model.
        is_new_model (bool): Whether to create a new model.
        num_targets (int): Number of target columns.
        group_col_name (str): Name of the group column.

    Returns:
        tuple: (pattern_dfs, original_indexes, target_columns)
    """
    all_target_columns = set()
    pattern_dfs = {}
    original_indexes = {}
    patterns_list = []

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = {
            executor.submit(process_table, (db_path, table_name, num_targets, group_col_name)): table_name
            for table_name in table_names
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tables"):
            table_name = futures[future]
            try:
                table_name, pattern_df, original_index, target_columns = future.result()
                pattern_dfs[table_name] = pattern_df
                original_indexes[table_name] = original_index
                all_target_columns.update(target_columns)
                patterns_list.append(pattern_df)
            except Exception as e:
                logging.error(f"Error processing table {table_name}: {str(e)}")

    if not patterns_list:
        raise ValueError("No patterns were identified across all tables")

    # Combine patterns for classification
    all_patterns_df = pd.concat(patterns_list, ignore_index=True)
    if all_patterns_df.empty:
        raise ValueError("Combined patterns DataFrame is empty")

    logging.info("Classifying patterns across all tables")
    classes = classify_patterns(all_patterns_df, model_path=model_path, is_new_model=is_new_model)

    # Assign classes back to pattern dataframes
    start_idx = 0
    for table_name in pattern_dfs:
        pattern_df = pattern_dfs[table_name]
        end_idx = start_idx + len(pattern_df)
        pattern_classes = classes[start_idx:end_idx]
        pattern_df['Pattern_Class'] = pattern_classes
        pattern_dfs[table_name] = pattern_df
        start_idx = end_idx

    return pattern_dfs, original_indexes, list(all_target_columns)

def main():
    """
    Main function to orchestrate the processing of data.
    """
    multiprocessing.freeze_support()
    db_path = "C:/Users/crgon/OneDrive/Desktop/trading/data/backup/multi_stacked_HLC.db"
    new_db_path = 'C://Users//crgon//OneDrive//Desktop//trading//data//Processed_multi_stacked_HLC_Train_2000.db'
    model_path = 'C://Users//crgon//OneDrive//Desktop//trading//data//data_scalers//kmeans_model_multi_stacked_3.pkl'

    # Validate user input for is_new_model
    is_new_model_input = input("Do you want to create a new model? (yes/no): ").strip().lower()
    while is_new_model_input not in ['yes', 'no']:
        is_new_model_input = input("Invalid input. Please enter 'yes' or 'no': ").strip().lower()
    is_new_model = is_new_model_input == 'yes'

    # Validate user input for num_targets
    while True:
        try:
            num_targets = int(input("Enter the number of target columns: "))
            if num_targets < 0:
                raise ValueError("Number of target columns must be non-negative.")
            break
        except ValueError as ve:
            print(f"Invalid input: {ve}. Please enter a valid integer.")

    group_col_name = 'Group'
    table_names = get_table_names(db_path)
    if not table_names:
        logging.error("No tables found in the database.")
        return

    try:
        pattern_dfs, original_indexes, target_columns = collect_and_classify_patterns(
            db_path, table_names, model_path, is_new_model, num_targets, group_col_name
        )
    except ValueError as e:
        logging.error(f"Error in pattern collection and classification: {str(e)}")
        return

    for table_name in tqdm(table_names, desc="Saving processed data"):
        try:
            df = load_data_from_sqlite(db_path, table_name)
            pattern_df = pattern_dfs.get(table_name)
            if pattern_df is None:
                logging.warning(f"No pattern data found for table {table_name}. Skipping.")
                continue
            pattern_df.index = original_indexes[table_name]
            non_target_columns = [col for col in df.columns if col not in target_columns]
            final_df = pd.concat([
                df[non_target_columns],
                pattern_df,
                df[target_columns]
            ], axis=1)
            final_df = ensure_unique_column_names(final_df)
            save_to_new_db(final_df, new_db_path, table_name)
        except Exception as e:
            logging.error(f"Error saving data for table {table_name}: {str(e)}")

    logging.info("All tables processed successfully.")
    print("Data processing and saving to new database complete.")

if __name__ == '__main__':
    main()