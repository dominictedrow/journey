import warnings
from contextlib import contextmanager
import numpy as np
import pandas as pd
import sqlite3
import os
from sklearn.cluster import MiniBatchKMeans
import joblib
from tqdm import tqdm
import logging
import multiprocessing

@contextmanager
def suppress_specific_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*attempt to set the values inplace instead of always setting a new array.*")
        yield

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_table_names(db_path):
    conn = sqlite3.connect(db_path)
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql_query(query, conn)
    conn.close()
    return tables['name'].tolist()

def load_data_from_sqlite(db_path, table_name):
    conn = sqlite3.connect(db_path, timeout=30)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def calculate_pattern_value(diff):
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

def compute_patterns(current, previous):
    patterns = np.zeros(4, dtype=np.int32)
    for i in range(4):
        diff = (current[i] - previous[i]) / max(abs(previous[i]), abs(current[i]), 1e-5)
        patterns[i] = calculate_pattern_value(diff)
    return patterns

def compute_last_patterns(data):
    patterns = np.zeros(3, dtype=np.int32)
    open_price, high_price, low_price, close_price = data[3], data[1], data[2], data[0]
    
    for i, (price1, price2) in enumerate([(close_price, open_price), (high_price, open_price), (low_price, open_price)]):
        diff = (price1 - price2) / max(abs(price1), abs(price2), 1e-5)
        patterns[i] = calculate_pattern_value(diff)
    
    return patterns

def identify_patterns(df, group_col):
    all_timeframes = [1, 5, 15, 30, 60]
    max_pattern_count = 19

    pattern_data = []

    for idx, row in df.iterrows():
        group = row[group_col]
        row_patterns = []
        
        if group == 1:
            # Find the 5 most recent rows with group 1, including the current row
            group_1_rows = df[df[group_col] == 1].iloc[max(0, idx-4):idx+1]
            
            if len(group_1_rows) >= 5:
                current_data = row[['1m_Close_0', '1m_High_0', '1m_Low_0', '1m_Open_0']].values
                for i in range(1, 5):  # Start from 1 to exclude the current row
                    previous_data = group_1_rows.iloc[-i-1][['1m_Close_0', '1m_High_0', '1m_Low_0', '1m_Open_0']].values
                    patterns = compute_patterns(current_data, previous_data)
                    row_patterns.extend(patterns)
                last_patterns = compute_last_patterns(current_data)
                row_patterns.extend(last_patterns)
            else:
                row_patterns.extend([0] * max_pattern_count)
        else:
            timeframes = [group, all_timeframes[all_timeframes.index(group) - 1]]

            for timeframe in timeframes:
                col_prefix = f'{timeframe}m'
                current_cols = [col for col in df.columns if col.startswith(col_prefix) and pd.notnull(row[col])]
                relevant_cols = [int(col.split('_')[-1]) for col in current_cols if col.split('_')[-1].isdigit()]

                if not relevant_cols:
                    continue

                num_groups = max(relevant_cols)

                for i in range(num_groups + 1):
                    cols = [f'{col_prefix}_Close_{i}', f'{col_prefix}_High_{i}', f'{col_prefix}_Low_{i}', f'{col_prefix}_Open_{i}']
                    if all(col in df.columns for col in cols):
                        current_data = row[cols].values

                        if i < num_groups:
                            next_cols = [f'{col_prefix}_Close_{i+1}', f'{col_prefix}_High_{i+1}', f'{col_prefix}_Low_{i+1}', f'{col_prefix}_Open_{i+1}']
                            next_data = row[next_cols].values
                            patterns = compute_patterns(current_data, next_data)
                            row_patterns.extend(patterns)
                        else:
                            last_patterns = compute_last_patterns(current_data)
                            row_patterns.extend(last_patterns)
        
        # Ensure the row has exactly 19 pattern features
        row_patterns = row_patterns[:max_pattern_count]
        if len(row_patterns) < max_pattern_count:
            row_patterns.extend([0] * (max_pattern_count - len(row_patterns)))

        pattern_data.append(row_patterns)

    pattern_df = pd.DataFrame(pattern_data, columns=[f'Pattern_{i+1}' for i in range(max_pattern_count)])
    
    return pattern_df

def classify_patterns(patterns, model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file at {model_path} does not exist.")
    
    kmeans = joblib.load(model_path)
    labels = kmeans.predict(patterns)
    return labels

def save_to_db(df, db_path, table_name):
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.commit()  # Ensure changes are committed
    conn.close()
    logging.info(f"Saved {len(df)} rows to table {table_name}")

def process_table(db_path, table_name, model_path, group_col):
    logging.info(f"Starting to process table {table_name}")
    df = load_data_from_sqlite(db_path, table_name)
    logging.info(f"Loaded {len(df)} rows from table {table_name}")

    # Identify rows that need processing
    pattern_cols = [f'Pattern_{i}' for i in range(1, 20)] + ['Pattern_Class']
    rows_to_process = df[pattern_cols].isnull().any(axis=1)
    
    num_rows_to_process = rows_to_process.sum()
    logging.info(f"Found {num_rows_to_process} rows to process in table {table_name}")
    
    if num_rows_to_process == 0:
        logging.info(f"No rows to process in table {table_name}")
        return
    
    df_to_process = df[rows_to_process]
    
    # Process only the rows that need it
    logging.info(f"Identifying patterns for {num_rows_to_process} rows")
    pattern_df = identify_patterns(df_to_process, group_col)
    
    # Classify patterns
    logging.info("Classifying patterns")
    pattern_classes = classify_patterns(pattern_df, model_path)
    
    # Update the original dataframe with new patterns and classes
    logging.info("Updating dataframe with new patterns and classes")
    with suppress_specific_warning():
        for col in pattern_cols[:-1]:
            df.loc[rows_to_process, col] = pattern_df[col].values
        df.loc[rows_to_process, 'Pattern_Class'] = pattern_classes
    
    # Save the updated dataframe back to the database
    logging.info(f"Saving updated data back to table {table_name}")
    save_to_db(df, db_path, table_name)
    
    logging.info(f"Processed and updated {num_rows_to_process} rows in table {table_name}")

    # Verify the update
    updated_df = load_data_from_sqlite(db_path, table_name)
    updated_rows = updated_df[pattern_cols].notnull().all(axis=1).sum()
    logging.info(f"After update: {updated_rows} rows have all pattern columns filled in table {table_name}")

def main(db_path, model_path):
    group_col = 'Group'
    table_names = get_table_names(db_path)
    
    for table_name in tqdm(table_names, desc="Processing tables"):
        try:
            process_table(db_path, table_name, model_path, group_col)
        except Exception as e:
            logging.error(f"Error processing table {table_name}: {str(e)}")
    
    # Verify the database after processing
    logging.info("Verifying database after processing")
    for table_name in table_names:
        df = load_data_from_sqlite(db_path, table_name)
        pattern_cols = [f'Pattern_{i}' for i in range(1, 20)] + ['Pattern_Class']
        filled_rows = df[pattern_cols].notnull().all(axis=1).sum()
        total_rows = len(df)
        logging.info(f"Table {table_name}: {filled_rows}/{total_rows} rows have all pattern columns filled")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    db_path = "C:/Users/crgon/OneDrive/Desktop/trading/metatrader/dwxconnect-main/python/data/databases/Processed_multi_stacked_1.db"
    model_path = "C:/Users/crgon/OneDrive/Desktop/trading/metatrader/dwxconnect-main/python/data/data_scalers/kmeans_model_multi_stacked.pkl"

    main(db_path, model_path)

    print("Data processing and updating complete.")
