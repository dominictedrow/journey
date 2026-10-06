import numpy as np
import pandas as pd
import sqlite3
import os
from sklearn.cluster import MiniBatchKMeans
import joblib
from tqdm import tqdm
import logging
from numba import jit, set_num_threads
import multiprocessing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the number of threads for Numba
set_num_threads(multiprocessing.cpu_count())

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

@jit(nopython=True)
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

@jit(nopython=True)
def compute_patterns(current, previous):
    patterns = np.zeros(4, dtype=np.int32)
    for i in range(4):
        diff = (current[i] - previous[i]) / max(abs(previous[i]), abs(current[i]), 1e-5)
        patterns[i] = calculate_pattern_value(diff)
    return patterns

@jit(nopython=True)
def compute_last_patterns(data):
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

def identify_patterns(df, group_col):
    all_timeframes = [1, 5, 15, 30, 60]
    max_pattern_count = 0
    pattern_data = {}

    # Determine available groups and max pattern features across the dataframe
    for group in all_timeframes:
        timeframes = [group] if group == 1 else [group, all_timeframes[all_timeframes.index(group) - 1]]
        
        for timeframe in timeframes:
            col_prefix = f'{timeframe}m'
            relevant_cols = [int(col.split('_')[-1]) for col in df.columns if col.startswith(f'{col_prefix}_') and col.split('_')[-1].isdigit()]
            
            if relevant_cols:
                max_group = max(relevant_cols)
                max_pattern_count = max(max_pattern_count, max_group * 4 + 3)  # 4 comparisons per group + 3 last patterns
    
    if max_pattern_count == 0:
        raise ValueError("No valid columns found for pattern identification. Please check your dataset.")

    logging.info(f"Maximum pattern feature count determined: {max_pattern_count}")

    # Now create pattern features for each row based on the group
    for idx, row in df.iterrows():
        group = row[group_col]
        row_patterns = []
        
        if group == 1:
            # Handle Group 1 by searching for the most recent past rows
            past_group1_rows = []
            current_idx = idx - 1
            while len(past_group1_rows) < 4 and current_idx >= 0:
                if df.loc[current_idx, group_col] == 1:
                    past_group1_rows.append(df.loc[current_idx])
                current_idx -= 1
            
            if len(past_group1_rows) == 4:
                current_data = row[['1m_Close_0', '1m_High_0', '1m_Low_0', '1m_Open_0']].values
                for past_row in past_group1_rows:
                    previous_data = past_row[['1m_Close_0', '1m_High_0', '1m_Low_0', '1m_Open_0']].values
                    patterns = compute_patterns(current_data, previous_data)
                    row_patterns.extend(patterns)
                last_patterns = compute_last_patterns(current_data)
                row_patterns.extend(last_patterns)
            else:
                # If there are not enough past Group 1 rows, fill with zeros
                row_patterns.extend([0] * max_pattern_count)
        else:
            timeframes = [group] if group == 1 else [group, all_timeframes[all_timeframes.index(group) - 1]]

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
        
        # Pad with zeros if necessary to ensure the row has the same number of pattern features
        if len(row_patterns) < max_pattern_count:
            row_patterns.extend([0] * (max_pattern_count - len(row_patterns)))

        pattern_data[idx] = row_patterns

    pattern_df = pd.DataFrame.from_dict(pattern_data, orient='index')
    pattern_df.columns = [f'Pattern_{i+1}' for i in range(max_pattern_count)]
    
    return pattern_df

def classify_patterns(patterns, model_path=None, is_new_model=False, batch_size=1000000):
    if is_new_model:
        kmeans = MiniBatchKMeans(n_clusters=360, batch_size=batch_size, random_state=42, n_init='auto')
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
    conn = sqlite3.connect(new_db_path)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

def collect_and_classify_patterns(db_path, table_names, model_path, is_new_model, num_targets, group_col):
    all_patterns = []
    all_target_columns = []
    pattern_dfs = {}
    original_indexes = {}

    for table_name in tqdm(table_names, desc="Processing tables"):
        df = load_data_from_sqlite(db_path, table_name)
        df = ensure_unique_column_names(df)
        
        # Store original indexes
        original_indexes[table_name] = df.index
        
        # Identify target columns
        target_columns = df.columns[-num_targets:].tolist()
        all_target_columns.extend(target_columns)
        
        # Remove target columns before pattern identification
        df_without_targets = df.drop(columns=target_columns)
        
        # Isolate pattern calculation for each table
        pattern_df = identify_patterns(df_without_targets, group_col)
        
        all_patterns.append(pattern_df)
        pattern_dfs[table_name] = pattern_df  # Store each pattern_df separately for merging later
    
    if not all_patterns:
        raise ValueError("No patterns were identified across all tables")
    
    all_patterns_df = pd.concat(all_patterns, ignore_index=True)

    if all_patterns_df.empty:
        raise ValueError("Combined patterns DataFrame is empty")

    # Classify all patterns together
    logging.info(f"Classifying patterns across all tables")
    classes = classify_patterns(all_patterns_df, model_path=model_path, is_new_model=is_new_model)
    
    # Split the classified data back into respective tables
    start_idx = 0
    for table_name in pattern_dfs.keys():
        pattern_df = pattern_dfs[table_name]
        end_idx = start_idx + len(pattern_df)
        pattern_classes = classes[start_idx:end_idx]
        pattern_df['Pattern_Class'] = pattern_classes
        pattern_dfs[table_name] = pattern_df  # Update with the pattern class
        start_idx = end_idx
    
    return pattern_dfs, original_indexes, list(set(all_target_columns))

def main(db_path, new_db_path, model_path, is_new_model, num_targets):
    group_col = 'Group'  # Column that identifies the group
    table_names = get_table_names(db_path)
    
    try:
        pattern_dfs, original_indexes, target_columns = collect_and_classify_patterns(db_path, table_names, model_path, is_new_model, num_targets, group_col)
    except ValueError as e:
        logging.error(f"Error in pattern collection and classification: {str(e)}")
        return
    
    for table_name in tqdm(table_names, desc="Saving processed data"):
        df = load_data_from_sqlite(db_path, table_name)
        pattern_df = pattern_dfs[table_name]
        
        # Reindex pattern_df to original index before merging back
        pattern_df.index = original_indexes[table_name]
        
        # Create the final dataframe
        non_target_columns = [col for col in df.columns if col not in target_columns]
        final_df = pd.concat([
            df[non_target_columns],
            pattern_df,
            df[target_columns]
        ], axis=1)
        
        final_df = ensure_unique_column_names(final_df)
        save_to_new_db(final_df, new_db_path, table_name)

    logging.info("All tables processed successfully.")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    db_path = "C:/Users/crgon/OneDrive/Desktop/trading/data/backup/multi_stacked_HLC.db"
    new_db_path = 'C://Users//crgon//OneDrive//Desktop//trading//data//multi_stacked_Train_large.db'
    model_path = 'C://Users//crgon//OneDrive//Desktop//trading//data//data_scalers//kmeans_model_multi_stacked_large.pkl'

    is_new_model = input("Do you want to create a new model? (yes/no): ").strip().lower() == 'yes'
    num_targets = int(input("Enter the number of target columns: "))

    main(db_path, new_db_path, model_path, is_new_model, num_targets)

    print("Data processing and saving to new database complete.")
