from tqdm import tqdm
import pandas as pd
import sqlite3
import os
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor


def parse_timeframe(timeframe):
    unit = timeframe[-1]
    return int(timeframe[:-1]) * (60 if unit == 'm' else 1)

def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = data['Date'].astype(str)
    data['Timestamp'] = data['Timestamp'].astype(str)
    try:
        data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Timestamp'], format='%Y%m%d %H:%M:%S')
    except Exception as e:
        raise ValueError(f"Error converting date and time: {e}")
    return data.drop(columns=['Date', 'Timestamp'])

def integrate_timeframes(base_data, smaller_timeframe_data, base_timeframe, smaller_timeframe, last_values=None):
    base_interval_seconds = parse_timeframe(base_timeframe)
    smaller_interval_seconds = parse_timeframe(smaller_timeframe)
    groups_count = int(base_interval_seconds / smaller_interval_seconds)

    columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    column_names = [f"{smaller_timeframe}_{col}_{i}" for i in range(groups_count) for col in columns]
    results = []

    for index, base_row in base_data.iterrows():
        start_time = base_row['DateTime']
        row_data = []

        last_known_segment_values = [None] * groups_count

        for i in range(groups_count):
            segment_start = start_time + timedelta(seconds=i * smaller_interval_seconds)
            segment_end = segment_start + timedelta(seconds=smaller_interval_seconds)
            segment_data = smaller_timeframe_data[(smaller_timeframe_data['DateTime'] >= segment_start) & (smaller_timeframe_data['DateTime'] < segment_end)]

            if not segment_data.empty:
                segment_values = segment_data.iloc[-1][columns].tolist()
                last_known_segment_values[i] = segment_values
            if last_known_segment_values[i] is None:
                for j in range(i - 1, -1, -1):
                    if last_known_segment_values[j] is not None:
                        last_known_segment_values[i] = last_known_segment_values[j]
                        break
            row_data.extend(last_known_segment_values[i] if last_known_segment_values[i] is not None else [None] * len(columns))

        results.append([start_time] + row_data)

    df = pd.DataFrame(results, columns=['DateTime'] + column_names)
    return df

def process_chunk(start_index, end_index, target_data, chosen_timeframes, base_path, file_name, target_timeframe):
    chunk_data = target_data.iloc[start_index:end_index].copy()
    full_data_chunk = chunk_data[['DateTime']]
    
    timeframe_data_dict = {tf: load_data(f"{base_path}{file_name}_{tf}.csv") for tf in chosen_timeframes}
    
    for tf in chosen_timeframes:
        filtered_data = timeframe_data_dict[tf][(timeframe_data_dict[tf]['DateTime'] >= chunk_data['DateTime'].min()) & (timeframe_data_dict[tf]['DateTime'] <= chunk_data['DateTime'].max())]
        integrated_data = integrate_timeframes(chunk_data, filtered_data, target_timeframe, tf)
        full_data_chunk = pd.merge(full_data_chunk, integrated_data, on='DateTime', how='outer')

    return full_data_chunk

def fill_forward(df, timeframe, target_timeframe):
    target_seconds = parse_timeframe(target_timeframe)
    interval_seconds = parse_timeframe(timeframe)
    num_intervals = target_seconds // interval_seconds

    df['DateTime'] = pd.to_datetime(df['DateTime'])

    for index, row in df.iterrows():
        for i in range(num_intervals):
            cols = [f"{timeframe}_{x}_{i}" for x in ['Open', 'High', 'Low', 'Close', 'Volume']]
            if i == 0 and index > 0:
                for col in cols:
                    if pd.isnull(df.at[index, col]):
                        last_col = f"{timeframe}_{col.split('_')[1]}_{num_intervals - 1}"
                        df.at[index, col] = df.at[index - 1, last_col]
            elif i > 0:
                for col in cols:
                    if pd.isnull(df.at[index, col]):
                        prev_col = f"{timeframe}_{col.split('_')[1]}_{i - 1}"
                        df.at[index, col] = df.at[index, prev_col]

    return df

def process_files(base_path, save_path, file_names, target_timeframes_chosen_timeframes, prediction_type):
    database_name = os.path.join(save_path, f'multi_{prediction_type}_Train.db')
    conn = sqlite3.connect(database_name)
    
    for file_name in file_names:
        all_groups_data = []

        for target_timeframe, chosen_timeframes in sorted(target_timeframes_chosen_timeframes.items(), key=lambda x: parse_timeframe(x[0])):
            target_data = load_data(f"{base_path}{file_name}_{target_timeframe}.csv")

            chunk_size = 1024
            total_chunks = (len(target_data) + chunk_size - 1) // chunk_size
            max_workers = os.cpu_count()
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_chunk, i * chunk_size, min((i + 1) * chunk_size, len(target_data)), target_data, chosen_timeframes, base_path, file_name, target_timeframe) for i in range(total_chunks)]
                
                full_data = pd.concat(tqdm((future.result() for future in futures), total=total_chunks, desc=f"Processing {file_name} for {target_timeframe}"), ignore_index=True)

            for tf in chosen_timeframes:
                full_data = fill_forward(full_data, tf, target_timeframe)

            target_col_name = f"{target_timeframe}_{prediction_type}_0"
            full_data['Target'] = full_data[target_col_name].shift(-1)
            full_data = full_data[:-1]

            full_data.insert(1, 'Group', int(target_timeframe[:-1]))
            
            datetime_col = full_data['DateTime']
            group_col = full_data['Group']
            target_col = full_data['Target']
            full_data = full_data.drop(columns=['DateTime', 'Group', 'Target'])
            
            sorted_columns = sorted(full_data.columns, key=lambda x: (parse_timeframe(x.split('_')[0]), x))
            full_data = full_data[sorted_columns]
            
            full_data.insert(0, 'DateTime', datetime_col)
            full_data.insert(1, 'Group', group_col)
            full_data['Target'] = target_col

            # Fill any remaining null values with 0.0
            full_data = full_data.fillna(0.0)

            all_groups_data.append(full_data)

        if all_groups_data:
            full_data_combined = pd.concat(all_groups_data, axis=0, ignore_index=True)
            full_data_combined = full_data_combined.sort_values('DateTime')
            
            # Move Target column to the end
            target_column = full_data_combined.pop('Target')
            full_data_combined['Target'] = target_column
            
            full_data_combined.to_sql(file_name, conn, if_exists='replace', index=False)
            print(f"Data processing complete for {file_name}. Data has been saved to the SQLite database.")
        else:
            print(f"No data combined for {file_name}. Check the input files and parameters.")

    conn.close()

def main():
    base_path = 'C:\\Users\\crgon\\OneDrive\\Desktop\\trading\\data\\main_data\\'
    save_path = 'C:\\Users\\crgon\\OneDrive\\Desktop\\trading\\data\\'
    file_names = input("Enter the base file names without time interval or extension, separated by commas (e.g., 'XAUUSD,USA30IDXUSD'): ").split(',')

    target_timeframes_chosen_timeframes = {}
    while True:
        target_timeframe = input("What is the target timeframe? (e.g., '60m'): ")
        chosen_timeframes = input(f"Enter the timeframes to include for target timeframe {target_timeframe}, separated by commas (e.g., '10s,1m,5m,15m,30m'): ").split(',')
        chosen_timeframes = [tf for tf in chosen_timeframes if tf]
        target_timeframes_chosen_timeframes[target_timeframe] = chosen_timeframes
        add_another = input("Would you like to add another target timeframe? (yes/no): ").strip().lower()
        if add_another != 'yes':
            break

    prediction_type = input("Choose prediction class: Open, High, Low, or Close: ")

    process_files(base_path, save_path, file_names, target_timeframes_chosen_timeframes, prediction_type)

if __name__ == "__main__":
    main()