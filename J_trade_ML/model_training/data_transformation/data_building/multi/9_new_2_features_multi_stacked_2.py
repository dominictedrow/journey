import pandas as pd
import numpy as np
import sqlite3
import pandas_ta as ta
from scipy.signal import argrelextrema

# Function to add new technical indicators without interfering with existing features
def add_new_features(df, close_col, high_col, low_col, open_col):
    """
    Add new technical indicators as features, preserving the original dataset.
    This includes moving averages, RSI, MACD, and Bollinger Bands.
    """
    if df[close_col].isnull().all() or len(df) < 20:
        return df

    # New Moving Averages
    df['New_SMA_5'] = ta.sma(df[close_col], length=5)
    df['New_SMA_20'] = ta.sma(df[close_col], length=20)

    # New RSI (Relative Strength Index)
    df['New_RSI_14'] = ta.rsi(df[close_col], length=14)

    # New MACD (Moving Average Convergence Divergence)
    macd = ta.macd(df[close_col], fast=12, slow=26, signal=9)
    df['New_MACD'] = macd['MACD_12_26_9']
    df['New_MACD_Signal'] = macd['MACDs_12_26_9']

    # New Bollinger Bands
    bbands = ta.bbands(df[close_col], length=20, std=2)
    df['New_BB_Upper'] = bbands['BBU_20_2.0']
    df['New_BB_Lower'] = bbands['BBL_20_2.0']

    # New Volatility
    df['New_Volatility_10'] = df[close_col].rolling(window=10).std()

    return df

# Function to detect peaks (sell) and valleys (buy) ensuring signals alternate
def detect_peaks_valleys(group_data, group_value, high_col, low_col):
    """
    Detect peaks and valleys ensuring the signals alternate.
    Peaks (sell signals) are set to 1000, Valleys (buy signals) to 5000, and other rows to 0.
    The signal is based on the current row's data and its surrounding window.
    Uses future data for better peak and valley detection.
    """
    # Initialize the target Series with zeros and the same index as group_data
    target = pd.Series(0, index=group_data.index)

    # Define the window size dynamically based on the Group (timeframe)
    if group_value == 1:
        lookback = 10  # For 1-minute timeframe, moderate window
    elif group_value == 5:
        lookback = 8   # For 5-minute timeframe, smaller window for quicker signals
    elif group_value == 15:
        lookback = 6   # For 15-minute timeframe, slightly smaller for more signals
    elif group_value == 30:
        lookback = 4   # For 30-minute timeframe, smaller window to increase signal frequency
    elif group_value == 60:
        lookback = 3   # For 60-minute timeframe, very small window to generate more signals
    else:
        lookback = 5  # Default value

    window_size = lookback * 2 + 1

    # Prepare arrays for high and low values
    high_values = group_data[high_col].values
    low_values = group_data[low_col].values

    indices = group_data.index

    # List to store detected signals
    signals = []

    # Loop over data, excluding edges where window is incomplete
    for i in range(lookback, len(group_data) - lookback):
        window_high = high_values[i - lookback: i + lookback + 1]
        window_low = low_values[i - lookback: i + lookback + 1]
        current_high = high_values[i]
        current_low = low_values[i]

        # Check for peak (local maximum)
        if current_high == window_high.max():
            signals.append((indices[i], 'peak'))

        # Check for valley (local minimum)
        if current_low == window_low.min():
            signals.append((indices[i], 'valley'))

    # Sort signals by index
    signals.sort(key=lambda x: x[0])

    # Ensure signals alternate
    last_signal = None
    signals_filtered = []

    for idx, sig_type in signals:
        if last_signal is None:
            # Accept the first signal
            signals_filtered.append((idx, sig_type))
            last_signal = sig_type
        else:
            if sig_type != last_signal:
                # Accept signal and update last_signal
                signals_filtered.append((idx, sig_type))
                last_signal = sig_type
            else:
                # Skip signal, as it's the same as last one
                pass

    # Assign signals to target Series
    for idx, sig_type in signals_filtered:
        if sig_type == 'peak':
            target.loc[idx] = 1000  # Peak (Sell Signal)
        elif sig_type == 'valley':
            target.loc[idx] = 5000  # Valley (Buy Signal)

    return target

# Process Each Group in the Table
def process_table(df, group_timeframes):
    """
    Process each table, adding technical indicators and generating buy/sell signals 
    for each group/timeframe. Ensure new features are inserted before Target1, Target2, Target3.
    """
    # Remove existing 'Target4' column if it exists to avoid duplication
    if 'Target4' in df.columns:
        df.drop(columns=['Target4'], inplace=True)

    # Add technical indicators for each timeframe without interfering with other features
    for group_value in group_timeframes:
        col_prefix = f'{group_value}m_'

        # Identify the columns corresponding to the current group
        required_cols = [f'{col_prefix}{type}_0' for type in ['Close', 'High', 'Low', 'Open']]
        if all(col in df.columns for col in required_cols):
            close_col, high_col, low_col, open_col = required_cols

            # Filter the data for the current group (timeframe)
            group_data = df.loc[df['Group'] == group_value].copy()

            if not group_data.empty:
                # Step 1: Add new technical indicators (no change to original features)
                group_data = add_new_features(group_data, close_col, high_col, low_col, open_col)

                # Step 2: Detect peaks and valleys
                target_vals = detect_peaks_valleys(group_data, group_value, high_col, low_col)

                # Assign the calculated values back into the original dataframe for the current group
                df.loc[df['Group'] == group_value, 'Target4'] = target_vals

                # Add the new technical indicators back to the main dataframe without modifying existing features
                new_cols = [col for col in group_data.columns if 'New_' in col]
                df.loc[df['Group'] == group_value, new_cols] = group_data[new_cols]

    # Remove rows from the top where any of the new features are NaN
    new_feature_cols = [col for col in df.columns if 'New_' in col]
    last_invalid_index = df[new_feature_cols].isna().any(axis=1)[::-1].idxmax()
    df = df.iloc[last_invalid_index + 1:]

    # Reorder columns to ensure new features come before Target1, Target2, and Target3
    non_target_cols = [col for col in df.columns if 'Target' not in col]
    target_cols = [col for col in df.columns if 'Target' in col and col != 'Target4']
    df = df[non_target_cols + target_cols + ['Target4']]  # Ensure Target4 is the last column

    return df

# Database Processing
def process_database(db_path, output_db_path):
    """
    Connect to the database, process each table, and save the results.
    """
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)

    # Get the list of table names
    table_query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql_query(table_query, conn)

    # Valid timeframes/groups
    group_timeframes = [1, 5, 15, 30, 60]

    # Loop through each table and process it
    for table in tables['name']:
        # Load the table data
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)

        # Process the DataFrame
        df = process_table(df, group_timeframes)

        # Save the processed data into the output SQLite file
        output_conn = sqlite3.connect(output_db_path)
        df.to_sql(table, output_conn, if_exists='replace', index=False)
        output_conn.close()

    # Close the original database connection
    conn.close()
    print("Processing complete. The new file with the 'Target4' column has been saved.")

# Paths to original and processed databases
db_path = r"C:\Users\crgon\OneDrive\Desktop\trading\data\Processed_multi_stacked_HLC_Train_2000.db"
output_db_path = r"C:\Users\crgon\OneDrive\Desktop\trading\data\Processed_multi_stacked_HLC_Train_3.db"

# Process the database
process_database(db_path, output_db_path)