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

# Function to detect peaks (sell) and valleys (buy) using local extrema detection with dynamic lookback windows
def detect_peaks_valleys(group_data, group_value, high_col, low_col):
    """
    Detect peaks and valleys using local extrema detection.
    Peaks (sell signals) are set to 1000, Valleys (buy signals) to 5000, and other rows to 0.
    The signal is based on the current row's data and its surrounding window.
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

    # Find local maxima (peaks) in the High column and local minima (valleys) in the Low column
    peaks = argrelextrema(group_data[high_col].values, np.greater_equal, order=lookback)[0]
    valleys = argrelextrema(group_data[low_col].values, np.less_equal, order=lookback)[0]

    # Assign the peaks and valleys to the target Series using .iloc to maintain alignment
    target.iloc[peaks] = 1000   # Peak (Sell Signal)
    target.iloc[valleys] = 5000  # Valley (Buy Signal)

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
db_path = r"C:\Users\crgon\OneDrive\Desktop\trading\data\multi_stacked_HLC_Train.db"
output_db_path = r"C:\Users\crgon\OneDrive\Desktop\trading\data\Processed_multi_stacked_HLC_Train_3.db"

# Process the database
process_database(db_path, output_db_path)
