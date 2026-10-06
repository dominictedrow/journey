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

def detect_peaks_valleys(group_data, group_value, high_col, low_col):
    """
    An enhanced approach to detect peaks (sell signals) and valleys (buy signals) using 
    both local maxima/minima checks AND newly created technical feature thresholds 
    (RSI, Bollinger Bands, MACD, Volatility) for more robust and frequent signals.

    Peaks (sell signals) are set to 1000 and Valleys (buy signals) to 5000. 
    All other bars are labeled 0 (hold).

    We allow forward/backward window checks and then confirm peaks/valleys 
    based on technical conditions. The intent is to:
      1) Identify enough peaks and valleys to be useful for ML.
      2) Avoid consecutive identical signals (peak → peak or valley → valley).
      3) Optionally enforce a minimal distance between consecutive signals.

    This code ensures signals alternate where possible so that we don't have
    back-to-back peaks or valleys, which doesn't make sense in a typical chart.
    """

    # Initialize the target Series with zeros
    target = pd.Series(0, index=group_data.index)

    # Dynamically define a smaller window, so we detect more turning points
    # You can tune lookback further if you want more or fewer signals.
    if group_value == 1:
        lookback = 5
    elif group_value == 5:
        lookback = 4
    elif group_value == 15:
        lookback = 3
    else:
        # For 30-minute & 60-minute or any other timeframe, reduce lookback for more signals
        lookback = 2

    # Convert to arrays for direct indexing
    indices = group_data.index
    high_values = group_data[high_col].values
    low_values = group_data[low_col].values
    close_col = high_col.replace('High', 'Close')
    close_values = group_data[close_col].values

    # Among the "New_" features provided by 'add_new_features'
    rsi_values = group_data['New_RSI_14'].values if 'New_RSI_14' in group_data.columns else None
    bb_upper_values = group_data['New_BB_Upper'].values if 'New_BB_Upper' in group_data.columns else None
    bb_lower_values = group_data['New_BB_Lower'].values if 'New_BB_Lower' in group_data.columns else None
    macd_values = group_data['New_MACD'].values if 'New_MACD' in group_data.columns else None
    macd_signal_values = group_data['New_MACD_Signal'].values if 'New_MACD_Signal' in group_data.columns else None

    # We'll collect raw potential signals before final filtering
    signals = []

    # 1) Identify potential peaks/valleys based on local window + technical filters
    for i in range(lookback, len(group_data) - lookback):
        window_high = high_values[i - lookback : i + lookback + 1]
        window_low = low_values[i - lookback : i + lookback + 1]
        cur_high = high_values[i]
        cur_low = low_values[i]
        cur_close = close_values[i]

        peak_score = 0
        valley_score = 0

        # -- Local maxima/minima checks
        if cur_high == window_high.max():
            peak_score += 1
        if cur_low == window_low.min():
            valley_score += 1

        # -- RSI checks
        if rsi_values is not None:
            # RSI above ~55 → more likely peak
            if rsi_values[i] >= 55:
                peak_score += 1
            # RSI below ~45 → more likely valley
            if rsi_values[i] <= 45:
                valley_score += 1

        # -- Bollinger Bands checks
        if bb_upper_values is not None and bb_lower_values is not None:
            if bb_upper_values[i] != 0 and cur_close >= bb_upper_values[i] * 0.95:
                peak_score += 1
            if bb_lower_values[i] != 0 and cur_close <= bb_lower_values[i] * 1.05:
                valley_score += 1

        # -- MACD checks
        if macd_values is not None and macd_signal_values is not None:
            # MACD < Signal → turning downward → peak
            if macd_values[i] < macd_signal_values[i]:
                peak_score += 1
            # MACD > Signal → turning upward → valley
            if macd_values[i] > macd_signal_values[i]:
                valley_score += 1

        # Decide if it's a valid peak or valley (score threshold = 2)
        # If both are ≥2, pick whichever is stronger or default to peak on a tie
        if peak_score >= 2 and valley_score >= 2:
            if peak_score > valley_score:
                signals.append((indices[i], 'peak'))
            elif valley_score > peak_score:
                signals.append((indices[i], 'valley'))
            else:
                # tie → choose peak by default
                signals.append((indices[i], 'peak'))
        elif peak_score >= 2:
            signals.append((indices[i], 'peak'))
        elif valley_score >= 2:
            signals.append((indices[i], 'valley'))

    # 2) Sort signals by index
    signals.sort(key=lambda x: x[0])

    # 3) Filter signals to enforce:
    #    - No consecutive identical signals (peak→peak or valley→valley)
    #    - Optional minimal distance (in bars) between signals to avoid noise.
    signals_filtered = []
    min_distance = lookback  # You can tweak or remove this if undesired
    last_idx = None
    last_signal_type = None

    for idx, sig_type in signals:
        if not signals_filtered:
            # First signal → accept straight away
            signals_filtered.append((idx, sig_type))
            last_idx = idx
            last_signal_type = sig_type
        else:
            # Enforce minimal distance
            if (idx - last_idx) < min_distance:
                continue
            # Enforce alternation
            if sig_type == last_signal_type:
                continue
            # If it passes both checks, we accept the new signal
            signals_filtered.append((idx, sig_type))
            last_idx = idx
            last_signal_type = sig_type

    # 4) Mark the final signals in the target (1000 for peak, 5000 for valley)
    for idx, sig_type in signals_filtered:
        if sig_type == 'peak':
            target.loc[idx] = 1000
        elif sig_type == 'valley':
            target.loc[idx] = 5000

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
db_path = r"C:\Users\crgon\OneDrive\Desktop\trading\data\new_new_multi_stacked_HLC_Train_4.db"
output_db_path = r"C:\Users\crgon\OneDrive\Desktop\trading\data\new_multi_stacked_XAUUSD_Train_4.db"

# Process the database
process_database(db_path, output_db_path)