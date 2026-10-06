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
    A forward-looking, high-frequency peak/valley detection method that aims to capture
    many intermediate tops and bottoms—even those that might look small relative to
    enormous swings—so as to produce a dense set of turning points for training.

    How it works:
    1) Identify the relevant Close column for this Group/timeframe (e.g., "5m_Close_0").  
    2) For each bar i (in chronological order):
       a) Look ahead a small forward_window of bars (only forward in time—no large backward window):
          - If Close[i] is near the highest close in [i..i+forward_window], label it a potential peak.
          - If Close[i] is near the lowest close in that range, label it a potential valley.
          - It's possible (though rare) that a bar ties for both a local max and min (if the range is almost flat).
       b) Use “New_” indicators (RSI, Bollinger, MACD) to break ties or add partial confidence to one side.
          - Example: if RSI is high, tilt slightly toward “peak;” if RSI is low, tilt toward “valley.”
       c) Accumulate every potential peak or valley in raw_signals, so we get a lot of candidates.
    3) Sort raw_signals in time and filter:
       - Impose strict alternation (valley → peak → valley → peak).
       - Impose a small minimum distance in bars to avoid signals spamming back-to-back.
    4) Convert the final set of signals to 1000 (peak) or 5000 (valley).

    Compared to a classical large-window approach, this method produces far more
    signals—closely mimicking a human who sees every wiggle top or bottom in a price chart
    at a given zoom level. It can be noisy, but that is often beneficial for a model that
    needs many examples of turning points.  

    Parameters:
      group_data: Data subset for a particular Group/timeframe.
      group_value: The timeframe numeric (e.g., 5, 15, 30…) to choose parameters.
      high_col, low_col: Column names (like "5m_High_0", "5m_Low_0") from which we deduce the Close column.

    Returns:
      target: A pandas Series with the same index as group_data, containing:
        - 0 = no signal
        - 1000 = peak
        - 5000 = valley
    """

    import numpy as np
    import pandas as pd

    # Initialize the output target
    target = pd.Series(0, index=group_data.index)

    # Determine the relevant Close column (e.g. "5m_Close_0")
    close_col = high_col.replace("High", "Close")
    if close_col not in group_data.columns:
        return target

    close_values = group_data[close_col].values
    idx_array = group_data.index.values
    n_bars = len(close_values)

    # Grab “New_” indicators if available.
    rsi_data = group_data['New_RSI_14'].values if 'New_RSI_14' in group_data.columns else None
    bb_upper = group_data['New_BB_Upper'].values if 'New_BB_Upper' in group_data.columns else None
    bb_lower = group_data['New_BB_Lower'].values if 'New_BB_Lower' in group_data.columns else None
    macd_val = group_data['New_MACD'].values if 'New_MACD' in group_data.columns else None
    macd_sig = group_data['New_MACD_Signal'].values if 'New_MACD_Signal' in group_data.columns else None

    # 1) Choose a small forward-looking window to produce frequent signals.
    #    Increase or decrease these to tune how tight or loose the pivot detection is.
    if group_value == 1:
        forward_window = 6   # (8 bars = 8 minutes if it's 1-minute data)
    elif group_value == 5:
        forward_window = 6   # (8 bars * 5 minutes = 40 minutes look ahead)
    elif group_value == 15:
        forward_window = 4   # (6 bars * 15 min = 90 minutes)
    elif group_value == 30:
        forward_window = 4   # or 4-6
    else:
        forward_window = 4   # default is 4-6

    # 2) Tolerance factor to determine "near local max/min" within the forward window
    #    We use rolling std or a simpler approach to allow minor differences.
    #    Here, let's define a short rolling std used as a buffer so we don't require an exact match.
    vol_window = max(5, forward_window)
    rolling_std = pd.Series(close_values).rolling(vol_window).std().bfill().values
    rolling_std[np.isnan(rolling_std)] = 0
    buffer_factor = 0.15  # how close we must be to local max/min within the forward window

    raw_signals = []

    # 3) For each bar i, examine i..i+forward_window
    for i in range(n_bars - forward_window):
        current_close = close_values[i]

        # define the forward slice
        fwd_slice = close_values[i : i + forward_window + 1]
        local_max = fwd_slice.max()
        local_min = fwd_slice.min()
        tol = buffer_factor * rolling_std[i]  # local tolerance

        # Evaluate how close current_close is to local_max/min
        near_max = (current_close >= local_max - tol)
        near_min = (current_close <= local_min + tol)

        if not (near_max or near_min):
            # Not close to a local max or min in the forward window => skip
            continue

        # Build partial scores:
        peak_score = 0.0
        valley_score = 0.0

        if near_max:
            peak_score += 1.0
        if near_min:
            valley_score += 1.0

        # 4) Reinforce with the "New_" features
        #    For instance, RSI > 55 => more likely peak, RSI < 45 => more likely valley
        if rsi_data is not None:
            if rsi_data[i] > 50: # 55 was original
                peak_score += 0.5
            elif rsi_data[i] < 50: # 45 was original
                valley_score += 0.5

        # Bollinger
        if bb_upper is not None and bb_lower is not None:
            if bb_upper[i] != 0 and current_close >= 0.95 * bb_upper[i]:
                peak_score += 0.3
            if bb_lower[i] != 0 and current_close <= 1.05 * bb_lower[i]:
                valley_score += 0.3

        # MACD
        if macd_val is not None and macd_sig is not None:
            if macd_val[i] < macd_sig[i]:
                # downward bias => helps peak
                peak_score += 0.3
            else:
                # upward bias => helps valley
                valley_score += 0.3

        # Decide whether to label it a peak or valley
        # We allow ties => default to peak or valley, or skip. Let's default to separate them:
        #     if near both max and min, that means the range is extremely tight; pick whichever has bigger score.
        if peak_score >= 1.0 or valley_score >= 1.0:
            if peak_score > valley_score:
                raw_signals.append((idx_array[i], 'peak'))
            elif valley_score > peak_score:
                raw_signals.append((idx_array[i], 'valley'))
            else:
                # tie => pick peak, or pick valley. Let's pick peak by default for variety
                raw_signals.append((idx_array[i], 'peak'))

    # 5) Sort raw signals by time
    raw_signals.sort(key=lambda x: x[0])

    # 6) Filter signals to enforce alternation (valley→peak→valley→peak, etc.)
    #    plus a small min distance to avoid immediate duplicates.
    min_dist = max(3, forward_window)  # smaller => more signals (original: (1, forward_window // 2)

    final_signals = []
    last_idx = -999999999
    last_type = None

    for (sig_idx, sig_type) in raw_signals:
        # Enforce min distance
        if (sig_idx - last_idx) < min_dist:
            continue

        if not final_signals:
            # Accept the first one unconditionally
            final_signals.append((sig_idx, sig_type))
            last_idx = sig_idx
            last_type = sig_type
        else:
            # Strict alternation: if last_type == 'peak', new must be 'valley'
            #                     if last_type == 'valley', new must be 'peak'
            if last_type == 'peak' and sig_type == 'valley':
                final_signals.append((sig_idx, sig_type))
                last_idx = sig_idx
                last_type = sig_type
            elif last_type == 'valley' and sig_type == 'peak':
                final_signals.append((sig_idx, sig_type))
                last_idx = sig_idx
                last_type = sig_type
            else:
                # skip if it doesn't alternate
                continue

    # 7) Assign signals to the target as 1000 for peak, 5000 for valley
    for idx, stype in final_signals:
        if stype == 'peak':
            target.loc[idx] = 1000
        elif stype == 'valley':
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
db_path = r"C:\Users\crgon\OneDrive\Desktop\trading\data\new_new_multi_stacked_XAUUSD_Train_6.1.db"
output_db_path = r"C:\Users\crgon\OneDrive\Desktop\trading\data\new_multi_stacked_XAUUSD_Train_6.1.db"

# Process the database
process_database(db_path, output_db_path)