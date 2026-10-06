import pandas as pd
import numpy as np
import sqlite3
import pandas_ta as ta
from scipy.signal import argrelextrema

# Function to add new technical indicators without interfering with existing features
def add_new_features(df, close_col, high_col, low_col, open_col, group_value):
    """
    Enhanced version with fixed calculations and additional features.
    We now also add a 'New_MA_Long' with a large window, scaled down
    as we move to higher timeframes.
    """
    import pandas_ta as ta
    import numpy as np

    if df[close_col].isnull().all() or len(df) < 20:
        return df

    # Keep existing base calculations
    sma_length = group_value * 7
    rsi_length = group_value * 7
    macd_fast = group_value * 4
    macd_slow = group_value * 8
    macd_signal = group_value * 2
    bb_length = group_value * 7

    long_ma_lengths = {
        1: 48,   # 1-minute
        5: 21,   # 5-minute
        15: 21,   # 15-minute
        30: 21,   # 30-minute
        60: 21    # 60-minute
    }

    # Original features
    df['New_SMA'] = ta.sma(df[close_col], length=sma_length)
    df['New_RSI'] = ta.rsi(df[close_col], length=rsi_length)

    macd = ta.macd(df[close_col], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    df['New_MACD'] = macd[f'MACD_{macd_fast}_{macd_slow}_{macd_signal}']
    df['New_MACD_Signal'] = macd[f'MACDs_{macd_fast}_{macd_slow}_{macd_signal}']

    bbands = ta.bbands(df[close_col], length=bb_length, std=2)
    df['New_BB_Upper'] = bbands[f'BBU_{bb_length}_2.0']
    df['New_BB_Lower'] = bbands[f'BBL_{bb_length}_2.0']
    df['New_BB_Middle'] = bbands[f'BBM_{bb_length}_2.0']
    df['New_BB_Width'] = (df['New_BB_Upper'] - df['New_BB_Lower']) / df['New_BB_Middle']

    # 2) Long Moving Average (large window)
    long_ma_length = long_ma_lengths.get(group_value, 50)  # default 50 if not found
    df['New_MA_Long'] = ta.sma(df[close_col], length=long_ma_length)

    # Pattern-based features
    pattern_cols = [col for col in df.columns if col.startswith('Pattern_') and col[8:].isdigit()]
    if pattern_cols:
        df['New_Pattern_Sum'] = df[pattern_cols].sum(axis=1)
        df['New_Pattern_Mean'] = df[pattern_cols].mean(axis=1)
        
    if 'Pattern_Class' in df.columns:
        df['New_Pattern_Class_Norm'] = df['Pattern_Class'] / 360.0

    # Keep all of your other existing "New_" features if necessary...
    # Existing code to fill additional indicators:
    df['New_EMA_Fast'] = ta.ema(df[close_col], length=group_value * 2)
    df['New_EMA_Slow'] = ta.ema(df[close_col], length=group_value * 4)
    df['New_EMA_Trend'] = df['New_EMA_Fast'] - df['New_EMA_Slow']

    df['New_ATR'] = ta.atr(df[high_col], df[low_col], df[close_col], length=group_value)
    df['New_ATR_Percent'] = df['New_ATR'] / df[close_col] * 100

    stoch = ta.stoch(df[high_col], df[low_col], df[close_col], k=group_value, d=group_value)
    if not stoch.empty:
        df['New_Stoch_K'] = stoch.iloc[:, 0]
        df['New_Stoch_D'] = stoch.iloc[:, 1]

    df['New_ROC'] = ta.roc(df[close_col], length=group_value)
    df['New_MOM'] = ta.mom(df[close_col], length=group_value)
    df['New_HL_Range'] = (df[high_col] - df[low_col]) / df[close_col] * 100
    df['New_CO_Range'] = (df[close_col] - df[open_col]) / df[open_col] * 100
    df['New_Price_Volume_Trend'] = (df[close_col] - df[close_col].shift(1)) * df[open_col]

    df['New_Trend_Strength'] = (
        df[close_col] - df[close_col].rolling(window=group_value*3).mean()
    ) / df[close_col].rolling(window=group_value*3).std()

    vol_window = group_value * 3
    df['New_Vol_Regime'] = df[close_col].rolling(window=vol_window).std() / \
                           df[close_col].rolling(window=vol_window*2).std()

    df['New_Price_Distance'] = (
        (df[close_col] - df[close_col].rolling(window=group_value*2).mean()) /
        df[close_col].rolling(window=group_value*2).std()
    )

    df['New_Support_Level'] = df[low_col].rolling(window=group_value*2).min()
    df['New_Resistance_Level'] = df[high_col].rolling(window=group_value*2).max()
    df['New_Price_Position'] = (df[close_col] - df['New_Support_Level']) / (
        df['New_Resistance_Level'] - df['New_Support_Level']
    )

    df['New_Price_Mom'] = df[close_col].diff(group_value)
    df['New_RSI_Mom'] = df['New_RSI'].diff(group_value)

    df = df.replace([np.inf, -np.inf], np.nan)

    return df

def detect_peaks_valleys(group_data, group_value, high_col, low_col):
    """
    Revised peak/valley detection method:
      - Heavily relies on technical indicators (RSI, BB, MACD, Long MA).
      - Removed pattern_sum, pattern_mean, pattern_class_norm from scoring.
      - Uses a larger forward_window.
      - Labels remain the same: peak -> 5000, valley -> 1000, carrying forward
        until the next signal. No zero labels at start.
    """

    import numpy as np
    import pandas as pd

    target = pd.Series(0, index=group_data.index)

    if 'New_MA_Long' not in group_data.columns:
        return target
    
    ma_long = group_data['New_MA_Long'].values
    rsi_data = group_data['New_RSI'].values if 'New_RSI' in group_data.columns else None
    bb_upper = group_data['New_BB_Upper'].values if 'New_BB_Upper' in group_data.columns else None
    bb_lower = group_data['New_BB_Lower'].values if 'New_BB_Lower' in group_data.columns else None
    macd_val = group_data['New_MACD'].values if 'New_MACD' in group_data.columns else None
    macd_sig = group_data['New_MACD_Signal'].values if 'New_MACD_Signal' in group_data.columns else None
    higher_timeframe_signals = group_data['higher_timeframe_target'].values if 'higher_timeframe_target' in group_data.columns else None

    idx_array = group_data.index.values
    n_bars = len(ma_long)

    fw_dict = {
        1: 20,
        5: 6,
        15: 9,
        30: 12,
        60: 6
    }
    forward_window = fw_dict.get(group_value, 5)

    # Rolling std for local tolerance
    vol_window = max(12, forward_window)
    rolling_std_ma = pd.Series(ma_long).rolling(vol_window).std().bfill().values
    rolling_std_ma[np.isnan(rolling_std_ma)] = 0

    raw_signals = []

    for i in range(n_bars - forward_window):
        peak_score = 0.0
        valley_score = 0.0

        fwd_slice = ma_long[i : i + forward_window + 1]
        ma_local_max = fwd_slice.max()
        ma_local_min = fwd_slice.min()

        # ------------------------------------------------------------------
        # Dynamic buffer factor: ratio of local MA range over mean, clamped.
        # ------------------------------------------------------------------
        local_range = ma_local_max - ma_local_min
        local_mean = np.mean(fwd_slice) if np.mean(fwd_slice) != 0 else 1e-12
        ratio = local_range / local_mean
        buffer_factor = np.clip(ratio, 0.2, 0.6)  # adapt min/max as desired

        tol = buffer_factor * rolling_std_ma[i]

        near_ma_max = (ma_long[i] >= ma_local_max - tol)
        near_ma_min = (ma_long[i] <= ma_local_min + tol)

        if near_ma_max:
            peak_score += 1.0
        if near_ma_min:
            valley_score += 1.0

        if rsi_data is not None:
            if rsi_data[i] > 65:
                peak_score += 0.35
            elif rsi_data[i] < 35:
                valley_score += 0.35

        if bb_upper is not None and bb_lower is not None:
            if bb_upper[i] != 0 and ma_long[i] >= bb_upper[i]:
                peak_score += 0.25
            if bb_lower[i] != 0 and ma_long[i] <= bb_lower[i]:
                valley_score += 0.25

        if macd_val is not None and macd_sig is not None:
            if macd_val[i] > macd_sig[i]:
                peak_score += 0.25
            else:
                valley_score += 0.25

        if higher_timeframe_signals is not None:
            if higher_timeframe_signals[i] == 5000:
                peak_score += 0.25
            elif higher_timeframe_signals[i] == 1000:
                valley_score += 0.25

        if peak_score >= 1.0 or valley_score >= 1.0:
            if peak_score > valley_score:
                raw_signals.append((idx_array[i], 'peak'))
            elif valley_score > peak_score:
                raw_signals.append((idx_array[i], 'valley'))
            else:
                slope = ma_long[i] - ma_long[i - 1] if i > 0 else 0
                if slope < 0:
                    raw_signals.append((idx_array[i], 'peak'))
                elif slope > 0:
                    raw_signals.append((idx_array[i], 'valley'))
                else:
                    if rsi_data is not None:
                        if rsi_data[i] >= 50:
                            raw_signals.append((idx_array[i], 'peak'))
                        else:
                            raw_signals.append((idx_array[i], 'valley'))
                    else:
                        continue

    raw_signals.sort(key=lambda x: x[0])

    if group_value <= 5:
        min_dist = 12
    elif group_value == 15:
        min_dist = 9
    elif group_value == 30:
        min_dist = 9
    elif group_value == 60:
        min_dist = 6

    final_signals = []
    last_idx = -9999999
    last_type = None

    for (sig_idx, sig_type) in raw_signals:
        if (sig_idx - last_idx) < min_dist:
            continue
        if not final_signals:
            final_signals.append((sig_idx, sig_type))
            last_idx = sig_idx
            last_type = sig_type
        else:
            if last_type != sig_type:
                final_signals.append((sig_idx, sig_type))
                last_idx = sig_idx
                last_type = sig_type

    if not final_signals:
        return target

    for i in range(len(final_signals)):
        current_idx = final_signals[i][0]
        current_type = final_signals[i][1]
        segment_label = 5000 if current_type == 'peak' else 1000
        if i < len(final_signals) - 1:
            next_idx = final_signals[i + 1][0]
            target.loc[current_idx: next_idx - 1] = segment_label
        else:
            target.loc[current_idx:] = segment_label

    nonzero_indices = target[target != 0].index
    if len(nonzero_indices) > 0:
        first_nonzero = nonzero_indices[0]
        zero_slice = target.loc[:first_nonzero - 1]
        if not zero_slice.empty:
            target.loc[zero_slice.index] = np.nan

    return target

def process_table(df, group_timeframes):
    """
    Process each table with enhanced timeframe alignment for peaks and valleys
    """
    # Remove existing Target4 if it exists
    if 'Target4' in df.columns:
        df.drop(columns=['Target4'], inplace=True)

    # Process timeframes from highest to lowest
    for group_value in sorted(group_timeframes, reverse=True):
        col_prefix = f"{group_value}m_"
        required_cols = [f"{col_prefix}{t}_0" for t in ['Close', 'High', 'Low', 'Open']]

        if all(col in df.columns for col in required_cols):
            close_col, high_col, low_col, open_col = required_cols
            group_mask = (df['Group'] == group_value)
            group_data = df.loc[group_mask].copy()

            if not group_data.empty:
                # Add technical features
                group_data = add_new_features(group_data, close_col, high_col, low_col, open_col, group_value)
                
                # Clean up NaN values
                new_cols = [c for c in group_data.columns if c.startswith('New_')]
                group_data.dropna(subset=new_cols, how='any', inplace=True)

                # Generate targets
                target_vals = detect_peaks_valleys(group_data, group_value, high_col, low_col)
                group_data['Target4'] = target_vals

                # Store signals for lower timeframe alignment
                if group_value != min(group_timeframes):
                    df.loc[group_mask, 'higher_timeframe_target'] = target_vals

                # Update main dataframe
                df.drop(index=df[group_mask].index, inplace=True)
                df = pd.concat([df, group_data], axis=0)

    # Clean up temporary columns
    if 'higher_timeframe_target' in df.columns:
        df.drop(columns=['higher_timeframe_target'], inplace=True)

    # Sort and reorder columns
    df.sort_index(inplace=True)
    new_feature_cols = [c for c in df.columns if c.startswith('New_')]
    target_cols = [c for c in df.columns if c.startswith('Target') and c != 'Target4']
    non_target_cols = [c for c in df.columns if not c.startswith('Target') and not c.startswith('New_')]
    df = df[non_target_cols + new_feature_cols + target_cols + ['Target4']]

    # -----------------------------
    # Replace the existing signal correction with a short-run merge
    # -----------------------------
    unique_groups_in_df = df['Group'].unique()

    for gval in unique_groups_in_df:
        # Assign a min_dist
        if gval <= 5:
            min_dist = 12
        elif gval == 15:
            min_dist = 9
        elif gval == 30:
            min_dist = 9
        elif gval == 60:
            min_dist = 6
        else:
            min_dist = 6

        # Isolate this group's Target4 as a NumPy array
        group_mask = df['Group'] == gval        
        target_array = df.loc[group_mask, 'Target4'].values

        # We only care about rows that are 1000 or 5000
        valid_mask = np.isin(target_array, [1000, 5000])

        if not np.any(valid_mask):
            # No signals, just continue
            continue

        # Convert to runs (start_idx, end_idx, value)
        runs = []
        N = len(target_array)
        i = 0
        while i < N:
            if valid_mask[i]:
                val = target_array[i]
                start_idx = i
                while i + 1 < N and target_array[i+1] == val:
                    i += 1
                end_idx = i
                runs.append((start_idx, end_idx, val))
            i += 1

        # Merge short runs if length < min_dist
        merged_runs = []
        for run in runs:
            start_i, end_i, val = run
            run_len = end_i - start_i + 1

            if run_len < min_dist:
                # Attempt to unify with previous run if same val,
                # otherwise unify with next run if same val,
                # else unify with previous run by default.
                if merged_runs:
                    prev_start, prev_end, prev_val = merged_runs[-1]
                    if prev_val == val:
                        # Extend previous run
                        merged_runs[-1] = (prev_start, end_i, prev_val)
                    else:
                        # Look ahead if next run has same val
                        # We check runs[] directly if there's a next run
                        idx_of_run = runs.index(run)
                        if idx_of_run < len(runs) - 1:
                            nxt_start, nxt_end, nxt_val = runs[idx_of_run + 1]
                            if nxt_val == val:
                                # We'll unify with next run by marking this
                                # run but leaving the merges for post-pass
                                # For simplicity, just flip to prev_val here
                                # if not matching next:
                                # but the user specifically wants
                                # to unify with *somebody* if possible.
                                # Let's unify with next if it matches:
                                # We'll store it as is, then unify in a second pass
                                pass
                            else:
                                # unify with previous anyway
                                merged_runs[-1] = (prev_start, end_i, prev_val)
                                val = prev_val
                        else:
                            # unify with previous (no next)
                            merged_runs[-1] = (prev_start, end_i, prev_val)
                            val = prev_val

                        # Add a dummy run to keep indexing consistent
                        # if we didn't unify with next
                        if val != run[2]:
                            # We changed the run's val
                            continue
                        else:
                            # We didn't unify with prev, keep run as is
                            merged_runs.append((start_i, end_i, val))
                    # We either merged or appended; no duplication needed
                else:
                    # No previous run, unify with next if same val
                    idx_of_run = runs.index(run)
                    if idx_of_run < len(runs) - 1:
                        nxt_start, nxt_end, nxt_val = runs[idx_of_run + 1]
                        if nxt_val == val:
                            # We basically skip this run,
                            # unify with the next by letting next run handle it
                            # Just skip
                            pass
                        else:
                            # or we accept this run as is if there's no match
                            merged_runs.append((start_i, end_i, val))
                    else:
                        # It's the only run, leave as is
                        merged_runs.append((start_i, end_i, val))
            else:
                # run is large enough, keep it
                merged_runs.append(run)

        # Second pass: unify small leftover runs with the next if needed
        # (in case the short run was right before a matching run)
        final_runs = []
        i = 0
        while i < len(merged_runs):
            s_i, e_i, v = merged_runs[i]
            length_i = e_i - s_i + 1

            if length_i < min_dist and i < len(merged_runs) - 1:
                # check next
                s_nxt, e_nxt, v_nxt = merged_runs[i+1]
                if v_nxt == v:
                    # merge with next
                    final_runs.append((s_i, e_nxt, v))
                    i += 2
                    continue
            final_runs.append((s_i, e_i, v))
            i += 1

        # Overwrite target_array
        new_target = target_array.copy()
        # first clear out old signals
        new_target[:] = 0
        for (start_i, end_i, val) in final_runs:
            new_target[start_i:end_i+1] = val

        df.loc[group_mask, 'Target4'] = new_target

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

# Paths to original and processed databases (example)
# Paths to original and processed databases
db_path = r"C:\Users\crgon\OneDrive\Desktop\trading\data\backup\processed_multi_stacked_XAUUSD_Train_5_15_30_60.db"
output_db_path = r"C:\Users\crgon\OneDrive\Desktop\trading\data\new_multi_stacked_XAUUSD_Train_11_new.db"

process_database(db_path, output_db_path)