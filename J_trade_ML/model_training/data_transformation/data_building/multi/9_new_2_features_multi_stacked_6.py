import pandas as pd
import numpy as np
import sqlite3
import pandas_ta as ta
from scipy.signal import argrelextrema

# Function to add new technical indicators without interfering with existing features
def add_new_features(df, close_col, high_col, low_col, open_col, group_value):
    """
    Enhanced version with fixed calculations and additional features
    """

    if df[close_col].isnull().all() or len(df) < 20:
        return df

    # Keep existing base calculations
    sma_length = group_value * 5
    rsi_length = group_value * 3
    macd_fast = group_value * 2
    macd_slow = group_value * 4
    macd_signal = group_value
    bb_length = group_value * 4

    # Original features (keeping these intact)
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
    
    # Pattern-based features
    pattern_cols = [col for col in df.columns if col.startswith('Pattern_') and col[8:].isdigit()]
    if pattern_cols:
        df['New_Pattern_Sum'] = df[pattern_cols].sum(axis=1)
        df['New_Pattern_Mean'] = df[pattern_cols].mean(axis=1)
        
    if 'Pattern_Class' in df.columns:
        df['New_Pattern_Class_Norm'] = df['Pattern_Class'] / 360.0
    
    # Fixed and Enhanced Technical Indicators
    
    # 1. Trend Indicators
    df['New_EMA_Fast'] = ta.ema(df[close_col], length=group_value * 2)
    df['New_EMA_Slow'] = ta.ema(df[close_col], length=group_value * 4)
    df['New_EMA_Trend'] = df['New_EMA_Fast'] - df['New_EMA_Slow']
    
    # 2. Volatility Indicators
    df['New_ATR'] = ta.atr(df[high_col], df[low_col], df[close_col], length=group_value)
    df['New_ATR_Percent'] = df['New_ATR'] / df[close_col] * 100
    
    # 3. Stochastic
    stoch = ta.stoch(df[high_col], df[low_col], df[close_col], k=group_value, d=group_value)
    if not stoch.empty:
        df['New_Stoch_K'] = stoch.iloc[:, 0]
        df['New_Stoch_D'] = stoch.iloc[:, 1]
    
    # 4. Momentum Indicators
    df['New_ROC'] = ta.roc(df[close_col], length=group_value)
    df['New_MOM'] = ta.mom(df[close_col], length=group_value)
    
    # 5. Price Action Features
    df['New_HL_Range'] = (df[high_col] - df[low_col]) / df[close_col] * 100
    df['New_CO_Range'] = (df[close_col] - df[open_col]) / df[open_col] * 100
    
    # 6. Price-Volume Trend
    df['New_Price_Volume_Trend'] = (df[close_col] - df[close_col].shift(1)) * df[open_col]
    
    # 7. Trend Strength
    df['New_Trend_Strength'] = (
        df[close_col] - df[close_col].rolling(window=group_value*3).mean()
    ) / df[close_col].rolling(window=group_value*3).std()
    
    # 8. Volatility Regime
    vol_window = group_value * 3
    df['New_Vol_Regime'] = df[close_col].rolling(window=vol_window).std() / \
                          df[close_col].rolling(window=vol_window*2).std()
    
    # 9. Price Patterns (ensuring integer type)
    df['New_Higher_High'] = (
        df[high_col] > df[high_col].rolling(window=group_value).max().shift(1)
    ).astype(np.int32)
    
    df['New_Lower_Low'] = (
        df[low_col] < df[low_col].rolling(window=group_value).min().shift(1)
    ).astype(np.int32)
    
    # 10. Trend Reversal
    df['New_Price_Distance'] = (
        (df[close_col] - df[close_col].rolling(window=group_value*2).mean()) /
        df[close_col].rolling(window=group_value*2).std()
    )
    
    # 11. Support/Resistance
    df['New_Support_Level'] = df[low_col].rolling(window=group_value*2).min()
    df['New_Resistance_Level'] = df[high_col].rolling(window=group_value*2).max()
    df['New_Price_Position'] = (df[close_col] - df['New_Support_Level']) / \
                              (df['New_Resistance_Level'] - df['New_Support_Level'])
    
    # 12. Momentum Divergence (ensuring integer type)
    df['New_Price_Mom'] = df[close_col].diff(group_value)
    df['New_RSI_Mom'] = df['New_RSI'].diff(group_value)
    
    # Calculate Momentum Divergence as integers (-1, 0, 1)
    conditions = [
        (df['New_Price_Mom'] > 0) & (df['New_RSI_Mom'] < 0),
        (df['New_Price_Mom'] < 0) & (df['New_RSI_Mom'] > 0)
    ]
    choices = [-1, 1]
    df['New_Mom_Divergence'] = np.select(conditions, choices, default=0).astype(np.int32)
    
    # Clean up any infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df


def detect_peaks_valleys(group_data, group_value, high_col, low_col):
    """
    Enhanced forward-looking, high-frequency peak/valley detection method with
    timeframe alignment and pattern-based features integration.
    """
    import numpy as np
    import pandas as pd

    # Initialize the output target
    target = pd.Series(0, index=group_data.index)

    # Relevant Close column
    close_col = high_col.replace("High", "Close")
    if close_col not in group_data.columns:
        return target

    close_values = group_data[close_col].values
    idx_array = group_data.index.values
    n_bars = len(close_values)

    # Get technical indicators
    rsi_data = group_data['New_RSI'].values if 'New_RSI' in group_data.columns else None
    bb_upper = group_data['New_BB_Upper'].values if 'New_BB_Upper' in group_data.columns else None
    bb_lower = group_data['New_BB_Lower'].values if 'New_BB_Lower' in group_data.columns else None
    macd_val = group_data['New_MACD'].values if 'New_MACD' in group_data.columns else None
    macd_sig = group_data['New_MACD_Signal'].values if 'New_MACD_Signal' in group_data.columns else None
    pattern_sum = group_data['New_Pattern_Sum'].values if 'New_Pattern_Sum' in group_data.columns else None
    pattern_mean = group_data['New_Pattern_Mean'].values if 'New_Pattern_Mean' in group_data.columns else None
    pattern_class_norm = group_data['New_Pattern_Class_Norm'].values if 'New_Pattern_Class_Norm' in group_data.columns else None

    # Timeframe-specific parameters
    if group_value == 1:
        forward_window = 4
        min_dist = 8
        buffer_factor = 0.25
        rsi_threshold_high = 65
        rsi_threshold_low = 35
    elif group_value == 5:
        forward_window = 4
        min_dist = 8
        buffer_factor = 0.30
        rsi_threshold_high = 63
        rsi_threshold_low = 37
    elif group_value == 15:
        forward_window = 4
        min_dist = 6
        buffer_factor = 0.35
        rsi_threshold_high = 60
        rsi_threshold_low = 40
    elif group_value == 30:
        forward_window = 4
        min_dist = 5
        buffer_factor = 0.40
        rsi_threshold_high = 58
        rsi_threshold_low = 42
    else:  # 60m
        forward_window = 4
        min_dist = 4
        buffer_factor = 0.45
        rsi_threshold_high = 55
        rsi_threshold_low = 45

    # Get higher timeframe signals if available
    higher_timeframe_signals = group_data['higher_timeframe_target'].values if 'higher_timeframe_target' in group_data.columns else None

    # Calculate rolling volatility
    vol_window = max(12, forward_window)
    rolling_std = pd.Series(close_values).rolling(vol_window).std().bfill().values
    rolling_std[np.isnan(rolling_std)] = 0

    raw_signals = []

    for i in range(n_bars - forward_window):
        current_close = close_values[i]
        
        # Forward-looking price analysis
        fwd_slice = close_values[i : i + forward_window + 1]
        local_max = fwd_slice.max()
        local_min = fwd_slice.min()
        tol = buffer_factor * rolling_std[i]

        near_max = (current_close >= local_max - tol)
        near_min = (current_close <= local_min + tol)

        if not (near_max or near_min):
            continue

        peak_score = 0.0
        valley_score = 0.0

        # Base scores from price action
        if near_max:
            peak_score += 1.0
        if near_min:
            valley_score += 1.0

        # Technical indicator scoring
        if rsi_data is not None:
            if rsi_data[i] > rsi_threshold_high:
                peak_score += 0.5
            elif rsi_data[i] < rsi_threshold_low:
                valley_score += 0.5

        if bb_upper is not None and bb_lower is not None:
            if bb_upper[i] != 0 and current_close >= 0.95 * bb_upper[i]:
                peak_score += 0.3
            if bb_lower[i] != 0 and current_close <= 1.05 * bb_lower[i]:
                valley_score += 0.3

        if macd_val is not None and macd_sig is not None:
            if macd_val[i] > macd_sig[i]:
                peak_score += 0.3
            else:
                valley_score += 0.3

        # Pattern-based scoring
        if pattern_sum is not None:
            if pattern_sum[i] > 130:
                peak_score += 0.4
            elif pattern_sum[i] < 65:
                valley_score += 0.4

        if pattern_mean is not None:
            if pattern_mean[i] > 4.5:
                peak_score += 0.2
            elif pattern_mean[i] < 2.0:
                valley_score += 0.2

        if pattern_class_norm is not None and not np.isnan(pattern_class_norm[i]):
            if pattern_class_norm[i] > 0.7:
                peak_score += 0.3
            elif pattern_class_norm[i] < 0.3:
                valley_score += 0.3

        # Higher timeframe alignment
        if higher_timeframe_signals is not None:
            if higher_timeframe_signals[i] == 1000:
                peak_score += 0.3
            elif higher_timeframe_signals[i] == 5000:
                valley_score += 0.3

        # Signal decision
        if peak_score >= 1.0 or valley_score >= 1.0:
            if peak_score > valley_score:
                raw_signals.append((idx_array[i], 'peak'))
            elif valley_score > peak_score:
                raw_signals.append((idx_array[i], 'valley'))
            else:
                raw_signals.append((idx_array[i], 'peak'))

    # Sort and filter signals
    raw_signals.sort(key=lambda x: x[0])
    final_signals = []
    last_idx = -999999999
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

    # Assign final signals
    for idx, stype in final_signals:
        target.loc[idx] = 1000 if stype == 'peak' else 5000

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
db_path = r"C:\Users\crgon\OneDrive\Desktop\trading\data\backup\new_new_multi_stacked_XAUUSD_Train_5.db"
output_db_path = r"C:\Users\crgon\OneDrive\Desktop\trading\data\new_multi_stacked_XAUUSD_Train_7.db"

process_database(db_path, output_db_path)