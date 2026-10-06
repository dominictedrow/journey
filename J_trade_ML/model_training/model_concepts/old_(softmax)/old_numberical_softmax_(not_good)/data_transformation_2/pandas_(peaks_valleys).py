import pandas as pd
import numpy as np
from tqdm import tqdm

# Load the CSV data
df = pd.read_csv(r"C:\Users\crgon\OneDrive\Desktop\thinkbe_app\ai_side_projects\trading\XAUUSD.csv")

# Convert "Date" and "Timestamp" columns to string
df['Date'] = df['Date'].astype(str)
df['Timestamp'] = df['Timestamp'].astype(str)

# Concatenate "Date" and "Timestamp" and convert to datetime
df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Timestamp'])
df.set_index('Timestamp', inplace=True)

# Create a new column "Target" and initialize it with "hold"
df['Target'] = 'hold'

# Calculate rolling mean and standard deviation
window = 15
df['Rolling_Mean'] = df['Close'].rolling(window).mean()

# Identify peaks
df['is_peak'] = df['Close'] > df['Rolling_Mean'].shift(-window)

# Identify valleys
df['is_valley'] = df['Close'] < df['Rolling_Mean'].shift(window)8

# Initialize variables
active_trades = []
profit = 0

# Iterate through the DataFrame row by row with a progress bar
for i in tqdm(range(len(df))):
    if df.iloc[i]['is_peak'] and active_trades:
        sell_price = df.iloc[i]['Close']
        profit += sum(sell_price - trade_price for trade_price in active_trades)
        active_trades = []
        df.iat[i, df.columns.get_loc('Target')] = 'sell'
    elif df.iloc[i]['is_valley']:
        active_trades.append(df.iloc[i]['Close'])
        df.iat[i, df.columns.get_loc('Target')] = 'buy'

print(f"Total profit: {profit}")

# Remove intermediate columns
df = df.drop(['Rolling_Mean', 'is_peak', 'is_valley'], axis=1)

# Save the modified dataframe to a new CSV file
df.to_csv(r"C:\Users\crgon\OneDrive\Desktop\thinkbe_app\ai_side_projects\trading\XAUUSD_modified.csv", index=True)
