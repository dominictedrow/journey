import pandas as pd
import numpy as np
import sqlite3

# Ask for target timeframes
target_timeframes = input("What are the target timeframes? (e.g., '1m,5m,15m,30m,60m'): ").split(',')

# Convert input to integers and remove 'm'
target_timeframes = [int(tf.strip('m')) for tf in target_timeframes]

# Determine if the 'Minute' column should be included
include_minutes = min(target_timeframes) < 60

# Connect to the SQLite database
conn = sqlite3.connect('C:\\Users\\crgon\\OneDrive\\Desktop\\trading\\data\\multi_stacked_High_Train.db')

# Function to process each table
def process_table(table_name):
    # Load the data into a DataFrame from the table
    df = pd.read_sql(f'SELECT * FROM {table_name}', conn)

    # Convert the 'DateTime' to datetime type if not already
    if not pd.api.types.is_datetime64_any_dtype(df['DateTime']):
        df['DateTime'] = pd.to_datetime(df['DateTime'])

    # Reset index to make 'DateTime' a column for processing if it's not already a column
    if 'DateTime' not in df.columns:
        df.reset_index(inplace=True)

    # Extract 'Day', 'Month', 'Hour', and 'Minute'
    df.insert(0, 'Minute', df['DateTime'].dt.minute)
    df.insert(0, 'Hour', df['DateTime'].dt.hour)
    df.insert(0, 'Month', df['DateTime'].dt.month)
    df.insert(0, 'Day', df['DateTime'].dt.day)

    # Remove the 'DateTime' column
    df.drop('DateTime', axis=1, inplace=True)

    # Adjust the 'Minute' column based on the timeframe
    for timeframe in target_timeframes:
        mask = df['Group'] == timeframe
        if timeframe >= 60:
            df.loc[mask, 'Minute'] = (df.loc[mask, 'Minute'] // timeframe) * timeframe

    # Save the modified DataFrame back to the SQLite database, replacing the original table
    df.to_sql(table_name, conn, if_exists='replace', index=False)

# Get the list of all tables in the database
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
table_names = tables['name'].tolist()

# Process each table
for table_name in table_names:
    process_table(table_name)

# Close the connection to the database
conn.close()

print("Data processing complete and all tables have been updated.")
