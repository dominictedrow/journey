import pandas as pd
import sqlite3

# Ask for target timeframe
target_timeframe = input("What is the target timeframe? (e.g., '60m'): ")

# Determine if the 'Minute' column should be included
include_minutes = True
if target_timeframe.endswith('m'):
    if int(target_timeframe[:-1]) >= 60:
        include_minutes = False

# Connect to the SQLite database
conn = sqlite3.connect('C:\\Users\\crgon\\OneDrive\\Desktop\\trading\\data\\multi_High_Train.db')

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

    # Extract 'Day', 'Month', 'Hour', and potentially 'Minute'
    df.insert(0, 'Hour', df['DateTime'].dt.hour)
    df.insert(0, 'Month', df['DateTime'].dt.month)
    df.insert(0, 'Day', df['DateTime'].dt.day)
    if include_minutes:
        df.insert(0, 'Minute', df['DateTime'].dt.minute)

    # Remove the 'DateTime' column
    df.drop('DateTime', axis=1, inplace=True)

    # If not including minutes, check if the 'Minute' column exists before trying to remove it
    if not include_minutes and 'Minute' in df.columns:
        df.drop('Minute', axis=1, inplace=True)

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
