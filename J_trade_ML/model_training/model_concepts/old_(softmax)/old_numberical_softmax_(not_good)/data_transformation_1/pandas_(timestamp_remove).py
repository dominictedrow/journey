import pandas as pd
import sqlite3
from tqdm import tqdm

# Connect to the SQLite database
sqlite_path = 'C:\\Users\\crgon\\OneDrive\\Desktop\\thinkbe_app\\ai_side_projects\\trading\\stock_data.db'
conn = sqlite3.connect(sqlite_path)
c = conn.cursor()

# Select columns whose names are in the form "Timestamp_x" where x is in the range [1, 20]
timestamp_columns = [f"Timestamp_{i}" for i in range(1, 21)]

# Get the total columns of the table
query = "PRAGMA table_info(stock_data)"
total_columns = pd.read_sql_query(query, conn)['name'].tolist()

# Create new column list without the first 7 columns
new_columns = total_columns[7:]

# Chunk size for loading and updating
chunk_size = 50000

# Create new table without first 7 columns
c.execute(f"CREATE TABLE new_stock_data AS SELECT {', '.join(new_columns)} FROM stock_data LIMIT 0")

# Query the total number of rows in the table
query = "SELECT COUNT(*) FROM stock_data"
total_rows = pd.read_sql_query(query, conn).iloc[0, 0]

# Create a progress bar for rows
progress_bar_rows = tqdm(total=total_rows, desc="Processing Rows")

# Iterate over the rows in chunks
for offset in range(0, total_rows, chunk_size):
    # Load only the timestamp columns along with ROWID
    query = f"SELECT ROWID, {', '.join(new_columns)} FROM stock_data LIMIT {chunk_size} OFFSET {offset}"
    chunk = pd.read_sql_query(query, conn)

    # Remove date from Timestamp and keep only the hour
    for column in timestamp_columns:
        if column in chunk.columns:
            chunk[column] = pd.to_datetime(chunk[column], errors='coerce').dt.hour

    # Append data to the new table
    chunk.to_sql('new_stock_data', conn, if_exists='append', index=False)

    progress_bar_rows.update(len(chunk))  # Update the row progress bar

# Drop the original table
c.execute("DROP TABLE stock_data")

# Rename the new table to the original table
c.execute("ALTER TABLE new_stock_data RENAME TO stock_data")

# Commit the transaction
conn.commit()

# Close the row progress bar
progress_bar_rows.close()

# Close the database connection
conn.close()
