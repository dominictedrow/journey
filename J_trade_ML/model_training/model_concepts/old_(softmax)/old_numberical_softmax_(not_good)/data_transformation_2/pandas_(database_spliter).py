import pandas as pd
import sqlite3
from tqdm import tqdm

sqlite_path = 'C:\\Users\\crgon\\OneDrive\\Desktop\\thinkbe_app\\ai_side_projects\\trading\\stock_data.db'
test_sqlite_path = 'C:\\Users\\crgon\\OneDrive\\Desktop\\thinkbe_app\\ai_side_projects\\trading\\stock_data_test.db'

# Connect to the original database
conn = sqlite3.connect(sqlite_path)
c = conn.cursor()

# Get the list of all tables
c.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = c.fetchall()

# Connect to the new test database
test_conn = sqlite3.connect(test_sqlite_path)

# For each table, select last 25% of the rows and insert them into the new database
for table_name in tqdm(tables, desc='Processing tables', unit='table'):
    table_name = table_name[0]
    
    # Get the total number of rows in the table
    c.execute(f"SELECT COUNT(*) FROM {table_name}")
    total_rows = c.fetchone()[0]
    
    # Calculate the starting row for the last 25% of the data
    starting_row = int(total_rows * 0.90)
    
    # Select only the last 25% of rows
    df_sample = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT -1 OFFSET {starting_row}", conn)
    
    df_sample.to_sql(table_name, test_conn, if_exists='replace')

# Close the connections
conn.close()
test_conn.close()
