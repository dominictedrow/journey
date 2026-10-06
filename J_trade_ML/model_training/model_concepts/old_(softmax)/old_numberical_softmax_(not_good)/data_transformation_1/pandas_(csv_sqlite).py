import pandas as pd
import sqlite3
import os

# define the paths
csv_path = r"C:\Users\crgon\OneDrive\Desktop\thinkbe_app\ai_side_projects\trading\XAUUSD_modified.csv"
database_path = r"C:\Users\crgon\OneDrive\Desktop\thinkbe_app\ai_side_projects\trading\stock_data.db"

# load the data into a pandas DataFrame
df = pd.read_csv(csv_path)

# create a connection to the SQLite database
conn = sqlite3.connect(database_path)

# write the data from the DataFrame into the SQLite database
df.to_sql('stock_data', conn, if_exists='replace', index=False)

# close the connection
conn.close()
