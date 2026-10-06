import sqlite3
import os

# define the path
path = r"C:\Users\crgon\OneDrive\Desktop\thinkbe_app\ai_side_projects\trading"
database_name = "stock_data.db"

# create the full path
full_path = os.path.join(path, database_name)

# connect to the database (this will create it if it doesn't exist)
conn = sqlite3.connect(full_path)

# close the connection
conn.close()
