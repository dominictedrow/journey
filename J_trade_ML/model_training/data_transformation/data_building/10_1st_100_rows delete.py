import sqlite3

# Path to the SQLite database
db_path = r"C:\Users\crgon\OneDrive\Desktop\trading\data\new_multi_stacked_XAUUSD_Train_4.db"

# Connect to the SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get the list of all tables in the database
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

# Iterate over each table and delete the first 100 rows
for table_name in tables:
    table_name = table_name[0]  # Extract the table name from the tuple
    
    # Use a Common Table Expression (CTE) to assign a row number and delete the first 100 rows
    cursor.execute(f"""
        DELETE FROM {table_name}
        WHERE ROWID IN (
            SELECT ROWID FROM {table_name}
            LIMIT 100
        );
    """)
    
    print(f"Deleted first 100 rows from table {table_name}")

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Done!")
