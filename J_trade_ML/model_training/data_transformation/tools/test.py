import sqlite3

def get_feature_names(db_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get the list of tables in the database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    if tables:
        # Use the first table found
        first_table_name = tables[0][0]
        
        # Query to get the column names from the first table
        cursor.execute(f"PRAGMA table_info({first_table_name})")
        columns_info = cursor.fetchall()
        
        # Extract and print the column names (feature names)
        feature_names = [column[1] for column in columns_info]
        print(f"Features in {db_path} (Table: {first_table_name}): {feature_names}")
    else:
        print(f"No tables found in {db_path}")
    
    # Close the connection
    conn.close()

# Paths to the databases
db_paths = [
    r"C:\Users\crgon\OneDrive\Desktop\trading\data\new_multi_stacked_XAUUSD_Train_8_new.db",
]

# Get and print feature names for each database
for db_path in db_paths:
    get_feature_names(db_path)