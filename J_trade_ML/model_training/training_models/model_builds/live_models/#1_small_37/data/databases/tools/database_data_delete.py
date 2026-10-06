import sqlite3

# Path to your database
db_path = r"C:\Users\crgon\OneDrive\Desktop\trading\metatrader\dwxconnect-main\python\data\databases\Processed_multi_stacked_1.db"

def delete_data_and_columns(db_path):
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get the list of all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    for table in tables:
        table_name = table[0]

        # Delete all data from the table
        cursor.execute(f'DELETE FROM "{table_name}";')

        # Get the column names
        cursor.execute(f'PRAGMA table_info("{table_name}");')
        columns = cursor.fetchall()

        # If there are more than 4 columns, drop the last 4
        if len(columns) > 4:
            columns_to_keep = [f'"{col[1]}"' for col in columns[:-4]]
            columns_to_keep_str = ", ".join(columns_to_keep)

            # Create a temporary table with the columns to keep
            cursor.execute(f'CREATE TEMPORARY TABLE "{table_name}_backup" AS SELECT {columns_to_keep_str} FROM "{table_name}";')

            # Drop the original table
            cursor.execute(f'DROP TABLE "{table_name}";')

            # Recreate the original table without the last 4 columns
            cursor.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM "{table_name}_backup";')

            # Drop the temporary table
            cursor.execute(f'DROP TABLE "{table_name}_backup";')

    # Commit changes and close the connection
    conn.commit()
    conn.close()

# Call the function
delete_data_and_columns(db_path)