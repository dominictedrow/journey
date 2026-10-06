import sqlite3

def copy_and_modify_tables(src_db_path, dest_db_path, target_timeframe):
    with sqlite3.connect(dest_db_path) as dest_conn:
        dest_cursor = dest_conn.cursor()

        # Attach the source database
        dest_cursor.execute(f"ATTACH DATABASE ? AS src_db", (src_db_path,))

        # Fetch all table names
        dest_cursor.execute("SELECT name FROM src_db.sqlite_master WHERE type='table';")
        tables = dest_cursor.fetchall()

        for table_name, in tables:
            try:
                # Get column names excluding any 'Target' columns
                dest_cursor.execute(f"PRAGMA src_db.table_info(\"{table_name}\")")
                columns_info = dest_cursor.fetchall()
                columns = [col[1] for col in columns_info if not col[1].startswith('Target')]
                columns_str = ', '.join([f'"{col}"' for col in columns])

                # Create the new table in the destination database
                dest_cursor.execute(f"""
                CREATE TABLE "{table_name}" AS
                SELECT {columns_str}, NULL AS "Target1", NULL AS "Target2", NULL AS "Target3"
                FROM src_db."{table_name}";
                """)

                # Populate the target columns with data from the next row
                dest_cursor.execute(f"""
                UPDATE "{table_name}"
                SET
                    "Target1" = (SELECT "{target_timeframe}_High_0" FROM "{table_name}" as next_row WHERE next_row.rowid = "{table_name}".rowid + 1),
                    "Target2" = (SELECT "{target_timeframe}_Low_0" FROM "{table_name}" as next_row WHERE next_row.rowid = "{table_name}".rowid + 1),
                    "Target3" = (SELECT "{target_timeframe}_Close_0" FROM "{table_name}" as next_row WHERE next_row.rowid = "{table_name}".rowid + 1)
                WHERE EXISTS (
                    SELECT 1 FROM "{table_name}" as next_row WHERE next_row.rowid = "{table_name}".rowid + 1
                );
                """)
                # Commit after updates
                dest_conn.commit()

                print(f"Table {table_name} modified and copied to destination with new target columns based on timeframe {target_timeframe}")
            except sqlite3.Error as e:
                print(f"Error processing table {table_name}: {e}")

        # Check and print the data types of each column
        for table_name, in tables:
            dest_cursor.execute(f"PRAGMA table_info(\"{table_name}\")")
            info = dest_cursor.fetchall()
            print(f"Data types in table {table_name}:")
            for column in info:
                print(f"Column {column[1]} has data type {column[2]}")

        # Detach the source database
        dest_cursor.execute("DETACH DATABASE src_db")

# User input for the target timeframe
target_timeframe = input("Enter the target timeframe (e.g., '15m'): ")

# Database paths
src_db_path = "C:\\Users\\crgon\\OneDrive\\Desktop\\trading\\data\\updated_multi_5m_High_Train.db"
dest_db_path = "C:\\Users\\crgon\\OneDrive\\Desktop\\trading\\data\\multi_5m_High_Train_1.db"

# Copy and modify tables in the destination database
copy_and_modify_tables(src_db_path, dest_db_path, target_timeframe)
