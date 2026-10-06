import sqlite3

def delete_blank_target_rows(db_path):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Fetch all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for table_name, in tables:
            try:
                # Check if the table has 'Target1', 'Target2', 'Target3', and 'Group' columns
                cursor.execute(f"PRAGMA table_info(\"{table_name}\")")
                columns_info = cursor.fetchall()
                column_names = [col[1] for col in columns_info]
                
                required_columns = {'Target1', 'Target2', 'Target3', 'Group'}
                if not required_columns.issubset(column_names):
                    print(f"Table {table_name} does not contain the required columns. Skipping.")
                    continue

                # Retrieve all distinct groups
                distinct_groups = cursor.execute(f'SELECT DISTINCT "Group" FROM "{table_name}"').fetchall()

                # Iterate over each group
                for group, in distinct_groups:
                    # Find the last row for the current group where Target columns are NULL
                    row_to_delete = cursor.execute(f'''
                        SELECT rowid
                        FROM "{table_name}"
                        WHERE "Group" = ? AND "Target1" IS NULL AND "Target2" IS NULL AND "Target3" IS NULL
                        ORDER BY rowid DESC
                        LIMIT 1
                    ''', (group,)).fetchone()

                    if row_to_delete:
                        # Delete the row if found
                        cursor.execute(f'DELETE FROM "{table_name}" WHERE rowid = ?', (row_to_delete[0],))
                        print(f"Deleted row {row_to_delete[0]} from table {table_name} for group {group}")

                # Commit the changes after processing each table
                conn.commit()

            except sqlite3.Error as e:
                print(f"Error processing table {table_name}: {e}")

# Database path
db_path = "C:\\Users\\crgon\\OneDrive\\Desktop\\trading\\data\\multi_stacked_HLC_Train.db"

# Delete blank target rows in the database
delete_blank_target_rows(db_path)
