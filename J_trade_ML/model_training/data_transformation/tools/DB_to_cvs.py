import sqlite3
import pandas as pd

# Path to the database file
db_path = r"C:\\Users\\crgon\\OneDrive\\Desktop\\trading\\data\\new_multi_stacked_XAUUSD_Val_11_new.db"
# Output Excel file path
output_excel_path = r"C:\\Users\\crgon\\OneDrive\\Desktop\\output_data_new_new_new_new.xlsx"

# Group values and corresponding columns for each group
group_columns = {
    5: "5m_Close_0",
    15: "15m_Close_0",
    30: "30m_Close_0",
    60: "60m_Close_0"
}

def main():
    print("Starting the script...")

    # Connect to the SQLite database
    print(f"Connecting to database at {db_path}...")
    conn = sqlite3.connect(db_path)

    # Fetch the first table name dynamically
    query_get_table = "SELECT name FROM sqlite_master WHERE type='table' LIMIT 1;"
    print("Fetching the first table name...")
    table_name = conn.execute(query_get_table).fetchone()
    if not table_name:
        print("No table found in the database.")
        raise ValueError("No table found in the database.")
    table_name = table_name[0]
    print(f"Using table: {table_name}")

    # Query to fetch all data from the first table
    query = f"SELECT * FROM {table_name}"
    print("Querying data from the table...")
    
    # Load the data into a pandas DataFrame
    df = pd.read_sql_query(query, conn)
    print(f"Data loaded. Total rows: {len(df)}")

    # Close the database connection
    print("Closing the database connection...")
    conn.close()

    # Check if required columns exist
    print("Validating required columns...")
    required_columns = ["Group", "Target4"] + list(group_columns.values())
    if not all(col in df.columns for col in required_columns):
        missing_columns = [col for col in required_columns if col not in df.columns]
        print(f"Missing columns: {missing_columns}")
        raise ValueError("One or more required columns are missing from the database.")

    # Create a Pandas Excel writer
    print(f"Creating Excel file at {output_excel_path}...")
    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        for group, close_column in group_columns.items():
            print(f"Processing group: {group}")
            # Filter data for the current group
            group_data = df[df['Group'] == group]
            print(f"Group {group} rows: {len(group_data)}")

            # Select required columns
            group_data = group_data[["Group", close_column, "Target4"]]

            # Rename columns
            group_data.columns = ["Group", close_column, "Target4"]

            # Write to a separate sheet in the Excel file
            sheet_name = f"Group_{group}"
            print(f"Writing data to sheet: {sheet_name}")
            group_data.to_excel(writer, sheet_name=sheet_name, index=False)
    print("Excel file created successfully.")

if __name__ == "__main__":
    main()
