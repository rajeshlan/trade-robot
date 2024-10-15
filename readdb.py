import sqlite3

def read_db_file(db_file):
    conn = None  # Initialize conn to None
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # Retrieve and print the names of all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print(f"Tables in the database '{db_file}':")
        for table_name in tables:
            print(f"- {table_name[0]}")
            
            # Retrieve and print the contents of each table
            cursor.execute(f"SELECT * FROM {table_name[0]};")
            rows = cursor.fetchall()
            
            if rows:
                print(f"Contents of '{table_name[0]}':")
                for row in rows:
                    print(row)
            else:
                print(f"The table '{table_name[0]}' is empty.")
                
            print()  # Add an empty line for better readability

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the connection if it was successfully created
        if conn:
            conn.close()

# Use raw string or double backslashes to avoid escape sequences
db_file_path = r'F:\trading\improvised-code-of-the-pdf-GPT-main\test.db'  # Use raw string
read_db_file(db_file_path)
