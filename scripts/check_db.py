import sqlite3

def check_db():
    try:
        # Connect to the database
        conn = sqlite3.connect("skin_disease.db")
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("Tables in database:", tables)
        
        # Get schema for each table
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table[0]});")
            columns = cursor.fetchall()
            print(f"\nSchema for table {table[0]}:")
            for col in columns:
                print(col)
        
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_db() 