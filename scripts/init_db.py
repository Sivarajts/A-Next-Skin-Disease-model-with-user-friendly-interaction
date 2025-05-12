import os
import sqlite3
import time

def init_db():
    # Wait a moment to ensure any existing connections are closed
    time.sleep(1)
    
    # Delete existing database files
    db_files = ["skin_disease.db", "appointments.db"]
    for db_file in db_files:
        try:
            if os.path.exists(db_file):
                os.remove(db_file)
                print(f"Deleted {db_file}")
        except Exception as e:
            print(f"Error deleting {db_file}: {e}")
    
    # Create new database
    try:
        conn = sqlite3.connect("skin_disease.db")
        cursor = conn.cursor()
        
        # Create appointments table
        cursor.execute("""
        CREATE TABLE appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create chat history table
        cursor.execute("""
        CREATE TABLE chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT,
            bot_response TEXT,
            prediction_context TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create prediction history table
        cursor.execute("""
        CREATE TABLE prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            predicted_disease TEXT,
            confidence REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        conn.commit()
        print("New database created successfully!")
        
        # Verify tables were created
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("\nCreated tables:", tables)
        
        # Verify schema for each table
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table[0]});")
            columns = cursor.fetchall()
            print(f"\nSchema for table {table[0]}:")
            for col in columns:
                print(col)
        
    except Exception as e:
        print(f"Error creating database: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    init_db() 