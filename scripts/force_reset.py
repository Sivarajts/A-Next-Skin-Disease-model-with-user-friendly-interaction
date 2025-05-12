import os
import psutil
import time
import sqlite3

def force_reset():
    # Kill all Python processes
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if 'python' in proc.info['name'].lower():
                psutil.Process(proc.info['pid']).kill()
                print(f"Killed process {proc.info['pid']}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    # Wait for processes to close
    time.sleep(2)
    
    # Delete all database-related files
    files_to_delete = [
        'skin_disease.db',
        'skin_disease.db-journal',
        'skin_disease.db-wal',
        'skin_disease.db-shm',
        'appointments.db',
        'appointments.db-journal',
        'appointments.db-wal',
        'appointments.db-shm'
    ]
    
    for file in files_to_delete:
        try:
            if os.path.exists(file):
                os.remove(file)
                print(f"Deleted {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")
    
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
        
    except Exception as e:
        print(f"Error creating database: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    force_reset() 