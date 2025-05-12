import os
import sqlite3
import streamlit as st

def reset_db():
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
        
    except Exception as e:
        print(f"Error creating database: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    reset_db()

    if 'doctors' not in st.session_state:
        st.session_state.doctors = EXPERT_DOCTORS
    if 'disease_info' not in st.session_state:
        st.session_state.disease_info = DISEASE_INFO 