import os
import psutil

def check_connections():
    # Get current process
    current_process = psutil.Process()
    
    # Get all open files
    open_files = current_process.open_files()
    
    # Check for database files
    db_files = ["skin_disease.db", "appointments.db"]
    for file in open_files:
        if any(db in file.path for db in db_files):
            print(f"Found open database file: {file.path}")
    
    # Check for Python processes
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if 'python' in proc.info['name'].lower():
                print(f"Found Python process: {proc.info}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

if __name__ == "__main__":
    check_connections() 