import os

def check_journal():
    # List all files in the current directory
    files = os.listdir('.')
    
    # Look for SQLite journal files
    journal_files = [f for f in files if f.endswith('-journal') or f.endswith('-wal') or f.endswith('-shm')]
    
    if journal_files:
        print("Found SQLite journal files:", journal_files)
        # Try to delete them
        for file in journal_files:
            try:
                os.remove(file)
                print(f"Deleted {file}")
            except Exception as e:
                print(f"Error deleting {file}: {e}")
    else:
        print("No SQLite journal files found.")

if __name__ == "__main__":
    check_journal() 