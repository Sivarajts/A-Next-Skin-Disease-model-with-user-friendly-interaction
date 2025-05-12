import os

db_files = ["skin_disease.db", "appointments.db"]

for db_file in db_files:
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
            print(f"Database file {db_file} deleted successfully.")
        except Exception as e:
            print(f"Error deleting database {db_file}: {e}")
    else:
        print(f"Database file {db_file} does not exist.") 