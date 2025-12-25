from database_manager import DatabaseManager

print("Initializing Database Manager...")
try:
    db = DatabaseManager()
    print("Database Initialized Successfully. Tables should be created.")
except Exception as e:
    print(f"Error initializing database: {e}")
