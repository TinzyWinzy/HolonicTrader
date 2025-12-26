from HolonicTrader.agent_diagnostic import DiagnosticHolon
from database_manager import DatabaseManager
import sys

def main():
    print("==========================================")
    print("   HOLONIC TRADER - SYSTEM DIAGNOSTICS    ")
    print("==========================================")
    
    try:
        db = DatabaseManager()
        diagnostic = DiagnosticHolon()
        result = diagnostic.run_system_check(db)
        
        if result:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        print(f"CRITICAL ERROR IN DIAGNOSTICS: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
