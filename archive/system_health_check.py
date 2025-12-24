"""
System Health Check Script for HolonicTrader V2
Verifies database integrity, log analysis, and system status.
"""

import sqlite3
import os
from datetime import datetime

def check_database():
    """Check database integrity and stats."""
    db_path = "holonic_trader.db"
    
    print("=" * 60)
    print("DATABASE HEALTH CHECK")
    print("=" * 60)
    
    if not os.path.exists(db_path):
        print("❌ Database file not found!")
        return False
    
    # File size
    size_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"✓ Database Size: {size_mb:.2f} MB")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check integrity
        cursor.execute("PRAGMA integrity_check;")
        integrity = cursor.fetchone()[0]
        if integrity == "ok":
            print("✓ Integrity Check: PASSED")
        else:
            print(f"❌ Integrity Check: {integrity}")
            return False
        
        # Get table list
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"✓ Tables Found: {len(tables)}")
        for table in tables:
            print(f"  - {table[0]}")
        
        # Count records in key tables
        key_tables = ['trades', 'market_data', 'portfolio_snapshots', 'ledger']
        for table_name in key_tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                count = cursor.fetchone()[0]
                print(f"  {table_name}: {count} records")
            except sqlite3.OperationalError:
                print(f"  {table_name}: Table not found")
        
        conn.close()
        print()
        return True
        
    except Exception as e:
        print(f"❌ Database Error: {e}")
        return False

def analyze_logs():
    """Analyze recent log files for errors."""
    print("=" * 60)
    print("LOG ANALYSIS")
    print("=" * 60)
    
    log_files = [f for f in os.listdir('.') if f.startswith('live_trading_session_') and f.endswith('.log')]
    log_files.sort(reverse=True)
    
    if not log_files:
        print("❌ No log files found!")
        return
    
    print(f"✓ Found {len(log_files)} log files")
    print(f"  Latest: {log_files[0]}")
    
    # Analyze latest log
    latest_log = log_files[0]
    error_count = 0
    warning_count = 0
    exception_count = 0
    
    try:
        with open(latest_log, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            line_lower = line.lower()
            if 'error' in line_lower:
                error_count += 1
            if 'warning' in line_lower:
                warning_count += 1
            if 'exception' in line_lower or 'traceback' in line_lower:
                exception_count += 1
        
        print(f"\n  Analysis of {latest_log}:")
        print(f"  - Total Lines: {len(lines)}")
        print(f"  - Errors: {error_count}")
        print(f"  - Warnings: {warning_count}")
        print(f"  - Exceptions: {exception_count}")
        
        if exception_count > 0:
            print("\n  ⚠️  Exceptions detected! Review log for details.")
        elif error_count > 0:
            print("\n  ⚠️  Errors detected! Review log for details.")
        else:
            print("\n  ✓ No critical issues detected.")
        
    except Exception as e:
        print(f"❌ Error reading log: {e}")
    
    print()

def check_models():
    """Check ML model files."""
    print("=" * 60)
    print("ML MODELS CHECK")
    print("=" * 60)
    
    models = {
        'dqn_model.keras': 'DQN Policy Network',
        'lstm_model.keras': 'LSTM Predictor',
        'scaler.pkl': 'Feature Scaler'
    }
    
    for filename, description in models.items():
        if os.path.exists(filename):
            size_kb = os.path.getsize(filename) / 1024
            print(f"✓ {description}: {size_kb:.1f} KB")
        else:
            print(f"❌ {description}: NOT FOUND")
    
    print()

def main():
    print("\n" + "=" * 60)
    print("HOLONICTRADER V2 - SYSTEM HEALTH CHECK")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()
    
    db_ok = check_database()
    analyze_logs()
    check_models()
    
    print("=" * 60)
    if db_ok:
        print("OVERALL STATUS: ✓ HEALTHY")
    else:
        print("OVERALL STATUS: ⚠️  NEEDS ATTENTION")
    print("=" * 60)

if __name__ == "__main__":
    main()
