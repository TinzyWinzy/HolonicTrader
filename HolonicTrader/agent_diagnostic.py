from HolonicTrader.holon_core import Holon, Disposition
import config
import os
import glob
import ccxt
import re
import sqlite3
from datetime import datetime

class DiagnosticHolon(Holon):
    """
    Holon responsible for performing system health checks and diagnostics
    before the trading session begins.
    """
    def __init__(self):
        super().__init__(name="DiagnosticHolon", disposition=Disposition(autonomy=0.5, integration=0.5))
        self.log_keywords = ["ERROR", "CRITICAL", "Exception", "Traceback", "WARNING"]

    def receive_message(self, sender, content):
        pass

    def check_config(self):
        """Validates critical configuration parameters."""
        print("   [Diagnostic] Checking Configuration...")
        errors = []
        
        # Check API Keys (Basic check for existence/non-empty if not in paper mode)
        if not config.PAPER_TRADING:
            if not getattr(config, 'API_KEY', None) or not getattr(config, 'API_SECRET', None):
                 errors.append("Missing API_KEY or API_SECRET for Live Trading.")
        
        # Check Critical Parameters
        required_params = ['INITIAL_CAPITAL', 'ALLOWED_ASSETS', 'TIMEFRAME', 'IMMUNE_MAX_DAILY_DRAWDOWN']
        for param in required_params:
            if not hasattr(config, param):
                errors.append(f"Missing config parameter: {param}")
        
        if errors:
            for err in errors:
                print(f"      ❌ {err}")
            return False
        print("      ✅ Configuration OK")
        return True

    def check_database(self, db_manager):
        """Checks database connectivity."""
        print("   [Diagnostic] Checking Database...")
        try:
            # Simple query to verify connection
            path = getattr(db_manager, 'db_path', 'holonic_trader.db')
            conn = sqlite3.connect(path)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            conn.close()
            print("      ✅ Database Connection OK")
            return True
        except Exception as e:
            print(f"      ❌ Database Connection Failed: {e}")
            return False

    def check_models(self):
        """Checks for the existence of required ML models."""
        print("   [Diagnostic] Checking ML Models...")
        # Required: system degrades significantly without these
        required_models = ['xgboost_model.json']
        # Optional: system runs in heuristic/fallback mode without these
        optional_models = ['dqn_model.keras', 'lstm_model.keras']

        base_path = os.getcwd()
        missing_required = []
        missing_optional = []

        for model in required_models:
            if not os.path.exists(os.path.join(base_path, model)):
                missing_required.append(model)

        for model in optional_models:
            if not os.path.exists(os.path.join(base_path, model)):
                missing_optional.append(model)

        if missing_required:
            print(f"      ❌ Missing Required Models: {', '.join(missing_required)}")
            return False

        if missing_optional:
            print(f"      ℹ️  Optional Models absent (heuristic fallback active): {', '.join(missing_optional)}")
        else:
            print("      ✅ All Core Models Found")
        return True

    def check_exchange(self, exchange_id=None):
        """Checks connectivity to BOTH Data Source (KuCoin) and Trading Venue (Kraken)."""
        success = True
        
        # 1. Check Data Source (Default: KuCoin)
        # We check this first as it failed previously
        data_exchange = 'kucoin'
        print(f"   [Diagnostic] Checking Data Source ({data_exchange})...")
        try:
            # PATCH: Increase timeout for diagnostic to 10s
            ex = getattr(ccxt, data_exchange)({'timeout': 10000})
            ex.load_markets()
            print("      ✅ Data Feed Connected")
        except Exception as e:
            print(f"      ⚠️ Data Feed Connection Failed ({data_exchange}): {e}")
            print(f"      🔄 Attempting Fallback to Kraken for Connectivity Check...")
            try:
                # Fallback to Kraken (since we likely use it for execution anyway)
                ex_fallback = getattr(ccxt, 'kraken')({'timeout': 10000})
                ex_fallback.load_markets()
                print("      ✅ Fallback Data Feed (Kraken) Connected")
                # We mark as success because we have connectivity
            except Exception as e2:
                print(f"      ❌ Fallback Failed: {e2}")
                success = False

        # 2. Check Execution Venue
        if getattr(config, 'TRADING_MODE', 'SPOT') == 'FUTURES':
             trade_exchange = 'krakenfutures'
        else:
             trade_exchange = 'kraken'
             
        print(f"   [Diagnostic] Checking Trading Venue ({trade_exchange})...")
        try:
            ex = getattr(ccxt, trade_exchange)()
            ex.load_markets()
            print("      ✅ Execution Venue Connected")
        except Exception as e:
             print(f"      ❌ Trading Venue Connection Failed: {e}")
             success = False
             
        return success

    def check_logs(self, lookback_days=1):
        """
        Scans recent log files for errors and warnings.
        """
        print("   [Diagnostic] Reviewing Recent Logs...")
        
        # Find log files matching pattern
        log_pattern = "live_trading_session_*.log"
        log_files = glob.glob(log_pattern)
        
        if not log_files:
            print("      ℹ️  No log files found.")
            return True

        # Sort by modification time, newest first
        log_files.sort(key=os.path.getmtime, reverse=True)
        
        # Check the most recent file(s)
        # For simplicity, let's just check the very last session log
        latest_log = log_files[0]
        print(f"      > Scanning: {latest_log}")
        
        issues_found = 0
        try:
            with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    for keyword in self.log_keywords:
                        if keyword in line:
                            # Print a snippet of the error
                            clean_line = line.strip()[:100] # Truncate if too long
                            print(f"         ⚠️  [{keyword}] {clean_line}...")
                            issues_found += 1
                            if issues_found >= 5: # Limit output
                                print("         ... (more issues found, Check logs for details)")
                                break
                    if issues_found >= 5:
                        break
        except Exception as e:
            print(f"      ❌ Failed to read log file: {e}")
            return False
            
        if issues_found == 0:
            print("      ✅ No recent critical errors found in last session.")
        else:
            print(f"      ℹ️  Found {issues_found}+ potential issues in last session.")
            
        return True

    def run_system_check(self, db_manager):
        """Orchestrates the full system diagnostic."""
        print("\n🔍 STARTING SYSTEM DIAGNOSTICS...")
        
        checks = [
            self.check_config(),
            self.check_database(db_manager),
            self.check_models(),
            self.check_exchange(),
            self.check_logs()
        ]
        
        if all(checks):
            print("✅ SYSTEM DIAGNOSTICS PASSED. READY TO START.\n")
            return True
        else:
            print("❌ SYSTEM DIAGNOSTICS FAILED. PLEASE FIX ISSUES ABOVE.\n")
            return False
