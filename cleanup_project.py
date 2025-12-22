"""
HolonicTrader Project Cleanup Script
Removes obsolete files based on dependency analysis
"""

import os
import shutil

# Files to DELETE (obsolete/redundant)
FILES_TO_DELETE = [
    # Old entry points
    'main_simulation.py',
    'main_backtest.py',
    'main_live.py',
    'nexus.py',
    'nexus_live.py',
    'main_micro.py',
    
    # Obsolete agents
    'agent_sensor.py',
    'agent_rl.py',
    
    # One-time optimization scripts
    'optimize_nexus.py',
    'calibrate_entropy.py',
    'analyze_pareto.py',
    'compare_compounding.py',
    'benchmark_assets.py',
    'tune_dqn.py',
    
    # One-time tests
    'test_micro.py',
    'test_predator.py',
    'verify_strategy.py',
   
    # Training/fetching (already complete)
    'train_lstm.py',
    'fetch_history.py',
    'fetch_multi.py',
    
    # Utilities
    'read_whitepaper.py',
    
    # Generated result files
    'backtest_result.png',
    'compounding_vs_fixed.png',
    'entropy_analysis.png',
    'micro_results.png',
    'pareto_analysis.png',
    'results_ADAUSDT.png',
    'results_DOGEUSDT.png',
    'results_SUIUSDT.png',
    'results_XRPUSDT.png',
    
    # Result CSVs/JSONs
    'benchmark_results.csv',
    'best_params.json',
    'brain_memory.json',
    'portfolio_history_compounding.csv',
    'q_table.json',
    
    # Logs
    'paper_trading.log',
]

# Files to REVIEW before deleting
FILES_TO_REVIEW = [
    'agent_dqn.py',  # DQN agent - are you using?
    'dqn_model.keras',  # DQN model
    'dashboard_gui.py',  # GUI - do you want?
    'database_manager.py',  # Check if used by executor
    'test_entropy.py',  # Old tests
    'test_executor.py',
    'test_observer.py',
]

def cleanup_project(dry_run=True):
    """
    Clean up obsolete files from the project.
    
    Args:
        dry_run: If True, only print what would be deleted. If False, actually delete.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("="*60)
    print("HOLONICTRADER PROJECT CLEANUP")
    print("="*60)
    print(f"\nMode: {'DRY RUN (preview only)' if dry_run else 'LIVE DELETION'}")
    print(f"Directory: {base_dir}\n")
    
    deleted_count = 0
    deleted_size = 0
    missing_count = 0
    
    print("\nFiles to DELETE:")
    print("-" * 60)
    
    for filename in FILES_TO_DELETE:
        filepath = os.path.join(base_dir, filename)
        
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            deleted_size += size
            
            if dry_run:
                print(f"  [WOULD DELETE] {filename} ({size:,} bytes)")
            else:
                try:
                    os.remove(filepath)
                    print(f"  [DELETED] {filename} ({size:,} bytes)")
                    deleted_count += 1
                except Exception as e:
                    print(f"  [ERROR] {filename}: {e}")
        else:
            missing_count += 1
            if dry_run:
                print(f"  [MISSING] {filename}")
    
    print(f"\n{'Would delete' if dry_run else 'Deleted'}: {len(FILES_TO_DELETE) - missing_count} files")
    print(f"Total size: {deleted_size / 1024:.1f} KB")
    
    if FILES_TO_REVIEW:
        print("\n\nFiles to REVIEW (not auto-deleted):")
        print("-" * 60)
        for filename in FILES_TO_REVIEW:
            filepath = os.path.join(base_dir, filename)
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                print(f"  {filename} ({size:,} bytes)")
    
    print("\n" + "="*60)
    
    if dry_run:
        print("\nThis was a DRY RUN. No files were deleted.")
        print("To actually delete files, run:")
        print("  python cleanup_project.py --execute")
    else:
        print(f"\nCleanup complete! {deleted_count} files removed.")
    
    print("="*60)

if __name__ == "__main__":
    import sys
    
    # Check if user wants to actually delete
    execute = '--execute' in sys.argv or '-e' in sys.argv
    
    if execute:
        response = input("\n⚠️  WARNING: This will PERMANENTLY DELETE files. Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Cleanup cancelled.")
            sys.exit(0)
    
    cleanup_project(dry_run=not execute)
