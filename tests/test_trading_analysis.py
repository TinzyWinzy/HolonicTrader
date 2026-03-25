"""
test_trading_analysis.py
Test script for the Trading Analysis module.

Usage:
    python test_trading_analysis.py [database_path]
    
    If no database path is provided, defaults to 'holonic_trader.db'
"""

import pytest
import sys
import os

# This module is a standalone CLI script, not a pytest test suite.
# Its test functions require a db_path argument (not a fixture).
pytest.skip("Standalone script — not a pytest test suite", allow_module_level=True)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from HolonicTrader.trading_analysis import (
    ExecutionQualityAnalyzer,
    PositionManagementAnalyzer,
    PerformanceMetricsAnalyzer,
    TradingAnalysisDashboard
)


def test_execution_quality(db_path: str):
    """Test execution quality analysis."""
    print("\n" + "="*60)
    print("TESTING: Execution Quality Analyzer")
    print("="*60 + "\n")
    
    analyzer = ExecutionQualityAnalyzer(db_path)
    try:
        result = analyzer.generate_report()
        print(f"\n✅ Execution Quality Test: {result}")
        return True
    except Exception as e:
        print(f"\n❌ Execution Quality Test Failed: {e}")
        return False


def test_position_management(db_path: str):
    """Test position management analysis."""
    print("\n" + "="*60)
    print("TESTING: Position Management Analyzer")
    print("="*60 + "\n")
    
    analyzer = PositionManagementAnalyzer(db_path)
    try:
        result = analyzer.generate_report()
        print(f"\n✅ Position Management Test: {result}")
        return True
    except Exception as e:
        print(f"\n❌ Position Management Test Failed: {e}")
        return False


def test_performance_metrics(db_path: str):
    """Test performance metrics analysis."""
    print("\n" + "="*60)
    print("TESTING: Performance Metrics Analyzer")
    print("="*60 + "\n")
    
    analyzer = PerformanceMetricsAnalyzer(db_path)
    try:
        result = analyzer.generate_report()
        print(f"\n✅ Performance Metrics Test: {result}")
        return True
    except Exception as e:
        print(f"\n❌ Performance Metrics Test Failed: {e}")
        return False


def test_full_dashboard(db_path: str):
    """Test the unified dashboard."""
    print("\n" + "="*60)
    print("TESTING: Unified Trading Analysis Dashboard")
    print("="*60 + "\n")
    
    dashboard = TradingAnalysisDashboard(db_path)
    try:
        dashboard.generate_report()
        print(f"\n✅ Full Dashboard Test: SUCCESS")
        return True
    except Exception as e:
        print(f"\n❌ Full Dashboard Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    db_path = sys.argv[1] if len(sys.argv) > 1 else "holonic_trader.db"
    
    print(f"\n[bold cyan]Trading Analysis Module Test Suite[/bold cyan]")
    print(f"Database: {db_path}\n")
    
    results = {
        'Execution Quality': test_execution_quality(db_path),
        'Position Management': test_position_management(db_path),
        'Performance Metrics': test_performance_metrics(db_path),
        'Full Dashboard': test_full_dashboard(db_path)
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[bold green]🎉 All tests passed![/bold green]\n")
        return 0
    else:
        print(f"\n[bold yellow]⚠️ {total - passed} test(s) failed[/bold yellow]\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
