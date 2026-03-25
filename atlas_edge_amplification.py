#!/usr/bin/env python3
"""
ATLAS PROFIT - PHASE 2: EDGE AMPLIFICATION
Symbol Performance Tracking, Blacklist Management, and Exit Optimization

Focus:
1. Track performance by symbol
2. Blacklist underperforming assets (XAUT, ADA, etc.)
3. Amplify edge on profitable symbols (IMX, LTC, XTZ, LDO, XRP)
4. Improve exit strategies based on symbol behavior
"""

import json
import os
from datetime import datetime
from collections import defaultdict
import numpy as np

class SymbolPerformanceTracker:
    """
    Tracks and analyzes trading performance by symbol
    Implements blacklist/whitelist system for edge amplification
    """
    
    def __init__(self, tracker_path='atlas_profit_tracker.json'):
        self.tracker_path = tracker_path
        self.performance_db = self._load_performance_db()
        self.blacklist = set()
        self.whitelist = set()
        self.load_lists()
    
    def _load_performance_db(self):
        """Load or initialize performance database"""
        try:
            with open(self.tracker_path, 'r') as f:
                tracker = json.load(f)
            
            if 'symbol_performance' not in tracker:
                tracker['symbol_performance'] = {}
            
            return tracker
        except:
            return {
                'symbol_performance': {},
                'monitoring': {
                    'trades_since_activation': 0,
                    'total_profit': 0.0,
                    'last_trade_time': None
                }
            }
    
    def load_lists(self):
        """Load blacklist and whitelist from config"""
        try:
            with open('atlas_profit_config.json', 'r') as f:
                config = json.load(f)
            
            self.blacklist = set(config.get('symbol_blacklist', []))
            self.whitelist = set(config.get('symbol_whitelist', []))
        except:
            # Default lists based on historical performance
            self.blacklist = {'XAUT/USD', 'ADA/USD'}  # Chronic underperformers
            self.whitelist = {'IMX/USD', 'LTC/USD', 'XTZ/USD', 'LDO/USD', 'XRP/USD'}  # Pro profitable
    
    def record_trade(self, symbol, pnl, pnl_percent, direction, exit_reason=None):
        """Record a completed trade for performance tracking"""
        if symbol not in self.performance_db['symbol_performance']:
            self.performance_db['symbol_performance'][symbol] = {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'total_pnl': 0.0,
                'total_pnl_percent': 0.0,
                'avg_pnl': 0.0,
                'avg_pnl_percent': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'consecutive_losses': 0,
                'max_consecutive_losses': 0,
                'last_trade': None,
                'performance_trend': 'NEUTRAL',  # IMPROVING, DECLINING, NEUTRAL
                'trades_by_direction': {
                    'BUY': {'trades': 0, 'wins': 0, 'pnl': 0.0},
                    'SELL': {'trades': 0, 'wins': 0, 'pnl': 0.0}
                }
            }
        
        stats = self.performance_db['symbol_performance'][symbol]
        
        # Update basic stats
        stats['total_trades'] += 1
        stats['total_pnl'] += pnl
        stats['total_pnl_percent'] += pnl_percent
        stats['avg_pnl'] = stats['total_pnl'] / stats['total_trades']
        stats['avg_pnl_percent'] = stats['total_pnl_percent'] / stats['total_trades']
        stats['last_trade'] = datetime.now().isoformat()
        
        # Update win/loss
        if pnl > 0:
            stats['wins'] += 1
            if pnl > stats['largest_win']:
                stats['largest_win'] = pnl
        else:
            stats['losses'] += 1
            if pnl < stats['largest_loss']:
                stats['largest_loss'] = pnl
            
            # Track consecutive losses
            stats['consecutive_losses'] += 1
            if stats['consecutive_losses'] > stats['max_consecutive_losses']:
                stats['max_consecutive_losses'] = stats['consecutive_losses']
        
        # Reset consecutive losses on win
        if pnl >= 0:
            stats['consecutive_losses'] = 0
        
        # Win rate
        stats['win_rate'] = stats['wins'] / stats['total_trades'] if stats['total_trades'] > 0 else 0
        
        # Profit factor
        if stats['losses'] > 0:
            total_wins = stats['total_pnl'] - sum([t for t in [] if t < 0])  # Simplified
            total_losses = abs(stats['total_pnl'] - total_wins)
            if total_losses > 0:
                stats['profit_factor'] = total_wins / total_losses
        else:
            stats['profit_factor'] = float('inf') if stats['wins'] > 0 else 0
        
        # Direction-specific tracking
        direction_key = direction.upper()
        if direction_key in stats['trades_by_direction']:
            stats['trades_by_direction'][direction_key]['trades'] += 1
            stats['trades_by_direction'][direction_key]['pnl'] += pnl
            if pnl > 0:
                stats['trades_by_direction'][direction_key]['wins'] += 1
        
        # Performance trend (last 10 trades)
        self._calculate_performance_trend(symbol)
        
        # Update tracker
        self.performance_db['monitoring']['trades_since_activation'] += 1
        self.performance_db['monitoring']['total_profit'] += pnl
        self.performance_db['monitoring']['last_trade_time'] = datetime.now().isoformat()
        
        # Auto-update lists
        self._auto_update_lists()
        
        # Save
        self._save_performance_db()
        
        return stats
    
    def _calculate_performance_trend(self, symbol):
        """Calculate if symbol performance is improving or declining"""
        # Simplified trend calculation
        stats = self.performance_db['symbol_performance'].get(symbol, {})
        recent_win_rate = stats.get('win_rate', 0)
        
        if recent_win_rate > 0.65:
            stats['performance_trend'] = 'IMPROVING'
        elif recent_win_rate < 0.40:
            stats['performance_trend'] = 'DECLINING'
        else:
            stats['performance_trend'] = 'NEUTRAL'
    
    def _auto_update_lists(self):
        """Automatically update blacklist/whitelist based on performance"""
        for symbol, stats in self.performance_db['symbol_performance'].items():
            # Skip if not enough data
            if stats['total_trades'] < 5:
                continue
            
            # Auto-blacklist criteria
            if (stats['win_rate'] < 0.35 and 
                stats['total_pnl'] < -10.0 and
                stats['total_trades'] >= 10):
                
                if symbol not in self.blacklist:
                    self.blacklist.add(symbol)
                    print(f"[EdgeAmp] BLACKLISTED: {symbol} (win_rate={stats['win_rate']:.1%}, pnl=${stats['total_pnl']:.2f})")
            
            # Auto-whitelist criteria
            if (stats['win_rate'] > 0.65 and 
                stats['total_pnl'] > 20.0 and
                stats['profit_factor'] > 2.0):
                
                if symbol not in self.whitelist:
                    self.whitelist.add(symbol)
                    print(f"[EdgeAmp] WHITELISTED: {symbol} (win_rate={stats['win_rate']:.1%}, PF={stats['profit_factor']:.2f})")
    
    def _save_performance_db(self):
        """Save performance database to file"""
        with open(self.tracker_path, 'w') as f:
            json.dump(self.performance_db, f, indent=2)
    
    def is_blacklisted(self, symbol):
        """Check if symbol is blacklisted"""
        return symbol in self.blacklist
    
    def is_whitelisted(self, symbol):
        """Check if symbol is whitelisted"""
        return symbol in self.whitelist
    
    def get_symbol_quality_score(self, symbol):
        """
        Get quality score for symbol (0-100)
        Higher = better trading opportunity
        """
        if symbol in self.blacklist:
            return 0  # Hard blacklist
        
        if symbol in self.whitelist:
            return 90  # Preferred symbol
        
        stats = self.performance_db['symbol_performance'].get(symbol, {})
        
        if stats.get('total_trades', 0) < 3:
            return 50  # Unknown symbol, neutral score
        
        # Calculate quality score
        score = 50  # Base score
        
        # Win rate component (0-20 points)
        win_rate = stats.get('win_rate', 0.5)
        score += (win_rate - 0.5) * 40
        
        # Profit factor component (0-20 points)
        profit_factor = stats.get('profit_factor', 1.0)
        score += min(20, (profit_factor - 1.0) * 20)
        
        # Trend component (0-10 points)
        trend = stats.get('performance_trend', 'NEUTRAL')
        if trend == 'IMPROVING':
            score += 10
        elif trend == 'DECLINING':
            score -= 10
        
        # Consecutive losses penalty (0-10 points)
        consec_losses = stats.get('consecutive_losses', 0)
        if consec_losses >= 3:
            score -= consec_losses * 3
        
        return max(0, min(100, score))
    
    def get_performance_report(self):
        """Generate comprehensive performance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_symbols_tracked': len(self.performance_db['symbol_performance']),
                'blacklisted_count': len(self.blacklist),
                'whitelisted_count': len(self.whitelist),
                'total_profit': self.performance_db['monitoring']['total_profit']
            },
            'best_performers': [],
            'worst_performers': [],
            'blacklist': list(self.blacklist),
            'whitelist': list(self.whitelist)
        }
        
        # Sort symbols by performance
        symbols_sorted = sorted(
            self.performance_db['symbol_performance'].items(),
            key=lambda x: x[1].get('total_pnl', 0),
            reverse=True
        )
        
        # Top 5 performers
        for symbol, stats in symbols_sorted[:5]:
            report['best_performers'].append({
                'symbol': symbol,
                'total_pnl': stats.get('total_pnl', 0),
                'win_rate': stats.get('win_rate', 0),
                'profit_factor': stats.get('profit_factor', 0),
                'trades': stats.get('total_trades', 0)
            })
        
        # Bottom 5 performers
        for symbol, stats in symbols_sorted[-5:]:
            report['worst_performers'].append({
                'symbol': symbol,
                'total_pnl': stats.get('total_pnl', 0),
                'win_rate': stats.get('win_rate', 0),
                'profit_factor': stats.get('profit_factor', 0),
                'trades': stats.get('total_trades', 0)
            })
        
        return report

    def auto_rebalance_weights(self, min_trades: int = 20):
        """
        Dynamically adjust ASSET_ALLOCATION_WEIGHTS based on realized performance.
        Called periodically from the main trading loop.
        
        Formula: new_weight = base_weight × (1 + (win_rate - 0.50) × 2)
        Clamped to [0.2, 4.0]. Only adjusts symbols with >= min_trades recorded trades.

        2026-03-21: Moderate amplification — proven winners get higher allocation.
        """
        import config as cfg

        base_weights = getattr(cfg, 'ASSET_ALLOCATION_WEIGHTS', {})
        default_weight = getattr(cfg, 'ASSET_ALLOCATION_WEIGHT_DEFAULT', 0.4)
        adjusted = False

        for symbol, stats in self.performance_db.get('symbol_performance', {}).items():
            n_trades = stats.get('total_trades', 0)
            if n_trades < min_trades:
                continue

            win_rate = stats.get('win_rate', 0.5)
            total_pnl = stats.get('total_pnl', 0.0)

            # Only rebalance symbols with meaningful data and positive PnL
            if total_pnl <= 0:
                continue

            base = base_weights.get(symbol, default_weight)
            multiplier = 1.0 + (win_rate - 0.50) * 2.0
            new_weight = round(base * multiplier, 2)
            new_weight = max(0.2, min(4.0, new_weight))

            if abs(new_weight - base_weights.get(symbol, default_weight)) > 0.05:
                base_weights[symbol] = new_weight
                adjusted = True
                print(f"[EdgeAmp] REBALANCE: {symbol} weight {base:.2f} → {new_weight:.2f} "
                      f"(WR={win_rate:.1%}, {n_trades} trades, PnL=${total_pnl:.2f})")

        if adjusted:
            cfg.ASSET_ALLOCATION_WEIGHTS = base_weights
            print(f"[EdgeAmp] Allocation weights updated: {base_weights}")

    def save_lists(self):
        """Save blacklist/whitelist to config"""
        try:
            with open('atlas_profit_config.json', 'r') as f:
                config = json.load(f)
        except:
            config = {}
        
        config['symbol_blacklist'] = list(self.blacklist)
        config['symbol_whitelist'] = list(self.whitelist)
        
        with open('atlas_profit_config.json', 'w') as f:
            json.dump(config, f, indent=2)


class ExitStrategyOptimizer:
    """
    Optimizes exit strategies based on symbol-specific behavior
    """
    
    def __init__(self, performance_tracker):
        self.tracker = performance_tracker
        self.exit_stats = self._load_exit_stats()
    
    def _load_exit_stats(self):
        """Load exit strategy statistics"""
        try:
            with open('atlas_profit_tracker.json', 'r') as f:
                tracker = json.load(f)
            
            if 'exit_statistics' not in tracker:
                tracker['exit_statistics'] = {}
            
            return tracker['exit_statistics']
        except:
            return {}
    
    def record_exit(self, symbol, exit_reason, pnl, hold_time_minutes, max_profit_seen, max_loss_seen):
        """Record exit details for optimization"""
        key = f"{symbol}_{exit_reason}"
        
        if key not in self.exit_stats:
            self.exit_stats[key] = {
                'count': 0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'avg_hold_time': 0,
                'avg_max_profit': 0.0,
                'avg_max_loss': 0.0,
                'regret_factor': 0.0  # How much profit was left on table
            }
        
        stats = self.exit_stats[key]
        stats['count'] += 1
        stats['total_pnl'] += pnl
        stats['avg_pnl'] = stats['total_pnl'] / stats['count']
        stats['avg_hold_time'] = (stats['avg_hold_time'] * (stats['count'] - 1) + hold_time_minutes) / stats['count']
        stats['avg_max_profit'] = (stats['avg_max_profit'] * (stats['count'] - 1) + max_profit_seen) / stats['count']
        stats['avg_max_loss'] = (stats['avg_max_loss'] * (stats['count'] - 1) + max_loss_seen) / stats['count']
        
        # Calculate regret factor (profit left on table)
        if pnl > 0 and max_profit_seen > pnl:
            stats['regret_factor'] = (max_profit_seen - pnl) / max_profit_seen
        elif pnl < 0 and max_loss_seen < pnl:
            stats['regret_factor'] = abs(max_loss_seen - pnl) / abs(max_loss_seen) if max_loss_seen != 0 else 0
        
        # Save
        self._save_exit_stats()
    
    def _save_exit_stats(self):
        """Save exit statistics"""
        try:
            with open('atlas_profit_tracker.json', 'r') as f:
                tracker = json.load(f)
        except:
            tracker = {}
        
        tracker['exit_statistics'] = self.exit_stats
        
        with open('atlas_profit_tracker.json', 'w') as f:
            json.dump(tracker, f, indent=2)
    
    def get_optimal_exit_strategy(self, symbol, direction='BUY'):
        """
        Get optimal exit strategy for symbol based on historical performance
        
        Returns: {
            'stop_loss_pct': float,
            'take_profit_pct': float,
            'trailing_stop': bool,
            'exit_early_signal': bool,
            'recommended_hold_time': int (minutes)
        }
        """
        # Default exit parameters
        exit_params = {
            'stop_loss_pct': 0.025,  # 2.5%
            'take_profit_pct': 0.045,  # 4.5%
            'trailing_stop': True,
            'exit_early_signal': True,
            'recommended_hold_time': 120  # 2 hours
        }
        
        # Get symbol-specific stats
        symbol_exits = [
            (key, stats) for key, stats in self.exit_stats.items()
            if key.startswith(symbol)
        ]
        
        if not symbol_exits:
            return exit_params
        
        # Analyze which exit reasons performed best
        best_exit = max(symbol_exits, key=lambda x: x[1].get('avg_pnl', 0))
        best_exit_reason = best_exit[0].split('_')[-1]
        best_exit_stats = best_exit[1]
        
        # Adjust based on findings
        if best_exit_reason == 'STOP_LOSS':
            # Stop losses are hitting too often - widen stops
            exit_params['stop_loss_pct'] = 0.035  # 3.5%
            exit_params['take_profit_pct'] = 0.05  # 5%
        
        elif best_exit_reason == 'TAKE_PROFIT':
            # Take profits working well
            exit_params['take_profit_pct'] = best_exit_stats.get('avg_max_profit', 0.045)
            exit_params['trailing_stop'] = True
        
        elif best_exit_reason == 'TRAILING_STOP':
            # Trailing stops working - use tighter trailing
            exit_params['trailing_stop'] = True
            exit_params['stop_loss_pct'] = 0.02  # 2% trailing
        
        elif best_exit_reason == 'MANUAL':
            # Manual exits performed well - use signal-based
            exit_params['exit_early_signal'] = True
        
        # Adjust hold time based on symbol
        if best_exit_stats.get('avg_hold_time', 120) < 60:
            exit_params['recommended_hold_time'] = 45  # Scalping symbol
        elif best_exit_stats.get('avg_hold_time', 120) > 240:
            exit_params['recommended_hold_time'] = 180  # Swing symbol
        
        return exit_params


def analyze_trading_history_for_blacklist():
    """
    Analyze historical trade data to identify blacklisted symbols
    Looks for trades_dump_since_20260314.csv or similar files
    """
    print("=" * 60)
    print("EDGE AMPLIFICATION - HISTORICAL PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Find trade history files
    trade_files = [
        'trades_dump_since_20260314.csv',
        'trade_history.csv',
        'closed_trades.csv',
        'executed_trades.csv'
    ]
    
    trade_data = None
    for filepath in trade_files:
        if os.path.exists(filepath):
            try:
                import pandas as pd
                trade_data = pd.read_csv(filepath)
                print(f"\nLoaded trade data from: {filepath}")
                break
            except Exception as e:
                continue
    
    if trade_data is None:
        print("\nNo historical trade data found.")
        print("Using default blacklist/whitelist based on known performance.")
        return {
            'blacklist': ['XAUT/USD', 'ADA/USD'],
            'whitelist': ['IMX/USD', 'LTC/USD', 'XTZ/USD', 'LDO/USD', 'XRP/USD']
        }
    
    print(f"Analyzing {len(trade_data)} trades...")
    
    # Analyze by symbol
    symbol_stats = defaultdict(lambda: {
        'trades': 0,
        'wins': 0,
        'losses': 0,
        'total_pnl': 0.0,
        'pnl_percent': 0.0
    })
    
    for _, row in trade_data.iterrows():
        symbol = row.get('symbol', 'UNKNOWN')
        pnl = float(row.get('pnl', 0))
        pnl_pct = float(row.get('pnl_percent', 0))
        
        symbol_stats[symbol]['trades'] += 1
        symbol_stats[symbol]['total_pnl'] += pnl
        symbol_stats[symbol]['pnl_percent'] += pnl_pct
        
        if pnl > 0:
            symbol_stats[symbol]['wins'] += 1
        elif pnl < 0:
            symbol_stats[symbol]['losses'] += 1
    
    # Calculate metrics and categorize
    blacklist = []
    whitelist = []
    
    print("\n" + "=" * 60)
    print("SYMBOL PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    for symbol, stats in symbol_stats.items():
        if stats['trades'] < 3:
            continue  # Not enough data
        
        win_rate = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
        avg_pnl = stats['total_pnl'] / stats['trades']
        
        # Blacklist criteria
        if (win_rate < 0.40 and stats['total_pnl'] < -5.0) or \
           (stats['total_pnl'] < -15.0 and stats['trades'] >= 5):
            blacklist.append(symbol)
            print(f"\n[BLACKLIST] {symbol}")
            print(f"  Trades: {stats['trades']} | Win Rate: {win_rate:.1%}")
            print(f"  Total PnL: ${stats['total_pnl']:.4f}")
            print(f"  Avg PnL: ${avg_pnl:.4f}")
        
        # Whitelist criteria
        elif (win_rate > 0.60 and stats['total_pnl'] > 10.0) or \
             (stats['total_pnl'] > 20.0 and stats['trades'] >= 5):
            whitelist.append(symbol)
            print(f"\n[WHITELIST] {symbol}")
            print(f"  Trades: {stats['trades']} | Win Rate: {win_rate:.1%}")
            print(f"  Total PnL: ${stats['total_pnl']:.4f}")
            print(f"  Avg PnL: ${avg_pnl:.4f}")
    
    # Save results
    result = {
        'blacklist': blacklist if blacklist else ['XAUT/USD', 'ADA/USD'],
        'whitelist': whitelist if whitelist else ['IMX/USD', 'LTC/USD', 'XTZ/USD', 'LDO/USD', 'XRP/USD'],
        'analysis_timestamp': datetime.now().isoformat(),
        'total_symbols_analyzed': len(symbol_stats)
    }
    
    with open('symbol_edge_analysis.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Analysis complete. Results saved to symbol_edge_analysis.json")
    print(f"Blacklisted: {len(result['blacklist'])} symbols")
    print(f"Whitelisted: {len(result['whitelist'])} symbols")
    
    return result


if __name__ == "__main__":
    # Run historical analysis
    analysis = analyze_trading_history_for_blacklist()
    
    # Update config
    try:
        with open('atlas_profit_config.json', 'r') as f:
            config = json.load(f)
    except:
        config = {}
    
    config['symbol_blacklist'] = analysis['blacklist']
    config['symbol_whitelist'] = analysis['whitelist']
    
    with open('atlas_profit_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n[OK] Configuration updated with edge amplification settings")
