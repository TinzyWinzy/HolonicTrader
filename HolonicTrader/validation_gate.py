import logging
import time
import json
import os
from typing import Dict, Any, Tuple
from sandbox.playground import Playground
from sandbox.strategies.evo_strategy import EvoStrategy

logger = logging.getLogger("ValidationGate")

class LiveValidationGate:
    def __init__(self, paper_period_hours: int = 48):
        self.paper_period_hours = paper_period_hours
        # FIX #3 2026-03-02: STRICT MODE - Block losing genomes
        # Old settings allowed -3% PnL and 5% win rate to pass
        self.min_trades = 5  # Was 2 - require more statistical significance
        self.max_drawdown = 0.30  # Was 0.40 - tighter drawdown control
        self.min_win_rate = 0.40  # NEW: Require 40%+ win rate
        self.min_pnl = 0.0  # NEW: Require POSITIVE PnL (was -0.03)
        self.validation_log = 'validation_history.json'

        
    def validate_genome(self, genome: Dict[str, Any], symbol: str, external_df = None) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Run a simulated 48-hour trial on recent data (Paper Trading Proxy).
        Note: In a true live system, this would happen in real-time. 
        In this local setup, we use the last 48 hours of market data.
        """
        logger.info(f"🛡️ STARTING VALIDATION TRIAL for {symbol}...")
        
        # 1. Setup Playground for "Paper Trial"
        arena = Playground(symbol=symbol, initial_capital=1000.0, verbose=False)
        
        # Check for External Data Injection (Speed Boost & Reliability Fix)
        if external_df is not None and not external_df.empty:
            arena.df = external_df.copy()
        else:
            # Fallback to Disk Load
            arena.load_data(limit=1000)
            
        if arena.df is None or arena.df.empty:
            return False, "Data Unavailable", {}
            
        # Slice for the Trial Period (approx 48h)
        # If 15m candles, 48h = 48 * 4 = 192 candles
        candles_needed = self.paper_period_hours * 4 
        
        # --- DATA AVAILABILITY CHECK ---
        if len(arena.df) < (candles_needed * 0.5): # Require at least 50% (24h)
             return False, f"Insufficient Data ({len(arena.df)} < {candles_needed*0.5})", {}
             
        trial_df = arena.df.tail(candles_needed).copy()
        arena.df = trial_df
        
        # 2. Run Trial
        arena.inject_strategy(EvoStrategy(genome))
        arena.run()
        
        # 3. Assessment
        stats = self._calculate_stats(arena)

        # FIX #3: STRICT VALIDATION - All checks must pass
        checks = {
            'sufficient_trades': stats['trade_count'] >= self.min_trades,
            'controlled_drawdown': stats['max_drawdown'] <= self.max_drawdown,
            'positive_pnl': stats['pnl_pct'] >= self.min_pnl,  # STRICT: Must be >= 0
            'decent_win_rate': stats['win_rate'] >= self.min_win_rate,  # STRICT: Must be >= 40%
            'sanity': self._params_sanity(genome)
        }
        
        is_promoted = all(checks.values())
        reason = "Validation Passed" if is_promoted else f"Failed: {[k for k, v in checks.items() if not v]}"
        
        result = {
            'timestamp': time.time(),
            'symbol': symbol,
            'genome': genome,
            'stats': stats,
            'checks': checks,
            'promoted': is_promoted
        }
        
        self._log_result(result)
        return is_promoted, reason, result

    def _calculate_stats(self, arena: Playground) -> Dict[str, Any]:
        equity_final = arena.equity_curve[-1]['equity'] if arena.equity_curve else 1000.0
        pnl_pct = (equity_final - 1000.0) / 1000.0
        
        # Max DD
        peak = 1000.0
        mdd = 0.0
        for pt in arena.equity_curve:
            if pt['equity'] > peak: peak = pt['equity']
            dd = (peak - pt['equity']) / peak
            if dd > mdd: mdd = dd
            
        return {
            'trade_count': len([t for t in arena.trades if t['type'] == 'SELL']),
            'pnl_pct': float(pnl_pct),
            'max_drawdown': float(mdd),
            'win_rate': len([t for t in arena.trades if t.get('pnl', 0) > 0]) / len(arena.trades) if arena.trades else 0
        }

    def _params_sanity(self, genome: Dict[str, Any]) -> bool:
        # Activation should be less than 5x take profit
        if genome['trailing_activation'] * genome['stop_loss'] > 5 * genome['take_profit']:
            return False
        return True

    def _log_result(self, result: Dict[str, Any]):
        history = []
        if os.path.exists(self.validation_log):
            try:
                with open(self.validation_log, 'r') as f:
                    history = json.load(f)
            except: pass
            
        history.append(result)
        with open(self.validation_log, 'w') as f:
            json.dump(history[-100:], f, indent=4) # Keep last 100
