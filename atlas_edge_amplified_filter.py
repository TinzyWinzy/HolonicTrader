#!/usr/bin/env python3
"""
ATLAS PROFIT FILTER - PHASE 2: EDGE AMPLIFICATION
Enhanced trade filtering with symbol blacklist/whitelist and quality scoring
"""

import json
from datetime import datetime
from atlas_edge_amplification import SymbolPerformanceTracker, ExitStrategyOptimizer

class EdgeAmplifiedProfitFilter:
    """
    Atlas Profit Filter with Edge Amplification
    - Enforces symbol blacklist (hard block on underperformers)
    - Prioritizes whitelist symbols (profitable assets)
    - Quality scoring for unknown symbols
    - Symbol-specific exit strategies
    """
    
    def __init__(self, config_path='atlas_profit_config.json'):
        self.config_path = config_path
        self.load_config()
        
        # Initialize performance tracker
        self.performance_tracker = SymbolPerformanceTracker()
        
        # Initialize exit optimizer
        self.exit_optimizer = ExitStrategyOptimizer(self.performance_tracker)
        
        # Trade statistics
        self.trade_stats = {
            'total_evaluated': 0,
            'passed_filters': 0,
            'rejected_by_blacklist': 0,
            'rejected_by_quality': 0,
            'rejected_by_edge': 0,
            'whitelist_trades': 0,
            'expected_profits': []
        }
    
    def load_config(self):
        """Load profit configuration"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = {
                'symbol_blacklist': ['XAUT/USD', 'ADA/USD'],
                'symbol_whitelist': ['IMX/USD', 'LTC/USD', 'XTZ/USD', 'LDO/USD', 'XRP/USD'],
                'edge_amplification': {
                    'enabled': True,
                    'blacklist_enforcement': 'HARD',
                    'whitelist_priority': True,
                    'min_quality_score': 40
                },
                'minimum_trade_size_usd': 25.0,
                'minimum_edge_pct': 0.008
            }
    
    def evaluate_trade(self, signal_data, market_data, portfolio_state):
        """
        Evaluate trade with edge amplification
        
        Returns: (approved: bool, reason: str, metadata: dict)
        """
        self.trade_stats['total_evaluated'] += 1
        
        symbol = signal_data.get('symbol', 'UNKNOWN')
        direction = signal_data.get('direction', 'BUY').upper()
        
        # ================================================================
        # PHASE 2: EDGE AMPLIFICATION FILTERS
        # ================================================================
        
        # 1. BLACKLIST CHECK (Hard Block)
        if self.performance_tracker.is_blacklisted(symbol):
            self.trade_stats['rejected_by_blacklist'] += 1
            return False, f'BLACKLISTED_SYMBOL_{symbol}', {}
        
        # Also check config blacklist
        if symbol in self.config.get('symbol_blacklist', []):
            self.trade_stats['rejected_by_blacklist'] += 1
            return False, f'CONFIG_BLACKLIST_{symbol}', {}
        
        # 2. QUALITY SCORE CHECK
        quality_score = self.performance_tracker.get_symbol_quality_score(symbol)
        min_quality = self.config.get('edge_amplification', {}).get('min_quality_score', 40)
        
        # Watchlist symbols get neutral score (50) if no data yet
        is_watchlisted = symbol in self.config.get('symbol_watchlist', [])
        if is_watchlisted and quality_score < 40:
            quality_score = 50  # Neutral score for watchlist
        
        if quality_score < min_quality:
            self.trade_stats['rejected_by_quality'] += 1
            return False, f'LOW_QUALITY_SCORE_{quality_score}', {}
        
        # 3. WHITELIST PRIORITY (Boost approval chances)
        is_whitelisted = self.performance_tracker.is_whitelisted(symbol) or \
                        symbol in self.config.get('symbol_whitelist', [])
        
        if is_whitelisted:
            self.trade_stats['whitelist_trades'] += 1
            # Whitelisted symbols get relaxed edge requirements
            min_edge_pct = self.config.get('minimum_edge_pct', 0.008) * 0.7  # 30% reduction
        else:
            min_edge_pct = self.config.get('minimum_edge_pct', 0.008)
        
        # ================================================================
        # STANDARD ATLAS PROFIT FILTERS
        # ================================================================
        
        # 4. STRATEGY ISOLATION (BUY only during isolation phase)
        if self.config.get('profit_phase') == 'ISOLATION' or self.config.get('profit_phase') == 'NANO_ISOLATION':
            if direction != 'BUY':
                return False, 'SELL_DISABLED_IN_ISOLATION_PHASE', {}
        
        # 5. SIGNAL STRENGTH FILTER
        signal_strength = signal_data.get('strength', 0.0)
        signal_threshold = self.config.get('trade_filters', {}).get('signal_strength_threshold', 0.65)
        
        # Whitelisted symbols get slightly lower threshold
        if is_whitelisted:
            signal_threshold *= 0.9  # 10% reduction
        
        if signal_strength < signal_threshold:
            return False, f'SIGNAL_TOO_WEAK_{signal_strength:.2f}', {}
        
        # 6. MARKET CONDITION FILTERS
        market_checks = [
            ('volatility', self._check_volatility, market_data),
            ('spread', self._check_spread, market_data),
            ('liquidity', self._check_liquidity, market_data)
        ]
        
        for check_name, check_func, data in market_checks:
            passed, reason = check_func(data)
            if not passed:
                return False, reason, {}
        
        # 7. PROFITABILITY CALCULATION
        expected_profit_pct = self._calculate_expected_profit(signal_data, market_data)
        execution_costs_pct = self._estimate_execution_costs(market_data)
        net_profit_pct = expected_profit_pct - execution_costs_pct
        
        if net_profit_pct < min_edge_pct:
            self.trade_stats['rejected_by_edge'] += 1
            return False, f'INSUFFICIENT_NET_EDGE_{net_profit_pct:.4f}', {}
        
        # 8. POSITION SIZING
        position_size = self._calculate_optimal_position_size(
            net_profit_pct, portfolio_state, is_whitelisted
        )
        
        min_trade_size = self.config.get('minimum_trade_size_usd', 25.0)
        if position_size < min_trade_size:
            return False, f'POSITION_TOO_SMALL_{position_size:.1f}', {}
        
        # 9. CAPITAL ALLOCATION CHECK
        if not self._check_capital_allocation(position_size, portfolio_state):
            return False, 'EXCEEDS_CAPITAL_ALLOCATION', {}
        
        # ================================================================
        # ALL CHECKS PASSED - APPROVE TRADE
        # ================================================================
        self.trade_stats['passed_filters'] += 1
        self.trade_stats['expected_profits'].append(net_profit_pct)
        
        # Get symbol-specific exit strategy
        exit_strategy = self.exit_optimizer.get_optimal_exit_strategy(symbol, direction)
        
        metadata = {
            'expected_profit_pct': net_profit_pct,
            'execution_costs_pct': execution_costs_pct,
            'position_size_usd': position_size,
            'quality_score': quality_score,
            'is_whitelisted': is_whitelisted,
            'signal_strength': signal_strength,
            'exit_strategy': exit_strategy,
            'filter_timestamp': datetime.now().isoformat(),
            'atlas_phase': self.config.get('profit_phase', 'NANO_ISOLATION')
        }
        
        return True, 'ATLAS_EDGE_AMPLIFIED_APPROVED', metadata
    
    def _check_volatility(self, market_data):
        """Check sufficient volatility for profit"""
        volatility = market_data.get('volatility_pct', 0)
        threshold = self.config.get('trade_filters', {}).get('volatility_threshold_pct', 0.005)
        
        if volatility < threshold:
            return False, f'INSUFFICIENT_VOLATILITY_{volatility:.4f}'
        return True, ''
    
    def _check_spread(self, market_data):
        """Check spread is within profitable range"""
        spread = market_data.get('spread_pct', 0)
        max_spread = self.config.get('trade_filters', {}).get('spread_max_pct', 0.001)
        
        if spread > max_spread:
            return False, f'SPREAD_TOO_WIDE_{spread:.4f}'
        return True, ''
    
    def _check_liquidity(self, market_data):
        """Check sufficient liquidity"""
        liquidity = market_data.get('liquidity_score', 1.0)
        min_liquidity = self.config.get('trade_filters', {}).get('liquidity_min_score', 0.7)
        
        if liquidity < min_liquidity:
            return False, f'INSUFFICIENT_LIQUIDITY_{liquidity:.2f}'
        return True, ''
    
    def _calculate_expected_profit(self, signal_data, market_data):
        """Calculate expected profit percentage"""
        signal_strength = signal_data.get('strength', 0.5)
        base_profit = signal_strength * 0.02  # 0-2% range
        
        volatility = market_data.get('volatility_pct', 0.01)
        volatility_multiplier = 1.0 + (volatility / 0.01)
        
        regime_score = market_data.get('regime_score', 0)
        regime_multiplier = 1.0 + abs(regime_score)
        
        return base_profit * volatility_multiplier * regime_multiplier
    
    def _estimate_execution_costs(self, market_data):
        """Estimate total execution costs"""
        spread_cost = market_data.get('spread_pct', 0.001)
        fee_cost = 0.002  # 0.2% trading fee (entry + exit)
        slippage_cost = market_data.get('slippage_estimate', 0.0005)
        
        return spread_cost + fee_cost + slippage_cost
    
    def _calculate_optimal_position_size(self, net_profit_pct, portfolio_state, is_whitelisted=False):
        """Calculate position size with whitelist bonus"""
        account_size = portfolio_state.get('account_balance', 90.0)
        available_margin = portfolio_state.get('available_margin', 45.0)
        win_rate = portfolio_state.get('win_rate', 0.6)
        win_loss_ratio = portfolio_state.get('win_loss_ratio', 1.5)
        
        # Base position: 25-30% of available margin
        base_pct = 0.25
        if is_whitelisted:
            base_pct = 0.30  # Whitelisted symbols get 20% more
        
        position = available_margin * base_pct
        
        # Kelly criterion adjustment
        if win_loss_ratio > 0 and win_rate > 0:
            kelly = (win_loss_ratio * win_rate - (1 - win_rate)) / win_loss_ratio
            kelly_position = account_size * max(0, kelly * 0.25)  # 25% of Kelly
            position = min(position, kelly_position)
        
        # Apply limits
        min_size = self.config.get('minimum_trade_size_usd', 25.0)
        max_size_pct = self.config.get('capital_allocation', {}).get('max_position_size_pct', 0.3)
        max_size = available_margin * max_size_pct
        
        return max(min_size, min(position, max_size))
    
    def _check_capital_allocation(self, position_size, portfolio_state):
        """Check if trade fits within capital allocation"""
        available_margin = portfolio_state.get('available_margin', 45.0)
        max_position_pct = self.config.get('capital_allocation', {}).get('max_position_size_pct', 0.3)
        max_allowed = available_margin * max_position_pct
        
        return position_size <= max_allowed
    
    def record_trade_result(self, symbol, pnl, pnl_percent, direction, exit_reason):
        """Record trade result for performance tracking"""
        self.performance_tracker.record_trade(symbol, pnl, pnl_percent, direction, exit_reason)
    
    def get_statistics(self):
        """Get filter performance statistics"""
        stats = self.trade_stats.copy()
        
        if stats['total_evaluated'] > 0:
            stats['pass_rate'] = stats['passed_filters'] / stats['total_evaluated']
        
        if stats['expected_profits']:
            stats['avg_expected_profit'] = sum(stats['expected_profits']) / len(stats['expected_profits'])
        
        # Add performance tracker stats
        stats['symbol_performance'] = self.performance_tracker.performance_db.get('symbol_performance', {})
        stats['blacklist'] = list(self.performance_tracker.blacklist)
        stats['whitelist'] = list(self.performance_tracker.whitelist)
        
        return stats
    
    def get_performance_report(self):
        """Generate comprehensive performance report"""
        return self.performance_tracker.get_performance_report()


# Backward compatibility wrapper
class ProfitOptimizedFilter(EdgeAmplifiedProfitFilter):
    """Legacy class name for backward compatibility"""
    pass


if __name__ == "__main__":
    print("ATLAS PROFIT FILTER - PHASE 2: EDGE AMPLIFICATION")
    print("=" * 60)
    
    # Initialize filter
    filter = EdgeAmplifiedProfitFilter()
    
    print(f"\nConfiguration Loaded:")
    print(f"  Blacklist: {filter.config.get('symbol_blacklist', [])}")
    print(f"  Whitelist: {filter.config.get('symbol_whitelist', [])}")
    print(f"  Edge Amplification: {filter.config.get('edge_amplification', {}).get('enabled', False)}")
    
    # Test with blacklisted symbol
    print("\n" + "=" * 60)
    print("TEST: Blacklisted Symbol (Should Reject)")
    test_signal = {
        'direction': 'BUY',
        'strength': 0.85,
        'symbol': 'XAUT/USD'  # Blacklisted
    }
    
    approved, reason, metadata = filter.evaluate_trade(
        test_signal,
        {'volatility_pct': 0.01, 'spread_pct': 0.0005, 'liquidity_score': 0.8},
        {'account_balance': 90.0, 'available_margin': 45.0, 'win_rate': 0.6, 'win_loss_ratio': 1.5}
    )
    
    print(f"Approved: {approved}")
    print(f"Reason: {reason}")
    
    # Test with whitelisted symbol
    print("\n" + "=" * 60)
    print("TEST: Whitelisted Symbol (Should Approve)")
    test_signal = {
        'direction': 'BUY',
        'strength': 0.75,
        'symbol': 'IMX/USD'  # Whitelisted
    }
    
    approved, reason, metadata = filter.evaluate_trade(
        test_signal,
        {'volatility_pct': 0.01, 'spread_pct': 0.0005, 'liquidity_score': 0.8},
        {'account_balance': 90.0, 'available_margin': 45.0, 'win_rate': 0.6, 'win_loss_ratio': 1.5}
    )
    
    print(f"Approved: {approved}")
    print(f"Reason: {reason}")
    if approved:
        print(f"Quality Score: {metadata.get('quality_score', 'N/A')}")
        print(f"Is Whitelisted: {metadata.get('is_whitelisted', False)}")
        print(f"Exit Strategy: {metadata.get('exit_strategy', {})}")
    
    # Test with unknown symbol
    print("\n" + "=" * 60)
    print("TEST: Unknown Symbol (Quality-Based Decision)")
    test_signal = {
        'direction': 'BUY',
        'strength': 0.70,
        'symbol': 'UNKNOWN/USD'
    }
    
    approved, reason, metadata = filter.evaluate_trade(
        test_signal,
        {'volatility_pct': 0.01, 'spread_pct': 0.0005, 'liquidity_score': 0.8},
        {'account_balance': 90.0, 'available_margin': 45.0, 'win_rate': 0.6, 'win_loss_ratio': 1.5}
    )
    
    print(f"Approved: {approved}")
    print(f"Reason: {reason}")
    
    print("\n" + "=" * 60)
    print("Edge Amplification Filter Ready")
