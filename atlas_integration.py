#!/usr/bin/env python3
"""
Atlas Profit Architect - Main Integration Module
Connects Atlas profit system to existing trading infrastructure

PHASE 2: EDGE AMPLIFICATION
- Symbol blacklist/whitelist enforcement
- Quality scoring for symbols
- Improved exit strategies
"""

import json
import time
import logging
from datetime import datetime

# PHASE 2: Use edge-amplified filter
from atlas_edge_amplified_filter import EdgeAmplifiedProfitFilter
from atlas_capital_manager import CapitalEfficiencyManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('atlas_profit.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AtlasProfit')

class AtlasProfitIntegration:
    """
    Main integration class for Atlas Profit Architecture
    
    PHASE 2: Includes edge amplification
    - Symbol blacklist/whitelist enforcement
    - Quality scoring
    - Improved exit strategies
    """

    def __init__(self, config_path='atlas_profit_config.json'):
        self.config_path = config_path
        # PHASE 2: Use edge-amplified filter
        self.profit_filter = EdgeAmplifiedProfitFilter(config_path)
        self.capital_manager = CapitalEfficiencyManager(config_path)
        self.integration_status = {
            'initialized': False,
            'last_trade_time': None,
            'total_trades_processed': 0,
            'profitable_trades': 0,
            'integration_version': '1.1-phase2'
        }

        logger.info("Atlas Profit Integration initialized (PHASE 2: EDGE AMPLIFICATION)")
    
    def initialize_with_account(self, account_balance):
        """Initialize with current account balance"""
        try:
            # Initialize capital management
            allocation = self.capital_manager.initialize_capital(account_balance)
            
            self.integration_status.update({
                'initialized': True,
                'account_balance': account_balance,
                'capital_allocation': allocation,
                'initialization_time': datetime.now().isoformat()
            })
            
            logger.info(f"Atlas initialized with account balance: ${account_balance:.2f}")
            logger.info(f"Capital allocation: BUY=${allocation['buy_strategy']:.2f}, "
                       f"RESERVE=${allocation['reserve']:.2f}")
            
            return True, "ATLAS_INITIALIZED_SUCCESSFULLY"
            
        except Exception as e:
            logger.error(f"Failed to initialize Atlas: {e}")
            return False, f"INITIALIZATION_FAILED: {e}"
    
    def process_trade_signal(self, signal_data, market_data, portfolio_state):
        """
        Main method to process trade signals through Atlas profit filters
        
        Returns: (approved: bool, reason: str, metadata: dict, position_size: float)
        """
        if not self.integration_status['initialized']:
            return False, "ATLAS_NOT_INITIALIZED", {}, 0.0
        
        try:
            # 1. Check daily loss limits
            loss_ok, loss_reason = self.capital_manager.check_daily_loss_limit()
            if not loss_ok:
                logger.warning(f"Trade rejected - Daily loss limit: {loss_reason}")
                return False, loss_reason, {}, 0.0
            
            # 2. Check drawdown limits
            drawdown_ok, drawdown_reason = self.capital_manager.check_drawdown_limit()
            if not drawdown_ok:
                logger.warning(f"Trade rejected - Drawdown limit: {drawdown_reason}")
                return False, drawdown_reason, {}, 0.0
            
            # 3. Apply profit filters
            approved, reason, metadata = self.profit_filter.evaluate_trade(
                signal_data, market_data, portfolio_state
            )
            
            if not approved:
                logger.info(f"Trade rejected by profit filter: {reason}")
                return False, reason, {}, 0.0
            
            # 4. Calculate optimal position size
            position_size = self.capital_manager.calculate_position_size(
                strategy_type=signal_data.get('direction', 'BUY'),
                win_rate=portfolio_state.get('win_rate', 0.6),
                win_loss_ratio=portfolio_state.get('win_loss_ratio', 1.0),
                volatility_pct=market_data.get('volatility_pct', 0.01)
            )
            
            # 5. Update integration status
            self.integration_status['total_trades_processed'] += 1
            self.integration_status['last_trade_time'] = datetime.now().isoformat()
            
            logger.info(f"Trade APPROVED: {reason}")
            logger.info(f"Expected profit: {metadata['expected_profit_pct']:.3f}%")
            logger.info(f"Position size: ${position_size:.2f}")
            
            return True, reason, metadata, position_size
            
        except Exception as e:
            logger.error(f"Error processing trade signal: {e}")
            return False, f"PROCESSING_ERROR: {e}", {}, 0.0
    
    def update_trade_result(self, trade_result):
        """Update Atlas with completed trade results"""
        try:
            # Update capital manager performance
            self.capital_manager.update_performance(trade_result)

            # Track profitable trades
            if trade_result.get('pnl', 0) > 0:
                self.integration_status['profitable_trades'] += 1
            
            # PHASE 2: Record trade in edge amplification tracker
            symbol = trade_result.get('symbol', 'UNKNOWN')
            pnl = trade_result.get('pnl', 0)
            pnl_percent = trade_result.get('pnl_percent', 0)
            direction = trade_result.get('direction', 'BUY')
            exit_reason = trade_result.get('exit_reason', 'UNKNOWN')
            
            # Record in performance tracker
            if hasattr(self.profit_filter, 'record_trade_result'):
                self.profit_filter.record_trade_result(
                    symbol=symbol,
                    pnl=pnl,
                    pnl_percent=pnl_percent,
                    direction=direction,
                    exit_reason=exit_reason
                )

            # Log trade result
            logger.info(f"Trade completed: PnL=${trade_result.get('pnl', 0):.2f}, "
                       f"Symbol: {symbol}, Total trades: {self.integration_status['total_trades_processed']}")

            return True

        except Exception as e:
            logger.error(f"Error updating trade result: {e}")
            return False
    
    def get_performance_report(self):
        """Generate comprehensive performance report"""
        try:
            # Get filter statistics
            filter_stats = self.profit_filter.get_statistics()
            
            # Get capital management report
            capital_report = self.capital_manager.get_capital_allocation_report()
            
            # Calculate overall performance
            total_trades = self.integration_status['total_trades_processed']
            profitable_trades = self.integration_status['profitable_trades']
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            
            report = {
                'integration_status': self.integration_status.copy(),
                'filter_performance': filter_stats,
                'capital_management': capital_report,
                'overall_performance': {
                    'total_trades': total_trades,
                    'profitable_trades': profitable_trades,
                    'win_rate': win_rate,
                    'profitability_status': 'PROFITABLE' if win_rate > 0.6 else 'DEVELOPING'
                },
                'scaling_recommendation': self.capital_manager.should_scale_capital(),
                'timestamp': datetime.now().isoformat(),
                'version': self.integration_status['integration_version']
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}
    
    def should_scale_operations(self):
        """Check if system should scale based on performance"""
        return self.capital_manager.should_scale_capital()
    
    def emergency_stop(self, reason="EMERGENCY_STOP"):
        """Emergency stop trading"""
        logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")
        
        # Update status
        self.integration_status['emergency_stop'] = {
            'activated': True,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        
        return True, reason

# Example usage and integration template
if __name__ == "__main__":
    print("Atlas Profit Integration Module")
    print("=" * 50)
    
    # Initialize Atlas
    atlas = AtlasProfitIntegration()
    
    # Initialize with account balance (replace with actual balance)
    success, message = atlas.initialize_with_account(10000)
    print(f"Initialization: {success} - {message}")
    
    # Example trade processing
    test_signal = {
        'direction': 'BUY',
        'strength': 0.8,
        'symbol': 'BTC/USDT',
        'source': 'momentum_strategy'
    }
    
    test_market = {
        'volatility_pct': 0.012,
        'spread_pct': 0.0008,
        'liquidity_score': 0.85,
        'regime': 'NEUTRAL',
        'regime_score': 0.3,
        'price': 50000
    }
    
    test_portfolio = {
        'account_balance': 10000,
        'win_rate': 0.65,
        'win_loss_ratio': 1.4,
        'current_positions': {}
    }
    
    # Process trade signal
    approved, reason, metadata, position_size = atlas.process_trade_signal(
        test_signal, test_market, test_portfolio
    )
    
    if approved:
        print(f"✓ Trade approved: {reason}")
        print(f"  Position size: ${position_size:.2f}")
        print(f"  Expected profit: {metadata['expected_profit_pct']:.3f}%")
        
        # Simulate trade completion
        trade_result = {
            'pnl': 42.50,  # Example profit
            'symbol': 'BTC/USDT',
            'direction': 'BUY',
            'entry_price': 50000,
            'exit_price': 50250,
            'position_size': position_size,
            'timestamp': datetime.now().isoformat()
        }
        
        atlas.update_trade_result(trade_result)
    else:
        print(f"✗ Trade rejected: {reason}")
    
    # Generate performance report
    report = atlas.get_performance_report()
    print(f"\nPerformance Report:")
    print(f"  Total trades processed: {report['overall_performance']['total_trades']}")
    print(f"  Win rate: {report['overall_performance']['win_rate']:.1%}")
    print(f"  Profitability: {report['overall_performance']['profitability_status']}")
    
    print("\nAtlas integration ready for deployment")
    print("Next: Import and integrate with main trading system")