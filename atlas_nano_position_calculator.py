#!/usr/bin/env python3
"""
ATLAS NANO ACCOUNT - POSITION SIZE CALCULATOR
Calculates optimal position size for $90-100 accounts with leverage
"""

import json

class NanoPositionCalculator:
    """
    Position sizing for nano accounts (<$100)
    Uses leverage to meet exchange minimums while managing risk
    """
    
    def __init__(self, account_balance, config_path='atlas_profit_config.json'):
        self.account_balance = account_balance
        self.load_config(config_path)
        
        # Exchange minimum order sizes (approximate)
        self.exchange_minimums = {
            'BTC/USD': 5.0,    # $5 minimum
            'ETH/USD': 10.0,   # $10 minimum
            'SOL/USD': 5.0,    # $5 minimum
            'DEFAULT': 10.0    # $10 default minimum
        }
    
    def load_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except:
            self.config = {
                'minimum_trade_size_usd': 25.0,
                'leverage_settings': {
                    'default_leverage': 3.0,
                    'max_leverage': 5.0
                },
                'capital_allocation': {
                    'max_position_size_pct': 0.30
                }
            }
    
    def calculate_position(self, symbol, signal_strength, available_margin):
        """
        Calculate position size with leverage for nano account
        
        Returns: {
            'margin_usd': margin to use,
            'leverage': leverage multiplier,
            'notional_value': total position value,
            'quantity': asset quantity to buy,
            'meets_minimum': bool (meets exchange minimum)
        }
        """
        # 1. Determine leverage based on signal strength
        if signal_strength >= 0.8:
            leverage = self.config['leverage_settings']['leverage_by_confidence']['high']
        elif signal_strength >= 0.65:
            leverage = self.config['leverage_settings']['leverage_by_confidence']['medium']
        else:
            leverage = self.config['leverage_settings']['leverage_by_confidence']['low']
        
        # Cap leverage
        leverage = min(leverage, self.config['leverage_settings']['max_leverage'])
        
        # 2. Calculate margin to use (conservative: 25-30% of available)
        max_position_pct = self.config['capital_allocation']['max_position_size_pct']
        margin_to_use = available_margin * max_position_pct
        
        # 3. Calculate notional value (margin * leverage)
        notional_value = margin_to_use * leverage
        
        # 4. Apply minimum trade size
        min_trade = self.config['minimum_trade_size_usd']
        if notional_value < min_trade:
            # Increase leverage to meet minimum (if possible)
            required_leverage = min_trade / margin_to_use
            if required_leverage <= self.config['leverage_settings']['max_leverage']:
                leverage = required_leverage
                notional_value = margin_to_use * leverage
            else:
                # Can't meet minimum with available margin
                return {
                    'margin_usd': 0,
                    'leverage': 0,
                    'notional_value': 0,
                    'quantity': 0,
                    'meets_minimum': False,
                    'reason': f'Insufficient margin: need ${min_trade/leverage:.2f} for ${min_trade} minimum'
                }
        
        # 5. Check exchange minimum
        exchange_min = self.exchange_minimums.get(symbol, self.exchange_minimums['DEFAULT'])
        meets_exchange_min = notional_value >= exchange_min
        
        return {
            'margin_usd': round(margin_to_use, 2),
            'leverage': round(leverage, 2),
            'notional_value': round(notional_value, 2),
            'quantity': 0,  # Will be calculated by executor with actual price
            'meets_minimum': meets_exchange_min,
            'exchange_minimum': exchange_min
        }
    
    def get_recommended_trade_size(self, available_margin):
        """Get recommended trade size for current margin"""
        if available_margin < self.config['nano_specific_rules']['min_margin_for_trade']:
            return {
                'can_trade': False,
                'reason': f'Margin too low: ${available_margin:.2f} < ${self.config["nano_specific_rules"]["min_margin_for_trade"]:.2f}',
                'action': 'ACCUMULATE_MORE_CAPITAL'
            }
        
        # Calculate with medium confidence (0.65-0.8)
        test_position = self.calculate_position('BTC/USD', 0.7, available_margin)
        
        return {
            'can_trade': test_position['meets_minimum'],
            'recommended_margin': test_position['margin_usd'],
            'recommended_leverage': test_position['leverage'],
            'expected_notional': test_position['notional_value'],
            'account_balance': self.account_balance
        }


# Example usage
if __name__ == "__main__":
    print("ATLAS Nano Position Calculator")
    print("=" * 50)
    
    # Test with $90 account, $45 available margin
    calculator = NanoPositionCalculator(account_balance=90.0)
    
    print(f"\nAccount Balance: $90.00")
    print(f"Available Margin: $45.00")
    
    result = calculator.get_recommended_trade_size(45.0)
    print(f"\nCan Trade: {result['can_trade']}")
    if result['can_trade']:
        print(f"Recommended Margin: ${result['recommended_margin']:.2f}")
        print(f"Leverage: {result['recommended_leverage']}x")
        print(f"Expected Notional: ${result['expected_notional']:.2f}")
    
    # Test specific position
    print(f"\n--- Position Calculation Example ---")
    position = calculator.calculate_position('BTC/USD', 0.75, 45.0)
    print(f"Margin: ${position['margin_usd']:.2f}")
    print(f"Leverage: {position['leverage']}x")
    print(f"Notional: ${position['notional_value']:.2f}")
    print(f"Meets Minimum: {position['meets_minimum']}")
