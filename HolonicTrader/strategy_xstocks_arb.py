"""
xStocks Funding Arbitrage Strategy Module
For HolonicTrader - Integrates with ArbitrageHolon

This module provides funding rate arbitrage strategies for xStocks on Kraken Futures.
"""

import ccxt
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# xStocks configuration
XSTOCKS = {
    'SPYX/USD:USD': {'name': 'S&P 500 ETF', 'min_qty': 0.1, 'category': 'ETF'},
    'QQQX/USD:USD': {'name': 'Nasdaq 100 ETF', 'min_qty': 0.1, 'category': 'ETF'},
    'NVDAX/USD:USD': {'name': 'NVIDIA', 'min_qty': 0.5, 'category': 'Tech'},
    'AAPLX/USD:USD': {'name': 'Apple', 'min_qty': 0.5, 'category': 'Tech'},
    'GOOGLX/USD:USD': {'name': 'Alphabet', 'min_qty': 0.5, 'category': 'Tech'},
    'TSLAX/USD:USD': {'name': 'Tesla', 'min_qty': 0.5, 'category': 'Tech'},
    'MSTRX/USD:USD': {'name': 'MicroStrategy', 'min_qty': 0.5, 'category': 'Tech'},
    'CRCLX/USD:USD': {'name': 'Circle', 'min_qty': 1.0, 'category': 'Other'},
    'HOODX/USD:USD': {'name': 'Robinhood', 'min_qty': 1.0, 'category': 'Other'},
}

# Funding rate thresholds for arbitrage entry
FUNDING_THRESHOLD_SHORT = -0.02  # -2% per 8h (earn by shorting)
FUNDING_THRESHOLD_LONG = 0.02   # +2% per 8h (earn by longing)

# Position limits (as % of Open Interest)
MAX_POSITION_OI_PCT = 0.30  # Max 30% of OI to avoid market impact


class XStocksArbitrage:
    """
    xStocks Funding Arbitrage Strategy
    
    Strategies:
    1. Pure Funding Arb - Market neutral, earn funding
    2. Basis Trade - Long/Short xStock vs underlying
    3. Cross-Asset Hedge - xStock vs correlated crypto
    """
    
    def __init__(self, exchange=None):
        self.exchange = exchange or ccxt.krakenfutures({'enableRateLimit': True})
        self.last_funding_check = 0
        self.funding_cache = {}
        
    def fetch_xstocks_funding(self) -> Dict[str, Dict]:
        """Fetch funding rates for all xStocks"""
        now = time.time()
        
        # Cache for 5 minutes
        if now - self.last_funding_check < 300 and self.funding_cache:
            return self.funding_cache
        
        result = {}
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def fetch_single_xstock(symbol, config):
            try:
                # Need to use a shorter timeout or implicit CCXT handles it
                ticker = self.exchange.fetch_ticker(symbol)
                info = ticker.get('info', {})
                
                funding_rate = float(info.get('fundingRate', 0))
                mark_price = ticker.get('markPrice', 0)
                oi = float(info.get('openInterest', 0))
                
                # APY = funding_rate * 3 * 365 * 100
                apy = funding_rate * 3 * 365 * 100
                
                return symbol, {
                    'name': config['name'],
                    'category': config['category'],
                    'funding_rate': funding_rate,
                    'funding_8h_pct': funding_rate * 100,
                    'apy': apy,
                    'mark_price': mark_price,
                    'open_interest': oi,
                    'max_position': oi * MAX_POSITION_OI_PCT * mark_price,  # USD value
                    'timestamp': now,
                }
            except Exception as e:
                return symbol, {'error': str(e)}

        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_symbol = {executor.submit(fetch_single_xstock, sym, cfg): sym for sym, cfg in XSTOCKS.items()}
            for future in as_completed(future_to_symbol):
                sym = future_to_symbol[future]
                try:
                    res_sym, res_data = future.result()
                    result[res_sym] = res_data
                except Exception as exc:
                    result[sym] = {'error': str(exc)}
        
        self.funding_cache = result
        self.last_funding_check = now
        return result
    
    def find_arbitrage_opportunities(self, 
                                      min_apy: float = 100,
                                      min_oi: float = 10) -> List[Dict]:
        """
        Find xStocks with extreme funding rates suitable for arbitrage
        
        Args:
            min_apy: Minimum APY threshold
            min_oi: Minimum open interest in USD
            
        Returns:
            List of arbitrage opportunities
        """
        funding_data = self.fetch_xstocks_funding()
        opportunities = []
        
        for symbol, data in funding_data.items():
            if 'error' in data:
                continue
            
            # Skip if OI too low
            if data['open_interest'] * data['mark_price'] < min_oi:
                continue
            
            # Check for extreme funding
            if abs(data['apy']) >= min_apy:
                opportunity = {
                    'symbol': symbol,
                    'name': data['name'],
                    'category': data['category'],
                    'apy': data['apy'],
                    'funding_rate': data['funding_rate'],
                    'mark_price': data['mark_price'],
                    'oi_usd': data['open_interest'] * data['mark_price'],
                    'max_position_usd': data['max_position'],
                }
                
                # Determine arbitrage direction
                if data['funding_rate'] < FUNDING_THRESHOLD_SHORT:
                    opportunity['strategy'] = 'SHORT_ARB'
                    opportunity['description'] = f"Earn {data['apy']:.1f}% APY by SHORTING"
                    opportunities.append(opportunity)
                    
                elif data['funding_rate'] > FUNDING_THRESHOLD_LONG:
                    opportunity['strategy'] = 'LONG_ARB'
                    opportunity['description'] = f"Earn {data['apy']:.1f}% APY by LONGING"
                    opportunities.append(opportunity)
        
        # Sort by absolute APY
        opportunities.sort(key=lambda x: abs(x['apy']), reverse=True)
        return opportunities
    
    def get_hedge_ratio(self, xstock_symbol: str, hedge_asset: str = 'BTC/USD:USD') -> Optional[float]:
        """
        Calculate optimal hedge ratio between xStock and crypto asset
        
        Args:
            xstock_symbol: xStock to hedge (e.g., 'SPYX/USD:USD')
            hedge_asset: Crypto asset to hedge with (e.g., 'BTC/USD:USD')
            
        Returns:
            Hedge ratio (units of hedge per unit of xStock)
        """
        try:
            # Fetch price data (simplified - would use historical correlation in production)
            xstock_ticker = self.exchange.fetch_ticker(xstock_symbol)
            hedge_ticker = self.exchange.fetch_ticker(hedge_asset)
            
            xstock_price = xstock_ticker.get('markPrice', 0)
            hedge_price = hedge_ticker.get('markPrice', 0)
            
            if xstock_price and hedge_price:
                # Simple price-based ratio (would use beta/correlation in production)
                return xstock_price / hedge_price
            
        except Exception as e:
            print(f"Hedge ratio calculation failed: {e}")
        
        return None
    
    def generate_arb_signal(self, 
                            symbol: str, 
                            account_equity: float,
                            risk_per_trade: float = 0.10) -> Optional[Dict]:
        """
        Generate complete arbitrage signal for a given xStock
        
        Args:
            symbol: xStock symbol
            account_equity: Total account equity in USD
            risk_per_trade: % of equity to risk per trade
            
        Returns:
            Trading signal dict or None if no opportunity
        """
        funding_data = self.fetch_xstocks_funding()
        
        if symbol not in funding_data or 'error' in funding_data[symbol]:
            return None
        
        data = funding_data[symbol]
        
        # Determine direction
        if data['funding_rate'] < FUNDING_THRESHOLD_SHORT:
            direction = 'SELL'  # Short to earn funding
            hedge_direction = 'BUY'  # Hedge with long
        elif data['funding_rate'] > FUNDING_THRESHOLD_LONG:
            direction = 'BUY'  # Long to earn funding
            hedge_direction = 'SELL'  # Hedge with short
        else:
            return None  # No arb opportunity
        
        # Calculate position size
        max_position = data['max_position']
        target_position = account_equity * risk_per_trade
        position_size = min(target_position, max_position)
        
        # Calculate quantity
        quantity = position_size / data['mark_price']
        quantity = round(quantity, 2)  # Round to 2 decimals
        
        # Find hedge asset (for tech stocks, use BTC; for ETFs, use diversified)
        if data['category'] == 'Tech':
            hedge_asset = 'BTC/USD:USD'
        else:
            hedge_asset = 'BTC/USD:USD'  # Default to BTC for now
        
        hedge_ratio = self.get_hedge_ratio(symbol, hedge_asset)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'xstock': {
                'symbol': symbol,
                'name': data['name'],
                'direction': direction,
                'quantity': quantity,
                'price': data['mark_price'],
                'notional': quantity * data['mark_price'],
                'funding_rate': data['funding_rate'],
                'expected_apy': data['apy'],
            },
            'hedge': {
                'symbol': hedge_asset,
                'direction': hedge_direction,
                'hedge_ratio': hedge_ratio,
                'quantity': round(quantity * hedge_ratio, 6) if hedge_ratio else 0,
            },
            'risk': {
                'max_position_usd': max_position,
                'target_position_usd': target_position,
                'actual_position_usd': quantity * data['mark_price'],
            },
            'metadata': {
                'strategy': 'FUNDING_CARRY',
                'pool': 'B',  # Pool B for arb strategies
                'category': data['category'],
            }
        }
    
    def monitor_and_rebalance(self, 
                               positions: Dict[str, Dict],
                               funding_threshold: float = 0.005) -> List[Dict]:
        """
        Monitor open arb positions and suggest rebalancing
        
        Args:
            positions: Current positions {symbol: position_data}
            funding_threshold: Funding rate change threshold for rebalance
            
        Returns:
            List of rebalance actions
        """
        actions = []
        current_funding = self.fetch_xstocks_funding()
        
        for symbol, position in positions.items():
            if symbol not in current_funding:
                continue
            
            data = current_funding[symbol]
            entry_funding = position.get('entry_funding_rate', 0)
            current_funding_rate = data['funding_rate']
            
            # Check if funding rate has converged (exit signal)
            if abs(current_funding_rate) < funding_threshold:
                actions.append({
                    'action': 'CLOSE',
                    'symbol': symbol,
                    'reason': f'Funding converged: {entry_funding_rate*100:.2f}% -> {current_funding_rate*100:.2f}%',
                    'priority': 'HIGH',
                })
            
            # Check if funding rate has reversed (flip signal)
            if (entry_funding < 0 and current_funding_rate > funding_threshold) or \
               (entry_funding > 0 and current_funding_rate < -funding_threshold):
                actions.append({
                    'action': 'FLIP',
                    'symbol': symbol,
                    'reason': f'Funding reversed: {entry_funding_rate*100:.2f}% -> {current_funding_rate*100:.2f}%',
                    'priority': 'MEDIUM',
                })
        
        return actions


# Convenience functions for integration with HolonicTrader

def scan_xstocks_arb(account_equity: float = 10000) -> List[Dict]:
    """
    Scan xStocks for arbitrage opportunities
    
    Args:
        account_equity: Current account equity in USD
        
    Returns:
        List of trading signals
    """
    arb = XStocksArbitrage()
    opportunities = arb.find_arbitrage_opportunities(min_apy=100, min_oi=10)
    
    signals = []
    for opp in opportunities[:5]:  # Top 5 opportunities
        signal = arb.generate_arb_signal(opp['symbol'], account_equity)
        if signal:
            signals.append(signal)
    
    return signals


def get_xstocks_summary() -> str:
    """Get formatted summary of xStocks funding rates"""
    arb = XStocksArbitrage()
    funding = arb.fetch_xstocks_funding()
    
    lines = ["=" * 80, "xSTOCKS FUNDING SUMMARY", "=" * 80]
    
    for symbol, data in sorted(funding.items(), key=lambda x: abs(x[1].get('apy', 0)), reverse=True):
        if 'error' in data:
            continue
        
        fire = "[ARB]" if abs(data['apy']) > 500 else "     "
        direction = "SHORT" if data['funding_rate'] < 0 else "LONG "
        
        lines.append(
            f"{fire} {symbol:<15} {data['name']:<20} {direction:<6} "
            f"{data['apy']:>8.1f}% APY  OI: ${data['open_interest']*data['mark_price']:,.0f}"
        )
    
    return "\n".join(lines)


if __name__ == '__main__':
    # Test run
    print(get_xstocks_summary())
    print("\n" + "=" * 80)
    print("TOP ARBITRAGE SIGNALS")
    print("=" * 80)
    
    signals = scan_xstocks_arb(account_equity=10000)
    for sig in signals:
        print(f"\n{sig['xstock']['symbol']}:")
        print(f"  Strategy: {sig['xstock']['direction']} @ {sig['xstock']['price']}")
        print(f"  Expected APY: {sig['xstock']['expected_apy']:.1f}%")
        print(f"  Position: ${sig['risk']['actual_position_usd']:.2f}")
