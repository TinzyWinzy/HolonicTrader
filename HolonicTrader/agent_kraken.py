"""
KrakenHolon - The Platform Informer (Phase 1)

This holon is specialized in understanding the Kraken Trading Platform.
It monitors API status, detailed funding rates, open interest, and account health.
"""

import time
import ccxt
import config
from typing import Any, Dict, List, Optional
from HolonicTrader.holon_core import Holon, Disposition, Message

class KrakenHolon(Holon):
    """
    KrakenHolon is the 'Informer' that provides platform-specific intelligence.
    It acts as a bridge between the exchange's raw metadata and the system's decision logic.
    """
    def __init__(self, name: str = "KrakenInformer"):
        super().__init__(name=name, disposition=Disposition(autonomy=0.7, integration=0.9))
        
        # Initialize CCXT exchange objects
        self.futures = ccxt.krakenfutures({
            'apiKey': config.KRAKEN_FUTURES_API_KEY or config.API_KEY,
            'secret': config.KRAKEN_FUTURES_PRIVATE_KEY or config.API_SECRET,
            'enableRateLimit': True
        })
        
        self.spot = ccxt.kraken({
            'apiKey': config.KRAKEN_SPOT_KEY or config.API_KEY,
            'secret': config.KRAKEN_SPOT_SECRET or config.API_SECRET,
            'enableRateLimit': True
        })

        self.last_status_check = 0
        self.cached_status = "UNKNOWN"
        self.last_funding_data = {}
        self.last_oi_data = {}

    def receive_message(self, sender: Any, content: Any) -> None:
        """
        Handle incoming requests for Kraken-specific data.
        """
        # Holons are primarily passive informers for now
        pass

    def get_platform_info(self) -> Dict[str, Any]:
        """
        Returns a comprehensive summary of Kraken's current state.
        """
        return {
            'status': self.get_system_status(),
            'account_health': self.get_account_health(),
            'market_intel': {
                'funding': self.last_funding_data,
                'open_interest': self.last_oi_data
            }
        }

    def get_system_status(self) -> str:
        """
        Fetch Kraken's current system status (Spot and Futures).
        """
        now = time.time()
        if now - self.last_status_check < 300: # Cache for 5 mins
            return self.cached_status

        try:
            status = "HEALTHY"
            # Check spot status
            spot_status = self.spot.fetch_status()
            if spot_status.get('status') != 'ok':
                status = "SPOT_DEGRADED"

            # Check futures status
            # Kraken Futures doesn't have a direct equivalent but we can check if markets are reachable
            self.futures.fetch_markets()
            
            self.cached_status = status
            self.last_status_check = now
            return status
        except Exception as e:
            # Avoid using ERROR so Trader doesn't spam warnings on CCXT HTTP timeouts
            self.cached_status = f"UNKNOWN ({str(e)[:30]}...)"
            return self.cached_status

    def get_account_health(self) -> Dict[str, Any]:
        """
        Calculate detailed account health metrics for Kraken Futures.
        """
        try:
            balance = self.futures.fetch_balance()
            info = balance.get('info', {})
            accounts = info.get('accounts', {})
            flex = accounts.get('flex', {})
            
            # Key Health Metrics
            margin_equity = float(flex.get('marginEquity', 0.0))
            used_margin = float(flex.get('usedMargin', 0.0))
            available_margin = float(flex.get('availableMargin', 0.0))
            
            # Maintenance Margin Ratio
            mm = float(flex.get('maintenanceMargin', 0.0))
            
            liquidation_distance = 1.0
            if margin_equity > 0:
                liquidation_distance = (margin_equity - mm) / margin_equity

            return {
                'equity': margin_equity,
                'used_margin': used_margin,
                'available': available_margin,
                'margin_level': (margin_equity / used_margin) if used_margin > 0 else 10.0,
                'liquidation_distance': max(0.0, liquidation_distance),
                'status': 'WARNING' if liquidation_distance < 0.2 else 'SAFE'
            }
        except Exception as e:
            return {'error': str(e), 'status': 'OFFLINE'}

    def update_market_intel(self, symbols: List[str]):
        """
        Update funding rates and open interest for targeted symbols.
        """
        for symbol in symbols:
            try:
                exec_symbol = config.KRAKEN_SYMBOL_MAP.get(symbol, symbol)
                ticker = self.futures.fetch_ticker(exec_symbol)
                
                # Funding Intel
                funding_rate = ticker.get('info', {}).get('fundingRate', 0.0)
                next_funding_ts = ticker.get('info', {}).get('nextFundingRateTime', 0)
                
                self.last_funding_data[symbol] = {
                    'rate': float(funding_rate),
                    'next_ts': next_funding_ts,
                    'apy': float(funding_rate) * 3 * 365 * 100 # Approx 8h funding
                }

                # Open Interest Intel
                # Some ccxt tickers include OI in 'info'
                oi = ticker.get('info', {}).get('openInterest', 0.0)
                self.last_oi_data[symbol] = float(oi)

            except Exception as e:
                # print(f"[KrakenHolon] ⚠️ Intel Fetch Fail for {symbol}: {e}")
                continue

    def detect_ghost_positions(self, internal_held_assets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scan for discrepancies between Kraken's reality and the system's ledger.

        FIX 2026-02-25: Convert exchange symbols to INTERNAL format (BTC/USDT) not CCXT format.
        The internal ledger uses BTC/USDT format, so we must match that format.
        """
        try:
            # 1. Fetch live positions from Kraken
            # FIX: Use fetch_positions() as info['positions'] is empty in fetch_balance()
            kraken_pos_list = self.futures.fetch_positions()

            # Map of live positions: symbol -> quantity (using INTERNAL format)
            live_positions = {}
            for p in kraken_pos_list:
                qty = float(p.get('contracts', 0.0))
                if qty == 0: continue

                # Check side
                if p.get('side') == 'short':
                    qty = -qty

                # FIX: Convert Kraken API symbol to INTERNAL format (BTC/USDT)
                raw_sym = p.get('info', {}).get('symbol', p.get('symbol', ''))
                
                # Convert to internal format using reverse lookup in KRAKEN_SYMBOL_MAP
                internal_sym = self._normalize_kraken_symbol_to_internal(raw_sym)

                if internal_sym:
                    live_positions[internal_sym] = qty

            report = {
                'ghosts': {},  # On Exchange, but NOT in Ledger
                'leaks': {},   # In Ledger, but NOT on Exchange
                'mismatch': {}  # Quantity difference
            }

            # 2. Check for Ghosts (On Exchange, Not in Ledger)
            for sym, qty in live_positions.items():
                if sym not in internal_held_assets:
                    report['ghosts'][sym] = qty
                else:
                    # Check for Mismatch (Magnitude difference > 0.1%)
                    local_qty = internal_held_assets[sym]
                    if abs(qty - local_qty) / max(abs(qty), 0.000001) > 0.001:
                        report['mismatch'][sym] = {'exchange': qty, 'ledger': local_qty}

            # 3. Check for Leaks (In Ledger, Not on Exchange)
            for sym, qty in internal_held_assets.items():
                if sym not in live_positions:
                    report['leaks'][sym] = qty

            return report
        except Exception as e:
            return {'error': str(e)}
    
    def _normalize_kraken_symbol(self, kraken_symbol: str) -> str:
        """
        Convert Kraken Futures API symbol to CCXT format.
        DEPRECATED: Use _normalize_kraken_symbol_to_internal() instead.

        Examples:
            PF_XBTUSD -> BTC/USD:USD
            PF_SOLUSD -> SOL/USD:USD
            PF_SPYXUSD -> SPYX/USD:USD
        """
        # Remove PF_ prefix
        if kraken_symbol.startswith('PF_'):
            base = kraken_symbol[3:]  # Remove 'PF_'
        else:
            base = kraken_symbol

        # Handle XBT -> BTC conversion
        if base.startswith('XBT'):
            base = 'BTC' + base[3:]

        # Extract base currency (remove trailing USD)
        if base.endswith('USD'):
            base_currency = base[:-3]
        else:
            base_currency = base

        # Return in CCXT format
        return f"{base_currency}/USD:USD"

    def _normalize_kraken_symbol_to_internal(self, kraken_symbol: str) -> str:
        """
        Convert Kraken Futures API symbol to INTERNAL ledger format (BTC/USDT).
        
        FIX 2026-02-25: Uses reverse lookup in KRAKEN_SYMBOL_MAP to ensure
        exchange symbols match the internal ledger format.

        Examples:
            PF_XBTUSD -> BTC/USDT
            PF_SOLUSD -> SOL/USDT
            BTC/USD:USD -> BTC/USDT
        """
        import config
        
        # Remove PF_ prefix if present
        if kraken_symbol.startswith('PF_'):
            base = kraken_symbol[3:]
        else:
            base = kraken_symbol

        # Handle XBT -> BTC conversion
        if base.startswith('XBT'):
            base = 'BTC' + base[3:]

        # Extract base currency (remove trailing USD)
        if base.endswith('USD'):
            base_currency = base[:-3]
        else:
            base_currency = base

        # First try direct format: BTC/USD:USD -> BTC/USDT
        ccxt_format = f"{base_currency}/USD:USD"
        
        # Reverse lookup in KRAKEN_SYMBOL_MAP
        for internal_sym, exchange_sym in config.KRAKEN_SYMBOL_MAP.items():
            if exchange_sym == ccxt_format:
                return internal_sym
        
        # Fallback: guess internal format
        guess = f"{base_currency}/USDT"
        if guess in config.ALLOWED_ASSETS:
            return guess
        
        # Last resort: return None (will be filtered out)
        return None

    def get_equity_truth(self) -> Dict[str, Any]:
        """
        Fetch the 'Ground Truth' for equity and collateral from Kraken.
        Crucial for stopping EQUITY DIVERGENCE loops.
        
        FIX 2026-03-02: Prefer USDT balance over USD for unified balance accounts.
        """
        try:
            bal = self.futures.fetch_balance()
            info = bal.get('info', {})
            accounts = info.get('accounts', {})
            flex = accounts.get('flex', {})
            
            # FIX 2026-03-02: Check for USDT collateral first (user's primary holding)
            # Kraken Futures unified balance shows all collateral values in USD equivalent
            # but we need to check what currency the user actually holds
            
            # Method 1: Check balances dict for USDT
            balances = bal.get('info', {}).get('balances', {})
            usdt_balance = float(balances.get('USDT', {}).get('value', 0.0)) if 'USDT' in balances else 0.0
            usd_balance = float(balances.get('USD', {}).get('value', 0.0)) if 'USD' in balances else 0.0
            
            # Use USDT if available and significant, otherwise fall back to USD
            if usdt_balance > 1.0:  # User holds USDT
                collateral = usdt_balance
                print(f"[KrakenHolon] 💵 Using USDT balance: ${collateral:.2f} (USD: ${usd_balance:.2f})")
            else:
                # Fall back to flex account balance (unified USD value)
                collateral = float(flex.get('balanceValue', 0.0))
                if usdt_balance > 0:
                    print(f"[KrakenHolon] 💵 USDT detected but small (${usdt_balance:.2f}), using unified: ${collateral:.2f}")
            
            # Total Equity (Collateral + Unrealized PnL) - always in USD equivalent
            margin_equity = float(flex.get('marginEquity', 0.0))
            
            # Unrealized PnL
            # FIX: Use 'totalUnrealized' as 'unrealizedPnL' is not present/deprecated
            unrealized_pnl = float(flex.get('totalUnrealized', 0.0))
            
            # KRAKEN FLEXLINE (2026-03-09): Net Worth Calculation
            loan_amount = getattr(config, 'LOAN_DETAILS', {}).get('ACTIVE_LOAN_AMOUNT', 0.0)
            net_worth = margin_equity - loan_amount

            return {
                'collateral': collateral,
                'collateral_currency': 'USDT' if usdt_balance > 1.0 else 'USD',
                'equity': margin_equity,
                'net_worth': net_worth,
                'loan_amount': loan_amount,
                'unrealized_pnl': unrealized_pnl,
                'available': float(flex.get('availableMargin', 0.0)),
                'usdt_balance': usdt_balance,
                'usd_balance': usd_balance,
                'timestamp': time.time()
            }
        except Exception as e:
            return {'error': str(e)}

    def project_funding_impact(self, symbols: List[str], held_assets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Project the USD impact of the next funding window on current equity.
        """
        impacts = {}
        total_impact = 0.0
        
        for sym in symbols:
            # We only care about assets we actually hold
            if sym not in held_assets:
                continue
                
            try:
                intel = self.last_funding_data.get(sym)
                if not intel:
                    continue
                
                # Funding Rate is decimal (e.g., 0.0001 per window)
                rate = intel['rate']
                # Position size (positive for Long, negative for Short)
                qty = held_assets[sym]
                
                # We need the current price to calculate notional value
                # Using last fetched price from intel or spot
                # Notional = Price * Qty
                # Impact = - (Notional * Rate) -> If Long and Rate is positive, you PAY.
                
                # Simplified for Informer: we'll use a placeholder or the last known ticker price
                # For high accuracy, we'd need current price here
                notional = 0.0 # Placeholder
                ticker = self.futures.fetch_ticker(self.translate_symbol(sym))
                notional = abs(qty * ticker['last'])
                
                impact = - (notional * rate) if qty > 0 else (notional * rate)
                impacts[sym] = impact
                total_impact += impact
            except:
                continue
                
        return {
            'total_projected_usd': total_impact,
            'asset_breakdown': impacts,
            'next_window_ts': min([v['next_ts'] for v in self.last_funding_data.values() if 'next_ts' in v] or [0])
        }

    def get_tiered_margin_status(self, symbol: str, current_qty: float, additional_qty: float) -> Dict[str, Any]:
        """
        Check if an additional trade will trigger a Kraken Tiered Margin hike.
        Note: Kraken Futures Tiers are typically:
        Tier 1: Up to $250k (Base Initial Margin)
        Tier 2: $250k - $500k (+0.5% margin)
        ... etc.
        For NANO accounts, we are almost always in Tier 1.
        """
        ticker = self.futures.fetch_ticker(self.translate_symbol(symbol))
        price = ticker['last']
        total_notional = abs(current_qty + additional_qty) * price
        
        # Simple Kraken Tier Threshold (Example: $250,000)
        TIER_THRESHOLD = 250000.0 
        
        is_approaching = total_notional > (TIER_THRESHOLD * 0.8)
        is_breached = total_notional > TIER_THRESHOLD
        
        return {
            'notional_usd': total_notional,
            'tier': 1 if not is_breached else 2,
            'status': 'BREACHED' if is_breached else ('WARNING' if is_approaching else 'OK'),
            'distance_to_tier_hike': max(0.0, TIER_THRESHOLD - total_notional)
        }

    def resolve_ghosts(self, report: Dict[str, Any], global_bias: float) -> List[Dict[str, Any]]:
        """
        Decision engine for Ghost/Leak resolution.
        Returns a list of actions to be taken by the system.
        Action types: 'ADOPT' (Add to Ledger), 'EXORCISE' (Close on Kraken), 'RECONCILE' (Update Ledger Qty)
        """
        actions = []
        
        # 1. Resolve Ghosts (On Exchange, missing in Ledger)
        for sym, qty in report.get('ghosts', {}).items():
            # Strategy Decision:
            # - If bias matches direction (Long/Bullish, Short/Bearish) AND it's a major asset -> ADOPT
            direction = 'BUY' if qty > 0 else 'SELL'
            is_bullish = global_bias >= 0.55
            is_bearish = global_bias <= 0.45
            
            should_adopt = False
            if direction == 'BUY' and is_bullish: should_adopt = True
            if direction == 'SELL' and is_bearish: should_adopt = True
            
            # ADOPT if it fits the trend, else EXORCISE
            if should_adopt:
                actions.append({
                    'type': 'ADOPT',
                    'symbol': sym,
                    'qty': qty,
                    'reason': f"Ghost fits global bias ({global_bias:.2f})"
                })
            else:
                actions.append({
                    'type': 'EXORCISE',
                    'symbol': sym,
                    'qty': qty,
                    'reason': f"Unsanctioned Ghost vs Bias ({global_bias:.2f})"
                })

        # 2. Resolve Leaks (In Ledger, missing on Kraken)
        for sym, qty in report.get('leaks', {}).items():
            # If it's in the ledger but gone from Kraken, we must purge it from Ledger
            # It was likely liquidated or manually closed.
            actions.append({
                'type': 'PURGE',
                'symbol': sym,
                'reason': "Leak: Position no longer exists on exchange"
            })

        # 3. Resolve Mismatches
        for sym, data in report.get('mismatch', {}).items():
            # Reconcile ledger to match Kraken reality
            actions.append({
                'type': 'RECONCILE',
                'symbol': sym,
                'new_qty': data['exchange'],
                'reason': f"Size mismatch corrected to Exchange truth ({data['exchange']})"
            })

        return actions

    def sync_server_side_stops(self, held_assets: Dict[str, Any], stop_loss_pct: float, executor=None) -> Dict[str, Any]:
        """
        Verify every open position has a Server-Side Stop Loss on Kraken.
        If missing, place the stop order automatically.
        FIX 2026-02-24: Handle Kraken Futures symbol format (PF_* -> internal).
        FIX 2026-03-01: Actually place missing stops instead of just reporting.
        FIX 2026-03-01: Use executor's actuator for stop placement if available.
        FIX 2026-03-01 #6: Better error handling and diagnostics for stop placement failures.
        """
        results = {'synced': [], 'missing': [], 'placed': [], 'errors': []}
        try:
            # 1. Fetch Open Trigger Orders from Kraken
            try:
                open_orders = self.futures.fetch_open_orders()
            except Exception as e:
                print(f"[{self.name}] ❌ Failed to fetch open orders: {e}")
                results['errors'].append('fetch_orders_failed')
                return results

            # Map of symbol -> list of stop orders
            stop_orders = {}
            for o in open_orders:
                if o.get('type') in ['stop', 'stop_loss', 'stop-loss-limit', 'trigger']:
                    sym = o.get('symbol')
                    # Back-translate Kraken format (PF_ADAUSD) to internal (ADA/USDT)
                    sys_sym = sym
                    if sym.startswith('PF_'):
                        # Extract base: PF_ADAUSD -> ADA
                        kraken_base = sym[3:].replace('USD', '').replace('USDT', '')
                        # Convert to internal format
                        sys_sym = f"{kraken_base}/USDT"
                    else:
                        # Try standard mapping
                        for s, k in config.KRAKEN_SYMBOL_MAP.items():
                            if k == sym:
                                sys_sym = s
                                break
                    if sys_sym not in stop_orders: stop_orders[sys_sym] = []
                    stop_orders[sys_sym].append(o)

            # 2. Check against Held Assets and place missing stops
            # FIX 2026-03-01: Use executor's actuator if available
            actuator = None
            if executor and hasattr(executor, 'actuator'):
                actuator = executor.actuator
            elif hasattr(self, 'actuator') and self.actuator:
                actuator = self.actuator
            
            for sym, qty in held_assets.items():
                if abs(qty) < 1e-8:  # Skip dust positions
                    continue
                    
                # FIX: Also check for PF_* format matches
                base_sym = sym.split(':')[0] if ':' in sym else sym
                kraken_pf = f"PF_{base_sym.replace('/USDT', '').replace('/USD', '')}USD"

                # Check both internal and Kraken format
                has_stop = (sym in stop_orders or 
                           kraken_pf in [f"PF_{s.replace('/USDT','').replace('/USD','')}USD" for s in stop_orders.keys()])
                
                if not has_stop:
                    # FIX 2026-03-01: Place the missing stop order
                    try:
                        # Get entry price from executor if available
                        entry_price = 0.0
                        if executor and hasattr(executor, 'entry_prices'):
                            entry_price = executor.entry_prices.get(sym, 0.0)

                        if entry_price <= 0:
                            # Fetch current price as fallback
                            try:
                                ticker = self.futures.fetch_ticker(kraken_pf if kraken_pf else sym)
                                entry_price = ticker.get('last', ticker.get('close', 0.0))
                            except Exception as price_err:
                                print(f"[{self.name}] ⚠️ Could not fetch price for {sym}: {price_err}")
                                results['errors'].append(f'{sym}:price_fetch_failed')
                                continue

                        # Calculate stop price
                        direction = 'BUY' if qty > 0 else 'SELL'
                        stop_direction = 'SELL' if direction == 'BUY' else 'BUY'
                        stop_price = entry_price * (1 - stop_loss_pct) if direction == 'BUY' else entry_price * (1 + stop_loss_pct)

                        # FIX 2026-03-01 #6: Validate quantity before attempting stop placement
                        # This prevents errors from floating point precision issues
                        base_asset = sym.split('/')[0]
                        min_qty = getattr(config, 'MIN_TRADE_QTY', {}).get(base_asset, 0.0)
                        
                        # Round quantity to avoid floating point errors (e.g., 0.009999999999999998)
                        rounded_qty = round(abs(qty), 8)
                        
                        if min_qty > 0 and rounded_qty < min_qty:
                            # Position is below minimum - use minimum if position allows
                            if abs(qty) >= min_qty:
                                rounded_qty = round(min_qty, 8)
                                print(f"[{self.name}] ⚠️ Stop qty rounded up to minimum {min_qty} for {sym}")
                            else:
                                print(f"[{self.name}] ⚠️ Position {sym} qty {qty:.8f} below minimum {min_qty}, using rounded qty {rounded_qty:.8f}")

                        # Place stop order via actuator
                        if actuator:
                            # Use rounded quantity to prevent precision errors
                            success = actuator.place_stop_order(sym, stop_direction, rounded_qty, stop_price)
                            if success:
                                results['placed'].append(sym)
                                print(f"[{self.name}] ✅ AUTO-PLACED missing stop for {sym}: {stop_direction} {rounded_qty} @ {stop_price:.4f}")
                            else:
                                results['errors'].append(f'{sym}:placement_failed')
                                print(f"[{self.name}] ❌ Failed to place stop for {sym} (qty={rounded_qty:.8f})")
                        else:
                            results['errors'].append(f'{sym}:no_actuator')
                            print(f"[{self.name}] ⚠️ No actuator available for stop placement on {sym}")
                    except Exception as e:
                        results['errors'].append(f'{sym}:{str(e)[:50]}')
                        print(f"[{self.name}] ❌ Error placing stop for {sym}: {e}")
                else:
                    results['synced'].append(sym)

            # Report summary
            if results['placed']:
                print(f"[{self.name}] 🛡️ Server-Side Stops: {len(results['synced'])} synced, {len(results['placed'])} auto-placed, {len(results['errors'])} errors")
            elif results['missing']:
                print(f"[{self.name}] 🛡️ Server-Side Stops: {len(results['synced'])} synced, {len(results['missing'])} missing (placement failed)")

            return results
        except Exception as e:
            return {'error': str(e)}

    def get_collateral_haircuts(self) -> Dict[str, float]:
        """
        Fetch Kraken's collateral haircut ratios.
        Note: Fixed for now based on Kraken documentation, but can be dynamic.
        """
        # Example Haircut Ratios (1.0 = No Haircut, 0.9 = 10% Haircut)
        # Source: Kraken Futures Multi-Collateral Docs
        return {
            'USD': 1.0,
            'USDT': 1.0,
            'USDC': 1.0,
            'BTC': 0.90,
            'ETH': 0.90,
            'DOT': 0.70,
            'ADA': 0.70
        }

    def monitor_execution_environment(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Senses 'Flash Crash' conditions: wide spreads or API latency.
        """
        env_status = {}
        dangerous_symbols = []
        
        try:
            tickers = self.futures.fetch_tickers(self.translate_symbol(s) for s in symbols)
            for sys_sym in symbols:
                k_sym = self.translate_symbol(sys_sym)
                t = tickers.get(k_sym)
                if not t: continue
                
                bid = t.get('bid', 0)
                ask = t.get('ask', 0)
                if bid > 0:
                    spread_pct = (ask - bid) / bid
                    # If spread > 0.5%, it's dangerous
                    if spread_pct > 0.005:
                        dangerous_symbols.append({'symbol': sys_sym, 'spread': spread_pct})
            
            status = 'CRASH_RISK' if dangerous_symbols else 'STABLE'
            return {
                'status': status,
                'dangerous_assets': dangerous_symbols,
                'latency_ms': 0 # Placeholder for API ping
            }
        except:
            return {'status': 'UNKNOWN'}

    def translate_symbol(self, symbol: str) -> str:
        """
        Translates a unified symbol to Kraken Futures format.
        """
        return config.KRAKEN_SYMBOL_MAP.get(symbol, symbol)
