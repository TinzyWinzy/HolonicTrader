
import pandas as pd
import numpy as np
import time
from typing import List, Dict, Any, Type, Tuple
import sys
import os

# Add parent dir to path to find HolonicTrader modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sandbox.strategies.base import Strategy, Signal
from HolonicTrader.agent_observer import ObserverHolon
from HolonicTrader.data_guard import DataGuard
try:
    import holonic_speed # Use Rust for Indicators
    print("[Sandbox] ðŸš€ Rust Engine Loaded (High Performance)")
except ImportError:
    # Fallback to pure Python if Rust module is missing/incompatible
    print("[Sandbox] Rust Engine not found. Using Python Fallback (Slower).")
    
    # Python Fallback for Rust Indicators
    class HolonicSpeedFallback:
        @staticmethod
        def calculate_rsi(closes: List[float], period: int) -> List[float]:
            series = pd.Series(closes)
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(0.0).tolist()

        @staticmethod
        def calculate_bollinger_bands(closes: List[float], period: int, std_dev: float) -> Tuple[List[float], List[float], List[float]]:
            series = pd.Series(closes)
            middle = series.rolling(window=period).mean()
            std = series.rolling(window=period).std()
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            return (
                upper.fillna(0.0).tolist(), 
                middle.fillna(0.0).tolist(), 
                lower.fillna(0.0).tolist()
            )

        @staticmethod
        def calculate_atr(highs: List[float], lows: List[float], closes: List[float], period: int) -> List[float]:
            high = pd.Series(highs)
            low = pd.Series(lows)
            close = pd.Series(closes)
            
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            return atr.fillna(0.0).tolist()

    holonic_speed = HolonicSpeedFallback()

class Playground:
    """
    The Holonic Sandbox Engine.
    Simulates a trading environment for a given Strategy.
    """
    def __init__(self, symbol: str = 'BTC/USDT', initial_capital: float = 1000.0, verbose: bool = True, leverage: float = 1.0):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.verbose = verbose # Control output
        self.leverage = leverage
        self.inventory = 0.0
        self.entry_price = 0.0
        self.margin_locked = 0.0
        
        # Simulation Settings
        self.fee_rate = 0.001 # 0.1% Taker
        self.slippage = 0.0005 # 0.05% Slippage estimate
        self.latency_candles = 1 # 1 = Execute on Next Open (REALISM)
        
        self.trades = []
        self.equity_curve = []
        
        self.active_entry_fee = 0.0 # Track fees for accurate PnL
        
        self.df = None
        self.df_secondary = None
        self.strategy: Strategy = None

        # Data Quality Shield
        self.guard = DataGuard()

    def load_data(self, source: str = 'kraken', timeframe: str = '1h', limit: int = 1000):
        """Load historical data using ObserverHolon with Pickle Caching."""
        # Windows UTF-8 fix
        msg = f"[Sandbox] Loading {limit} candles for {self.symbol}..."
        try:
            print(msg)
        except UnicodeEncodeError:
            print(msg.encode('ascii', errors='replace'))
        observer = ObserverHolon(exchange_id=source)
        
        # Cache Path
        clean_sym = self.symbol.replace('/', '')
        cache_path = os.path.join(observer.data_dir, f"{clean_sym}_{timeframe}_{limit}_processed.pkl")
        
        try:
            # 1. Try Cache
            if os.path.exists(cache_path):
                # Check if cache is fresh enough (e.g. < 1 hour old?) - actually just check valid
                try:
                    self.df = pd.read_pickle(cache_path)
                    print(f"   âš¡ Cache Hit: Loaded {len(self.df)} candles from {cache_path}")
                    return # Skip Fetch + Compute
                except Exception as e:
                    print(f"   âš ï¸ Cache Corrupt: {e}")
                    
            # 2. Load/Fetch Raw
            self.df = observer.load_local_history(self.symbol)
            if self.df.empty or len(self.df) < limit:
                 print("   Local data insufficient/missing. Fetching from API...")
                 self.df = observer.fetch_market_data(symbol=self.symbol, timeframe=timeframe, limit=limit)
            
            # Ensure we own the data and limit it if it's not empty
            if not self.df.empty:
                self.df = self.df.tail(limit).copy()
            
            if not self.df.empty and 'timestamp' in self.df.columns:
                 print(f"   Loaded {len(self.df)} candles ({self.df['timestamp'].iloc[0]} - {self.df['timestamp'].iloc[-1]})")
                 
                 # 3. Compute
                 self._compute_indicators(self.df)
                 self._audit_data_integrity(self.df)
                 
                 # 4. Save Cache
                 try:
                     self.df.to_pickle(cache_path)
                     print(f"   ðŸ’¾ Cache Saved: {cache_path}")
                 except Exception as e:
                     print(f"   âš ï¸ Cache Save Failed: {e}")
                     
            else:
                 print(f"   âš ï¸ Loaded {len(self.df)} candles (Empty or Missing Timestamp)")
            
        except Exception as e:
            print(f"[Sandbox] âŒ Data Load Error: {e}")

    def load_secondary_data(self, timeframe: str = '15m', limit: int = 4000, source: str = 'kraken'):
        """Load secondary timeframe data (e.g., 15m) for Satellite logic."""
        print(f"[Sandbox] ðŸ“¥ Loading Secondary Data ({timeframe}) for {self.symbol}...")
        observer = ObserverHolon(exchange_id=source) # Default to kraken
        try:
            # Local mapping might differ for 15m, Observer handles it? 
            # Observer.load_local_history takes symbol, but usually assumes 1H default?
            # Let's use fetch_market_data directly or implement a load_local helper in Sandbox if needed
            # For now, rely on fetch which writes to disk, so next time it might load?
            # Actually Observer.load_local_history hardcodes to '1h' usually or we need to check agent_observer.
            
            # Direct fetch fallback pattern
            # === OPTIMIZATION: Pickle Cache ===
            clean_sym = self.symbol.replace('/', '')
            base_filename = f"{clean_sym}_{timeframe}"
            pickle_path = os.path.join(observer.data_dir, f"{base_filename}.pkl")
            csv_path = os.path.join(observer.data_dir, f"{base_filename}.csv")
            
            # 1. Try Pickle First
            if os.path.exists(pickle_path):
                try:
                    # Check timestamp if CSV exists to ensure freshness? 
                    # For simplicity, if pickle exists, we trust it (or assume Observer updates it)
                    self.df_secondary = pd.read_pickle(pickle_path)
                    print(f"   âš¡ Secondary Data Cache Hit ({timeframe})")
                    self._compute_indicators(self.df_secondary)
                    return # Done
                except Exception:
                    pass # Fallback
            
            # 2. CSV Fallback with Lock Fix
            if os.path.exists(csv_path):
                # LOCK FIX: Copy to temp to avoid collision with Live Bot writing
                try:
                    import shutil
                    temp_path = csv_path + ".tmp"
                    shutil.copy(csv_path, temp_path)
                    self.df_secondary = pd.read_csv(temp_path)
                    self.df_secondary['timestamp'] = pd.to_datetime(self.df_secondary['timestamp'])
                    try: os.remove(temp_path)
                    except: pass
                    
                    # SAVE CACHE
                    try: self.df_secondary.to_pickle(pickle_path)
                    except: pass
                    
                except Exception as e:
                    print(f"   âš ï¸ Read Error (Lock?): {e}")
                    self.df_secondary = None
            else:
                 print(f"   Local {timeframe} missing. Fetching...")
                 self.df_secondary = observer.fetch_market_data(symbol=self.symbol, timeframe=timeframe, limit=limit)
                 # Save if fetched
                 if self.df_secondary is not None and not self.df_secondary.empty:
                      try: self.df_secondary.to_pickle(pickle_path)
                      except: pass
            
            if self.df_secondary is not None and not self.df_secondary.empty:
                if 'timestamp' in self.df_secondary.columns:
                    # self.df_secondary = self.df_secondary.sort_values('timestamp') # optimize: assume sorted from API
                    print(f"   Loaded {len(self.df_secondary)} secondary candles")
                    
                    self._audit_data_integrity(self.df_secondary)
                    self._compute_indicators(self.df_secondary)
                else:
                    print(f"   âš ï¸ Secondary data missing 'timestamp' column. Ignoring.")
                    self.df_secondary = None
            else:
                self.df_secondary = None
                
        except Exception as e:
            print(f"[Sandbox] âŒ Secondary Data Load Error: {e}")
            self.df_secondary = None

    def _audit_data_integrity(self, df: pd.DataFrame):
        """Scan for 'Glitch Candles' using the DataGuard logic."""
        if df is None or df.empty: return
        
        glitch_indices = []
        for i in range(len(df)):
            row = df.iloc[i]
            is_valid = self.guard.audit_candle(
                self.symbol,
                row['open'], row['high'], row['low'], row['close'], row['volume']
            )
            if not is_valid:
                glitch_indices.append(df.index[i])
        
        if glitch_indices:
            print(f"   âš ï¸ DATA INTEGRITY ALERT: Found {len(glitch_indices)} Glitch Candles. Guarding...")
            # Ideally we mask them or skip, for now we log and warn.

    def _compute_indicators(self, df):
        """Pre-calculate standard indicators using Rust Speed."""
        if df is None or df.empty: return

        #print(f"[Sandbox] âš¡ Computing Indicators for {len(df)} rows...")
        closes = df['close'].values.tolist()
        highs = df['high'].values.tolist()
        lows = df['low'].values.tolist()
        
        # Rust Calls
        df['rsi'] = holonic_speed.calculate_rsi(closes, 14)
        u, m, l = holonic_speed.calculate_bollinger_bands(closes, 20, 2.0)
        df['bb_upper'] = u
        df['bb_middle'] = m
        df['bb_lower'] = l
        df['atr'] = holonic_speed.calculate_atr(highs, lows, closes, 14)
        
        # Returns
        df['returns'] = df['close'].pct_change()
        
        # --- WHALE PROXY (Simulation Speed) ---
        # 1. Relative Volume
        df['vol_avg_20'] = df['volume'].rolling(20).mean()
        df['rvol'] = df['volume'] / df['vol_avg_20']
        
        # 2. Bullish Hammer (Wick > Body)
        # Vectorized for speed
        body = (df['close'] - df['open']).abs()
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        df['is_hammer'] = lower_wick > body
        
        # 3. Whale Signal: High Volume + Rejection
        # RVol threshold 2.5 (Significant interest)
        df['whale_signal'] = (df['rvol'] > 2.5) & df['is_hammer'] & (df['returns'] > -0.05)
        
        df.dropna(inplace=True)

    def inject_strategy(self, strategy_instance: Strategy):
        """Load a strategy instance."""
        self.strategy = strategy_instance
        if self.verbose: print(f"[Sandbox] ðŸ’‰ Injected Strategy: {self.strategy.name}")

    def run(self):
        """Run the simulation loop."""
        if self.df is None or self.df.empty:
            print("[Sandbox] âš ï¸ No Data Loaded. Aborting.")
            return

        if self.verbose: print(f"[Sandbox] â–¶ï¸ Running Simulation on {len(self.df)} candles...")
        
        if hasattr(self.strategy, 'genome') and 'holonic_speed' in globals():
            try:
                self._run_warp_speed() 
                return # Skip Python Loop
            except Exception as e:
                if self.verbose: print(f"[Sandbox] Warp fallback: {e}")
                pass
        
        start_time = time.time()
        
        # Convert timestamp to index search if needed, usually we just filter
        # Optimization: Pre-align secondary data if possible?
        
        for i in range(len(self.df)):
            current_idx = self.df.index[i]
            row = self.df.iloc[i]
            current_ts = row['timestamp']
            
            # Prepare Indicators Dict
            try:
                indicators = {
                    'rsi': row['rsi'],
                    'bb_upper': row['bb_upper'],
                    'bb_lower': row['bb_lower'],
                    'bb_middle': row['bb_middle'],
                    'atr': row['atr'],
                    'price': row['close']
                }
            except KeyError as e:
                # print(f"[Sandbox] âŒ CRITICAL: Missing Column {e} in row. Available: {row.index.tolist()}")
                return
            
            portfolio_state = {
                'balance': self.capital,
                'inventory': self.inventory,
                'avg_entry': self.entry_price,
                'equity': self.get_equity(row['close'])
            }
            
            # Ask Strategy for Decision
            start_slice = max(0, i-50)
            slice_df = self.df.iloc[start_slice:i+1]
            
            # Secondary Slice
            # Get last 50 candles of 15m data ending at or before current_ts
            # Secondary Slice
            # Get last 50 candles of 15m data ending at or before current_ts
            secondary_slice = None
            if self.df_secondary is not None and not self.df_secondary.empty and 'timestamp' in self.df_secondary.columns:
                try:
                    # Faster: Searchsorted
                    idx = self.df_secondary['timestamp'].searchsorted(current_ts, side='right')
                    start_sec = max(0, idx - 50)
                    secondary_slice = self.df_secondary.iloc[start_sec:idx]
                except Exception:
                    pass # Fallback if search fails

            
            # --- INTRA-CANDLE STOP LOSS / TAKE PROFIT ---
            # Simulate "Stop Market" or "Take Profit" hit during the candle
            if self.inventory > 0:
                 # Check Stop Loss (Low <= SL)
                 active_sl = getattr(self, 'active_stop_loss', None)
                 if active_sl and row['low'] <= active_sl:
                     # FORCE SELL AT SL with CHAOS TAX
                     # During a stop-out, volatility is usually AGAINST you.
                     # We use the same dynamic logic but simplified for the breakdown
                     atr = row.get('atr', 0)
                     price = row['close']
                     vol_penalty = (atr/price * 0.5) if price > 0 else 0
                     slip_rate = self.slippage + vol_penalty
                     
                     exit_price = active_sl * (1 - slip_rate)
                     
                     # Simple logic: We stopped out.
                     # Record Trade
                     self._force_exit(row['timestamp'], exit_price, "Stop Loss (Intra)")
                     
                     # Update final equity for this candle/curve
                     # self.capital is updated in _force_exit
                     self.equity_curve.append({
                        'timestamp': row['timestamp'],
                        'equity': self.capital # Cash is equity since inventory 0
                     })
                     continue # SKIP Strategy Logic for this candle (we are out)
            
            # --- FUNDING FEES (The Rent) ---
            # Charge 0.01% every 8 hours on Notional Value of Position
            # Realistic Perp Physics: Longs pay Shorts in Bull Market.
            last_funding = getattr(self, 'last_funding_time', None)
            if last_funding is None: self.last_funding_time = current_ts
            
            # Check delta (Safe for any timeframe)
            time_delta = (current_ts - self.last_funding_time).total_seconds() / 3600.0 # Hours
            if time_delta >= 8.0 and self.inventory > 0:
                # Funding Event
                leverage = getattr(self, 'leverage', 1.0)
                # Notional Value = Current Value? Or Entry Notional? usually Current Mark
                current_notional = self.inventory * row['close']
                
                funding_rate = 0.0001 # 0.01% Baseline
                funding_cost = current_notional * funding_rate
                
                self.capital -= funding_cost
                self.last_funding_time = current_ts
                # if self.verbose: print(f"[Sandbox] ðŸ’¸ Funding: -${funding_cost:.2f}")

            # --- EXECUTION LATENCY (Next-Open Rule) ---
            # If we have a pending signal from previous candle, execute it NOW on the OPEN
            if getattr(self, 'pending_signal', None):
                 # Execute on Open Price (Realistic)
                 # Note: row contains 'open', 'high', 'low', 'close'
                 # in backtest we assume we fill at Open
                 open_price_row = row.copy()
                 open_price_row['close'] = row['open'] # Hack to use _execute with Open price
                 self._execute(self.pending_signal, open_price_row)
                 self.pending_signal = None

            # Generate NEW Decision based on Closed Candle
            signal = self.strategy.on_candle(slice_df, indicators, portfolio_state, secondary_slice_df=secondary_slice)
            
            # Store for Next Candle Execution (latency_candles = 1)
            # Default to 1 for realism
            if self.latency_candles == 0:
                 self._execute(signal, row) # Instant Cheat Execution
            else:
                 self.pending_signal = signal # Queue for Next Open
            
            # Record State
            current_equity = self.get_equity(row['close'])
            self.equity_curve.append({
                'timestamp': row['timestamp'],
                'equity': current_equity
            })

            # --- LIQUIDATION ENGINE (Realistic Physics) ---
            # Maintenance Margin check.
            # Standard MM is roughly 50% of Initial Margin for tiered leverage.
            if self.inventory > 0:
                maintenance_margin = self.margin_locked * 0.50 
                # Liquidation Condition: Equity drops below MM
                if current_equity < maintenance_margin:
                     if self.verbose: print(f"[Sandbox] â˜ ï¸ LIQUIDATION! Equity ${current_equity:.2f} < MM ${maintenance_margin:.2f}")
                     
                     # Liquidation Penalty: You lose MM too (Exchange keeps it)
                     # Or just zero.
                     self.capital = 0.0 
                     self.inventory = 0
                     
                     self.trades.append({
                        'id': len(self.trades)+1,
                        'type': 'LIQUIDATION',
                        'time': row['timestamp'],
                        'price': row['close'],
                        'qty': 0,
                        'pnl': -self.margin_locked, 
                        'reason': 'Account Zeroed'
                     })
                     break # GAME OVER

            # Bankruptcy Check
            if self.capital <= 0 and self.inventory == 0:
                if self.verbose: print("[Sandbox] ðŸ’¸ BANKRUPTCY. Simulation Terminated.")
                break

        elapsed = time.time() - start_time
        if self.verbose: 
            print(f"[Sandbox] âœ… Simulation Complete in {elapsed:.2f}s")
            self.report()

    def _execute(self, signal: Signal, row):
        price = row['close']
        
        # --- DYNAMIC SLIPPAGE (The Chaos Tax) ---
        # Base Slippage (0.05%) + Volatility Penalty
        # If ATR is 1% of price, add 0.2 * 1% = 0.2% extra slippage
        atr = row.get('atr', 0)
        volatility_pct = (atr / price) if price > 0 else 0
        dynamic_slip = self.slippage + (volatility_pct * 0.5) # 50% of volatility is paid as slip
        
        # Exec Logic
        # Buy fills higher, Sell fills lower
        exec_price = price * (1 + dynamic_slip) if signal.direction == 'BUY' else price * (1 - dynamic_slip)
        
        if signal.direction == 'BUY':
            allow_buy = False
            is_stacking = False
            
            # 1. New Position
            if self.inventory == 0:
                allow_buy = True
                self.stack_count = 1
                
            # 2. Add to Position (Stacking)
            # Default Limit: 2 Stacks
            # WHALE OVERRIDE: 4 Stacks (Aggressive Accumulation)
            elif self.inventory > 0:
                 limit = 4 if row.get('whale_signal', False) else 2
                 if getattr(self, 'stack_count', 1) < limit:
                    allow_buy = True
                    is_stacking = True
                    self.stack_count = getattr(self, 'stack_count', 1) + 1
                    if self.verbose: 
                        reason = "ðŸ‹ WHALE" if limit == 4 else "ðŸ¥ž STACK"
                        pass # print(f"[Sandbox] {reason} count: {self.stack_count}")

            if allow_buy:
                # Size Logic (Margin)
                pct_size = max(0.0, min(1.0, signal.size)) # Clamp 0-1
                
                # --- LIQUIDITY GATE ---
                # Realism: You cannot buy more than 1% of the candle's volume without moving the market.
                candle_volume_usd = row.get('volume', 0) * price 
                max_liquidity_allowed = candle_volume_usd * 0.01 # Max 1% of volume
                
                # If Volume is missing/zero, assume infinite (for unit tests) or clamp?
                # Let's assume infinite if volume=0 to avoid breaking simple CSVs
                if candle_volume_usd > 0:
                    max_size_usd = max_liquidity_allowed
                    implied_notional = self.capital * pct_size * getattr(self, 'leverage', 1.0)
                    
                    if implied_notional > max_size_usd:
                        # Cap size to available liquidity
                        if self.verbose: 
                            print(f"[Sandbox] ðŸ’§ LIQUIDITY CONSTRAINED: Wanted ${implied_notional:.0f}, Capped at ${max_size_usd:.0f} (1% Vol)")
                        
                        # Recalculate pct_size to fit liquidity
                        # pct_size = (max_size_usd / leverage) / capital
                        new_pct = (max_size_usd / getattr(self, 'leverage', 1.0)) / self.capital
                        pct_size = min(pct_size, new_pct)

                # --- RISK MANAGEMENT CAP (2% Rule) ---
                # "No single trade > 2% risk"
                # Risk = (Entry - SL) * Qty
                # MaxRisk = Capital * 0.02
                # If SL is defined, limit Qty based on Risk.
                # If SL is NOT defined, limit Notional Size? Or assume 100% risk?
                
                max_risk_amount = self.capital * 0.02
                
                if signal.stop_loss and signal.stop_loss > 0:
                     # Stop Distance % = signal.stop_loss (e.g. 0.05 for 5%)
                     # RiskPerUnit = Price * StopPct
                     # Qty = MaxRisk / RiskPerUnit
                     # Qty = MaxRisk / (Price * StopPct)
                     
                     risk_per_unit_pct = signal.stop_loss
                     if risk_per_unit_pct > 0:
                         max_notional_risk_based = max_risk_amount / risk_per_unit_pct
                         # Cap the size so Notional <= max_notional_risk_based
                         
                         implied_size_pct = max_notional_risk_based / self.capital
                         pct_size = min(pct_size, implied_size_pct)
                
                # Default Leverage Clamp (1x for safety if not specified)
                leverage = getattr(self, 'leverage', 1.0) 
                
                margin_to_spend = self.capital * pct_size

                # Apply Fee on Notional
                notional_value = margin_to_spend * leverage
                
                fee = notional_value * self.fee_rate
                
                # Check if we can afford fee
                if (margin_to_spend + fee) > self.capital:
                    margin_to_spend = self.capital - fee
                    notional_value = margin_to_spend * leverage
                
                # Final Size Check
                if margin_to_spend < 0: margin_to_spend = 0
                
                qty = notional_value / exec_price
                
                if self.verbose:
                    print(f"[Sandbox] ðŸ›’ BUY Exec: Price ${exec_price:.2f} | Size {pct_size*100:.1f}% | Lev {leverage}x | Fee ${fee:.2f}")

                if qty > 0:
                    self.capital -= (margin_to_spend + fee) # Deduct Margin + Fee
                    
                    if is_stacking:
                        # Weighted Average Entry Price
                        old_val = self.inventory * self.entry_price
                        new_val = qty * exec_price
                        total_qty = self.inventory + qty
                        self.entry_price = (old_val + new_val) / total_qty
                        self.inventory = total_qty
                        self.margin_locked += margin_to_spend
                    else:
                        # New Position
                        self.inventory = qty
                        self.entry_price = exec_price
                        self.margin_locked = margin_to_spend 
                    
                    # Store Active Stops
                    if signal.stop_loss:
                         self.active_stop_loss = self.entry_price * (1 - signal.stop_loss)
                    else:
                         self.active_stop_loss = None
                    
                    # Store Entry Fee
                    self.active_entry_fee += fee 

                    self.trades.append({
                        'id': len(self.trades)+1,
                        'type': 'BUY' if not is_stacking else 'STACK',
                        'time': row['timestamp'],
                        'price': exec_price,
                        'qty': qty,
                        'leverage': leverage,
                        'reason': signal.reason
                    })

        elif signal.direction == 'SELL':
             if self.inventory > 0:
                 # Check PnL
                 avg_price = self.entry_price
                 pnl_pct = (exec_price - avg_price) / avg_price
                 
                 # Logic: Only sell if PnL > FEE (No churning) or STOP LOSS
                 # Or if signal is STRONG (-2 'force')
                 if pnl_pct > 0.002 or signal.strength < -1.5 or pnl_pct < -0.01:
                      self._force_exit(row['timestamp'], exec_price, signal.reason)

    def _force_exit(self, timestamp, exec_price, reason):
        if self.inventory > 0:
            qty = self.inventory
            entry_price = self.entry_price
            
            # PnL Calculation
            raw_pnl = (exec_price - entry_price) * qty
            gross_value = qty * exec_price
            fee = gross_value * self.fee_rate
            
            # Return Margin
            margin_returned = getattr(self, 'margin_locked', 0.0)
            # Return Margin
            margin_returned = getattr(self, 'margin_locked', 0.0)
            entry_fee = getattr(self, 'active_entry_fee', 0.0) # Cumulative fee logic
            self.active_entry_fee = 0.0 # Reset after exit
            
            net_return = margin_returned + raw_pnl - fee
            
            # Prevent Negative Capital (Bankruptcy)
            if net_return < 0: net_return = 0
            
            self.capital += net_return
            
            # Record Trade
            # True Net PnL = Raw - ExitFee - EntryFee
            true_pnl = raw_pnl - fee - entry_fee
            
            self.trades.append({
                'id': len(self.trades)+1,
                'type': 'SELL',
                'time': timestamp,
                'price': exec_price,
                'qty': qty,
                'pnl': true_pnl,
                'pnl_pct': (true_pnl / margin_returned) if margin_returned > 0 else 0,
                'reason': reason
            })
            
            # Reset State
            self.inventory = 0
            self.entry_price = 0
            self.margin_locked = 0
            self.active_stop_loss = None
            
            # Bankruptcy Check in Run Loop will catch capital < 0
                

                


    def calculate_entropy(self):
        """
        Calculate Shannon Entropy of the strategy's returns.
        High Entropy = Chaotic/Unpredictable (Bad)
        Low Entropy = Stable/Predictable (Good)
        From AEHML Framework: "Quantifying internal order or disorder."
        """
        import numpy as np
        
        if not self.equity_curve or len(self.equity_curve) < 2:
            return 0.0
            
        # Extract Equity Series
        equity_series = [e['equity'] for e in self.equity_curve]
        
        # Calculate Percentage Returns (Candle-to-Candle)
        # Using numpy for speed if available, else plain list comp
        try:
            arr = np.array(equity_series)
            returns = np.diff(arr) / arr[:-1]
            returns = returns[returns != 0] # Filter zeroes (idle checks)
        except:
             # Fallback
             returns = []
             for i in range(1, len(equity_series)):
                 prev = equity_series[i-1]
                 curr = equity_series[i]
                 if prev > 0:
                     ret = (curr - prev) / prev
                     if ret != 0: returns.append(ret)
        
        if len(returns) < 10:
            return 0.0 # Not enough data

        # OPTIMIZATION: Rust Engine
        # If available, delegate heavy lifting to Rust (~50x speedup)
        try:
            if 'holonic_speed' in globals():
                # returns is numpy array, convert to list for PyO3
                return holonic_speed.calculate_shannon_entropy(returns.tolist())
        except Exception as e:
            # Fallback to Python if Rust fails (e.g. type mismatch)
            pass
            
        # Fallback: Python/Numpy Implementation
        hist, _ = np.histogram(returns, bins='auto', density=True)
        
        # Normalize to Sum=1 (Probabilities)
        # Note: 'density=True' makes area=1. To get probabilities p_i, we need hist * bin_width
        # Simplified: Use counts for pure Shannon
        counts, _ = np.histogram(returns, bins='auto', density=False)
        probs = counts / np.sum(counts)
        probs = probs[probs > 0] # Filter zeroes for log
        
        entropy = -np.sum(probs * np.log2(probs))
        return float(entropy)
        
    def get_equity(self, current_price):
        if self.inventory > 0:
            # Equity = Free Cash + Asset Value - Debt
            # Debt = Cost Basis (Notional) - Margin Posted
            # We track 'margin_locked' but need original notional cost for debt
            # Debt is constant (unless we deleverage).
            # Debt = (self.inventory * self.entry_price) - self.margin_locked
            
            # Note: This assumes infinite liquidity/borrowing.
            debt = (self.inventory * self.entry_price) - self.margin_locked
            asset_val = self.inventory * current_price
            return self.capital + asset_val - debt
        return self.capital

    def report(self):
        """Print performance report."""
        print("\n" + "="*40)
        print(f"ðŸ¥ª SANDBOX REPORT: {self.strategy.name}")
        print("="*40)
        
        final_eq = self.equity_curve[-1]['equity'] if self.equity_curve else self.initial_capital
        total_ret = ((final_eq - self.initial_capital) / self.initial_capital) * 100
        
        # --- BENCHMARK: BUY & HOLD ---
        if self.df is not None and not self.df.empty:
            first_price = self.df['close'].iloc[0]
            last_price = self.df['close'].iloc[-1]
            bnh_ret = ((last_price - first_price) / first_price) * 100
            print(f"ðŸ“ˆ Buy & Hold: {bnh_ret:+.2f}% (${self.initial_capital * (1+bnh_ret/100):.2f})")
        else:
            bnh_ret = 0.0
            
        print(f"ðŸ¤– Strategy:  {total_ret:+.2f}% (${final_eq:.2f})")
        print(f"ðŸ“Š Trades: {len(self.trades)}")
        
        wins = [t for t in self.trades if (t['type'] == 'SELL' or t['type'] == 'LIQUIDATION') and t.get('pnl', 0) > 0]
        losses = [t for t in self.trades if (t['type'] == 'SELL' or t['type'] == 'LIQUIDATION') and t.get('pnl', 0) <= 0]
        
        closed_trades = wins + losses
        win_rate = (len(wins) / len(closed_trades) * 100) if closed_trades else 0
        print(f"âœ… Win Rate: {win_rate:.1f}% ({len(wins)} W / {len(losses)} L)")
        
        # Calculate Max Drawdown
        mdd = 0.0
        peak = self.initial_capital
        for point in self.equity_curve:
            if point['equity'] > peak: peak = point['equity']
            dd = (peak - point['equity']) / peak
            if dd > mdd: mdd = dd
            
        print(f"ðŸ“‰ Max Drawdown: {mdd*100:.1f}%")
        
        if bnh_ret > 0 and total_ret < bnh_ret:
             print("âš ï¸  Strategy Underperforming Benchmark (Buy & Hold)")
        if bnh_ret < 0 and total_ret > bnh_ret:
             print("ðŸ›¡ï¸  Strategy Outperforming Bear Market")
        
        print(f"Final Equity:   ${final_eq:.2f}")
        print(f"Total Return:   {total_ret:+.2f}%")
        print(f"Trades:         {len(wins) + len(losses)}")
        print(f"Win Rate:       {win_rate:.1f}%")
        print(f"Condition:      Fee={self.fee_rate*100}%, Slip={self.slippage*100}%")
        print("="*40 + "\n")

    def _run_warp_speed(self):
        """
        ðŸš€ WARP VELOCITY: Use Rust Engine for 10,000x Speed.
        Only works for Gene-based Strategies (EvoStrategy) that can be vectorized.
        """
        if self.df is None or self.df.empty: return

        # 0. Complexity Guard
        # Rust engine (v1) assumes 100% sizing and cannot handle Satellite logic or Trailing Stops correctly.
        # Fallback to Python for these cases.
        raise NotImplementedError("Rust Engine v1 lacks Sizing/Satellite logic. Using Python.")

        # 1. Vectorize Strategy (Gene Thresholds)
        try:
            import holonic_speed
            import numpy as np
            genome = self.strategy.genome
            
            # --- Extract Risk Params for Rust ---
            leverage = float(genome.get('leverage_cap', 1.0))
            stop_loss = float(genome.get('stop_loss', 0.0))
            take_profit = float(genome.get('take_profit', 0.0))
            
            # Calculate Trailing Params from R-Multiples
            # Calculate Trailing Params from R-Muliples
            # Activation = SL% * R-Multiple
            trail_act = stop_loss * float(genome.get('trailing_activation', 2.0))
            trail_dist = stop_loss * float(genome.get('trailing_distance', 0.3))
            
            signals = np.zeros(len(self.df), dtype=np.int32)
            buy_mask = (self.df['rsi'] < genome['rsi_buy']).values
            signals[buy_mask] = 1
            sell_mask = (self.df['rsi'] > genome['rsi_sell']).values
            signals[sell_mask] = -1
            
            timestamps = self.df['timestamp'].astype(np.int64) // 10**6 
            
            balance, rust_trades = holonic_speed.run_backtest_fast(
                timestamps.tolist(),
                self.df['open'].values.tolist(),
                self.df['high'].values.tolist(),
                self.df['low'].values.tolist(),
                self.df['close'].values.tolist(),
                signals.tolist(),
                self.initial_capital,
                self.fee_rate,
                leverage,
                stop_loss,
                take_profit,
                trail_act,
                trail_dist
            )
            
            self.capital = balance
            self.inventory = 0
            self.trades = []
            self.equity_curve = [{'timestamp': self.df['timestamp'].iloc[0], 'equity': self.initial_capital}]
            
            running_balance = self.initial_capital
            import pandas as pd
            
            for t in rust_trades:
                pnl = t.pnl
                running_balance += pnl
                exit_ts = pd.Timestamp(t.exit_time, unit='ms')
                
                self.trades.append({
                    'id': len(self.trades)+1,
                    'type': 'SELL',
                    'pnl': pnl,
                    'roi': t.roi,
                    'time': exit_ts
                })
                self.equity_curve.append({'timestamp': exit_ts, 'equity': running_balance})
                
            if self.verbose: 
                print(f"[Sandbox] ðŸš€ Warp Speed Complete. Trades: {len(self.trades)} | Balance: ${self.capital:.2f}")

        except Exception as e:
            if self.verbose: print(f"[Sandbox] âš ï¸ Warp Speed Failed ({e}). Refactoring to Python.")
            raise e

