import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import time
from datetime import datetime
from scipy.stats import linregress

# Import Holonic System
from HolonicTrader.agent_ppo import PPOHolon
from HolonicTrader.agent_entropy import EntropyHolon
from HolonicTrader.agent_oracle import EntryOracleHolon
from HolonicTrader.agent_governor import GovernorHolon
from HolonicTrader.agent_executor import ExecutorHolon, TradeSignal
import config

# Hyperparams
EPISODES = 20
BATCH_SIZE = 64
DRAWDOWN_PENALTY_MULT = config.PPO_REWARD_DRAWDOWN_PENALTY

def get_ppo_state(regime, entropy, win_rate, atr_ratio, drawdown, margin):
    """Matches the state vector in agent_trader.py"""
    regime_map = {'ORDERED': 0.0, 'TRANSITION': 0.5, 'CHAOTIC': 1.0}
    return np.array([
        regime_map.get(regime, 0.5),
        entropy,
        win_rate,
        atr_ratio,
        drawdown,
        margin
    ], dtype=np.float32)

def run_training():
    print("==================================================")
    print("   SOVEREIGN BRAIN - PPO HISTORICAL TRAINING      ")
    print("==================================================")
    
    # 0. Optimization: Disable GPU if not needed or avoid TF noise
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # 1. Initialize Agents
    ppo = PPOHolon(batch_size=BATCH_SIZE)
    entropy_agent = EntropyHolon()
    oracle = EntryOracleHolon()
    
    assets = config.ALLOWED_ASSETS
    
    for ep in range(EPISODES):
        # Pick a random asset for this episode to generalize
        symbol = str(np.random.choice(assets))
        clean_symbol = symbol.replace('/', '').replace(':', '')
        path = f"market_data/{clean_symbol}_1h.csv"
        # Special case for BTC
        if symbol == 'BTC/USDT' and not os.path.exists(path): path = 'market_data/BTCUSD_1h.csv'
        
        if not os.path.exists(path):
            continue
            
        print(f">> Episode {ep+1}/{EPISODES} | Training on {symbol}...")
        df = pd.read_csv(path)
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Reset Episode Agents
        governor = GovernorHolon(initial_balance=config.INITIAL_CAPITAL)
        executor = ExecutorHolon(initial_capital=config.INITIAL_CAPITAL)
        
        # State tracking
        data = df.to_dict('records')
        n_steps = len(data)
        total_pnl = 0
        trade_count = 0
        
        # Performance buffer for state
        stats = {'win_rate': 0.5}
        
        # Start Simulation (Skip window for indicators)
        for i in tqdm(range(100, n_steps - 1)):
            row = data[i]
            window = df.iloc[i-50:i+1].copy()
            
            # 1. Entropy & Regime
            ent_val = entropy_agent.calculate_shannon_entropy(window['returns'].tail(24))
            regime = entropy_agent.determine_regime(ent_val)
            
            # 2. Indicators (ATR, BB, OBV etc)
            tr = np.maximum(window['high'] - window['low'], 
                      np.maximum(abs(window['high'] - window['close'].shift(1)), 
                                 abs(window['low'] - window['close'].shift(1))))
            atr = tr.rolling(14).mean().iloc[-1]
            atr_ref = tr.rolling(14).mean().rolling(14).mean().iloc[-1]
            atr_ratio = min(2.0, atr / atr_ref) if atr_ref > 0 else 1.0
            
            ma = window['close'].rolling(20).mean()
            std = window['close'].rolling(20).std()
            bb_vals = {
                'upper': ma.iloc[-1] + 2*std.iloc[-1],
                'middle': ma.iloc[-1],
                'lower': ma.iloc[-1] - 2*std.iloc[-1]
            }
            
            obv = (np.sign(window['close'].diff()).fillna(0) * window['volume']).cumsum()
            obv_slope = 0
            if len(obv) >= 14:
                obv_slope, _, _, _, _ = linregress(np.arange(14), obv.iloc[-14:].values)
            
            # 3. Handle Exits (SL/TP)
            # -----------------------------------------------------------------
            hard_exit = executor.check_stop_loss_take_profit(symbol, row['close'])
            if hard_exit:
                meta = executor.position_metadata.get(symbol, {})
                ppo_info = meta.get('ppo_info')
                
                decision = executor.decide_trade(TradeSignal(symbol, 'SELL', 1.0, row['close']), regime, ent_val)
                pnl_res = executor.execute_transaction(decision, row['close'])
                
                if pnl_res is not None and ppo_info:
                    gov_h = governor.get_portfolio_health()
                    reward = pnl_res - (gov_h['drawdown_pct'] * DRAWDOWN_PENALTY_MULT)
                    
                    ppo.remember(ppo_info['state'], ppo_info['action'], reward, ppo_info['prob'], ppo_info['val'], done=True)
                    trade_count += 1
                    total_pnl += pnl_res
                    
                    # Update win rate
                    stats['win_rate'] = (stats['win_rate'] * 0.9) + (0.1 if pnl_res > 0 else 0)

            # 4. Handle Entry
            # -----------------------------------------------------------------
            if symbol not in executor.held_assets or abs(executor.held_assets[symbol]) < 1e-8:
                oracle.symbol_trends[symbol] = True # Mock GMB contribution
                sig = oracle.analyze_for_entry(symbol, window, bb_vals, obv_slope, 'PREDATOR')
                
                if sig and sig.direction in ['BUY', 'SELL']:
                    gov_h = governor.get_portfolio_health()
                    state = get_ppo_state(regime, ent_val, stats['win_rate'], atr_ratio, 
                                          gov_h['drawdown_pct'], gov_h['margin_utilization'])
                    
                    conviction = ppo.get_conviction(state, training=True)
                    prob = ppo.get_log_prob(state, conviction)
                    val = ppo.get_value(state)
                    
                    approved, quantity, leverage = governor.calc_position_size(symbol, row['close'], atr, atr_ref, conviction)
                    if approved and quantity > 0:
                        sig.size = quantity
                        decision = executor.decide_trade(sig, regime, ent_val)
                        if decision.action != 'HALT':
                            executor.execute_transaction(decision, row['close'])
                            
                            # Ensure position was recorded (it should be if not HALT)
                            if symbol in executor.position_metadata:
                                executor.position_metadata[symbol]['ppo_info'] = {
                                    'state': state, 'action': conviction, 'prob': prob, 'val': val
                                }

            # 5. Heartbeat
            governor.update_balance(executor.get_portfolio_value(row['close']))
            
        # PPO Learning step
        if len(ppo.states) >= BATCH_SIZE:
            aloss, closs = ppo.learn()
            print(f">> [PPO Update] Actor Loss: {aloss:.4f} | Critic Loss: {closs:.4f}")
            
        ppo.save_knowledge()
        print(f">> Episode Summary: trades={trade_count}, pnl_sum={total_pnl:.4f}, bal=${executor.balance_usd:.2f}")

if __name__ == "__main__":
    run_training()
