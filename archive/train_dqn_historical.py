
import os
import pandas as pd
import numpy as np
import random
from datetime import datetime
from HolonicTrader.agent_dqn import DeepQLearningHolon
from HolonicTrader.agent_entropy import EntropyHolon
import config

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def train_on_historical():
    print("🚀 DQN HISTORICAL PRE-TRAINING")
    print("="*40)
    
    # 1. Initialize Agents
    dqn = DeepQLearningHolon(epsilon=1.0) # Start with exploration
    entropy_agent = EntropyHolon()
    
    data_dir = 'market_data'
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    total_experiences = 0
    
    for file in files:
        filepath = os.path.join(data_dir, file)
        print(f"\nProcessing {file}...")
        
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        df['rsi'] = calculate_rsi(df['close'])
        df.dropna(inplace=True)
        
        # 2. Simulation Loop
        # We need a window for entropy (50) and a look-forward for reward (24)
        window_size = 50
        reward_lookahead = 24
        
        states_recorded = 0
        
        for i in range(window_size, len(df) - reward_lookahead, 10): # Step by 10 for speed
            # Current State
            sub_df = df.iloc[i-window_size:i]
            returns = sub_df['returns']
            current_entropy = entropy_agent.calculate_shannon_entropy(returns)
            current_rsi = df.iloc[i]['rsi']
            current_return = df.iloc[i]['returns']
            
            # State Vector: [Entropy, RSI, Returns, Constant]
            state = [current_entropy, current_rsi, current_return, 1.0]
            
            # Action: 0=EXECUTE, 1=HALT, 2=REDUCE
            # For training, we sample all actions or follow epsilon-greedy
            action_idx = dqn.get_action_index(dqn.get_action(state))
            
            # 3. Calculate Reward (Look-forward PnL)
            future_prices = df['close'].iloc[i:i+reward_lookahead]
            start_price = df['close'].iloc[i]
            max_pnl = (future_prices.max() - start_price) / start_price
            end_pnl = (future_prices.iloc[-1] - start_price) / start_price
            
            # Multi-objective reward: (Profitability + Risk management)
            if action_idx == 0: # EXECUTE
                reward = end_pnl * 100 # Multiplier for scale
            elif action_idx == 1: # HALT
                reward = 0.0 if end_pnl > 0 else abs(end_pnl) * 50 # Reward staying out of bad trades
            else: # REDUCE
                reward = end_pnl * 50
            
            # Next State
            next_i = i + 10
            if next_i < len(df) - reward_lookahead:
                next_sub_df = df.iloc[next_i-window_size:next_i]
                next_entropy = entropy_agent.calculate_shannon_entropy(next_sub_df['returns'])
                next_rsi = df.iloc[next_i]['rsi']
                next_return = df.iloc[next_i]['returns']
                next_state = [next_entropy, next_rsi, next_return, 1.0]
                done = False
            else:
                next_state = state # Terminal
                done = True
                
            dqn.remember(state, action_idx, reward, next_state, done)
            states_recorded += 1
            
            # Train in batches
            if states_recorded % 5 == 0:
                loss = dqn.replay()
                
        print(f"   Experiences: {states_recorded} | Final Epsilon: {dqn.epsilon:.2f}")
        total_experiences += states_recorded
        
    # 4. Save Knowledge
    dqn.save_knowledge()
    print(f"\n✅ PRE-TRAINING COMPLETE. Total Experiences: {total_experiences}")

if __name__ == "__main__":
    train_on_historical()
