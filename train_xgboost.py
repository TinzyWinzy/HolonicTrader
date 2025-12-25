import pandas as pd
import numpy as np
import os
import glob
import xgboost as xgb
import joblib
from HolonicTrader.agent_oracle import EntryOracleHolon

# Configuration
DATA_DIR = 'market_data'
MODEL_PATH = 'xgboost_model.json'

def compute_indicators(df):
    """Compute technical features for Stacked XGBoost."""
    # 1. Standard Technicals
    # RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    rolling_mean = df['close'].rolling(window=20).mean()
    rolling_std = df['close'].rolling(window=20).std()
    df['bb_pct_b'] = (df['close'] - (rolling_mean - 2 * rolling_std)) / (4 * rolling_std)
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Volatility
    df['volatility'] = df['close'].pct_change().rolling(14).std()
    
    # 2. TARGET: Is price higher in 3 candles?
    df['target'] = (df['close'].shift(-3) > df['close']).astype(int)
    
    return df

def generate_stacked_dataset(file_list=None):
    """Run LSTM inference on historical data to create the stacked feature set."""
    oracle = EntryOracleHolon()
    if oracle.model is None:
        print("ERROR: LSTM Brain not found. Cannot train V4.")
        return None
        
    # Final dataset expansion to 12 assets for live-readiness
    if file_list is None:
        top_symbols = [
            'BTCUSD_1h.csv', 'ETHUSDT_1h.csv', 'ADAUSDT_1h.csv', 'XRPUSDT_1h.csv',
            'SOLUSDT_1h.csv', 'DOGEUSDT_1h.csv', 'SUIUSDT_1h.csv', 'BNBUSDT_1h.csv',
            'LINKUSDT_1h.csv', 'LTCUSDT_1h.csv', 'UNIUSDT_1h.csv', 'AAVEUSDT_1h.csv'
        ]
        file_list = [os.path.join(DATA_DIR, f) for f in top_symbols if os.path.exists(os.path.join(DATA_DIR, f))]
    
    datasets = []
    
    print(f"Generating Stacked Dataset (Optimized) from {len(file_list)} files...")
    for f in file_list:
        df = pd.read_csv(f)
        if len(df) < 150: continue
        
        df = compute_indicators(df)
        
        # LSTM Inference (Temporal State Probe) - Optimized with STEP
        probs = [0.5] * len(df)
        step = 10 # Only infer every 10 candles, then forward fill
        
        for i in range(60, len(df), step):
            # Predict using the window ending at i
            window = df['close'].iloc[i-60:i]
            p = oracle.predict_trend_lstm(window)
            # Fill the next 'step' rows with this prediction
            for j in range(i, min(i + step, len(df))):
                probs[j] = p
            
        df['lstm_prob'] = probs
        datasets.append(df.iloc[60:-3].dropna()) # Drop padding and target lookahead
        print(f"  Processed {os.path.basename(f)}: {len(df)} samples")
        
    if not datasets:
        return None
        
    return pd.concat(datasets)

def train_model(train_df, params=None):
    """Train XGBoost model on the provided dataframe."""
    features = ['lstm_prob', 'rsi', 'bb_pct_b', 'macd_hist', 'volatility']
    X = train_df[features]
    y = train_df['target']
    
    dtrain = xgb.DMatrix(X, label=y)
    
    if params is None:
        params = {
            'max_depth': 6,
            'eta': 0.05,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
    
    model = xgb.train(params, dtrain, num_boost_round=150)
    return model

def train_stacked_holon(save_path=MODEL_PATH):
    full_df = generate_stacked_dataset()
    if full_df is None: return
    
    print(f"\nTraining Monolith-V4 (Stacked Holon) on {len(full_df)} samples...")
    
    model = train_model(full_df)
    
    # Save model
    model.save_model(save_path)
    print(f"Model saved to {save_path}")
    
    # Feature Importance
    importance = model.get_score(importance_type='gain')
    print("Feature Importance:", importance)

if __name__ == "__main__":
    train_stacked_holon()
