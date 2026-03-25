import pandas as pd
import numpy as np
import os
import glob
import xgboost as xgb
import joblib

# Configuration
DATA_DIR = 'market_data'
MODEL_PATH = 'xgboost_model.json'

def compute_indicators(df):
    """Compute technical features for XGBoost."""
    # RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands (20, 2)
    rolling_mean = df['close'].rolling(window=20).mean()
    rolling_std = df['close'].rolling(window=20).std()
    df['bb_high'] = rolling_mean + (2 * rolling_std)
    df['bb_low'] = rolling_mean - (2 * rolling_std)
    
    # BB %B
    df['bb_pct_b'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
    
    # Volatility
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(14).std()
    
    # MACD (12, 26, 9)
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df['macd_hist'] = macd - macd_signal
    
    # RVOL (Volume relative to average)
    df['rvol'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # LSTM Probability (Mocking for training)
    # In live, this comes from EntryOracleHolon.predict_trend_lstm
    # Default fallback in oracle is 0.53
    df['lstm_prob'] = 0.53
    
    # TARGET: Is price higher in 3 candles?
    df['target'] = (df['close'].shift(-3) > df['close']).astype(int)
    
    return df.dropna()

def train_ensemble():
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    datasets = []
    
    print(f"Loading {len(all_files)} files for training...")
    for f in all_files:
        df = pd.read_csv(f)
        if len(df) < 100: continue
        df = compute_indicators(df)
        datasets.append(df)
        
    full_df = pd.concat(datasets)
    
    features = ['rsi', 'bb_pct_b', 'volatility', 'macd_hist', 'rvol', 'lstm_prob']
    
    # Expanding dataset with 4 more productive symbols
    top_symbols = [
        'BTCUSD_1h.csv', 'ETHUSDT_1h.csv', 'ADAUSDT_1h.csv', 'XRPUSDT_1h.csv',
        'SOLUSDT_1h.csv', 'DOGEUSDT_1h.csv', 'SUIUSDT_1h.csv', 'BNBUSDT_1h.csv'
    ]
    X = full_df[features]
    y = full_df['target']
    
    print(f"Training XGBoost on {len(X)} samples...")
    
    # Use Native API to avoid sklearn mixin issues in new versions
    dtrain = xgb.DMatrix(X, label=y)
    
    params = {
        'max_depth': 4,
        'eta': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100
    )
    
    # Save model
    model.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Feature Importance
    importance = model.get_score(importance_type='gain')
    print("Feature Importance:", importance)

if __name__ == "__main__":
    train_ensemble()
