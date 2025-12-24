import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from HolonicTrader.agent_oracle import EntryOracleHolon
import pandas as pd
import numpy as np

def verify_ensemble():
    oracle = EntryOracleHolon()
    
    print("\n--- Oracle Brain Status ---")
    print(f"LSTM Loaded: {oracle.model is not None}")
    print(f"XGBoost Loaded: {oracle.xgb_model is not None}")
    
    if oracle.xgb_model:
        print("Testing Stacked XGBoost Inference...")
        dummy_features = {
            'lstm_prob': 0.65,
            'rsi': 30.0,
            'bb_pct_b': 0.1,
            'macd_hist': -0.005,
            'volatility': 0.002
        }
        prob = oracle.predict_trend_xgboost(dummy_features)
        print(f"Sample Stacked XGBoost Prediction: {prob:.4f}")
        
    print("--- Verification Complete ---")

if __name__ == "__main__":
    verify_ensemble()
