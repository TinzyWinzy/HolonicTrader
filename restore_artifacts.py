
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os
import glob

def restore_scaler():
    print("re-generating scaler.pkl...")
    
    # 1. Load Data
    data_dir = 'market_data'
    files = glob.glob(os.path.join(data_dir, '*_1h.csv'))
    
    if not files:
        print("Error: No market data found to fit scaler.")
        return

    all_prices = []
    # We fit on BTC primarily as it defines the macro range, or all?
    # Given the multi-asset nature and raw price input suspicion, 
    # fitting on ALL data gives the widest range (though arguably bad ML).
    # But fitting on just BTC might make ADA (small numbers) map to -1.
    # Fitting on *log prices* might be better but the code passes raw prices.
    
    # Heuristic: Fit on ALL available prices to handle the full range of 'Allowed Assets'
    print(f"Loading data from {len(files)} files...")
    for f in files:
        try:
            df = pd.read_csv(f)
            if 'close' in df.columns:
                all_prices.extend(df['close'].values)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            
    if not all_prices:
        print("No price data loaded.")
        return
        
    prices_array = np.array(all_prices).reshape(-1, 1)
    
    # 2. Fit Scaler
    scaler = StandardScaler()
    scaler.fit(prices_array)
    
    # 3. Save
    joblib.dump(scaler, 'scaler.pkl')
    print(f"✅ scaler.pkl saved. Mean: {scaler.mean_[0]:.2f}, Scale: {scaler.scale_[0]:.2f}")

if __name__ == "__main__":
    restore_scaler()
