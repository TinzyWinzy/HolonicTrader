import pandas as pd
import numpy as np
import os
from datetime import timedelta
import xgboost as xgb
from HolonicTrader.agent_oracle import EntryOracleHolon
from train_xgboost import generate_stacked_dataset, train_model
from run_backtest import run_backtest
import config

# Configuration
SYMBOL = 'XRP/USDT' # Primary asset for this test
TRAIN_WINDOW_DAYS = 90
TEST_WINDOW_DAYS = 30
DATA_DIR = 'market_data'

def load_full_data():
    """Load and merge data for the Walk-Forward Loop."""
    # Logic similar to generate_stacked_dataset but we need specific control
    # We will use the data loading form train_xgboost for FEATURE GENERATION
    # But we also need raw OHLCV for the BACKTEST simulation.
    
    # 1. Generate FULL Feature Set (with LSTM/Indicators) this is expensive but done once
    print("Generating Master Dataset with Features...")
    # Passing single file to generate_stacked_dataset currently returns a combined DF
    # We need to hack it slightly or just use the list.
    # For WFO, we usually want to train on a basket (Generalization) but test on specific.
    # To keep it simple for now, we train on the target symbol + correlated assets.
    
    # Let's train on the target symbol ONLY for the "Specific" test, or same List as before?
    # User said: "Train on Jan-March. Test on April."
    # Implicitly this usually means train on the SAME asset or the SAME basket.
    # Let's stick to the basket for training (more robust), but we need to track dates carefully.
    
    # ACTUALLY: generate_stacked_dataset returns a concatenated DF. We lose symbol info unless we add it.
    # The current implementation of generate_stacked_dataset does NOT add a symbol column.
    
    # For the Backtest step, we need the OHLCV of the TRADED symbol.
    # For the Training step, we need the Feature Matrix X, y.
    
    return generate_stacked_dataset()

# Revised Plan:
# 1. Define Time Splits.
# 2. For each split:
#    a. Filter Training Files (by date) -> Train Model.
#    b. Filter Test File (Target Symbol by date) -> Run Backtest.

def get_date_range_from_df(df):
    return df['timestamp'].min(), df['timestamp'].max()

def walk_forward_optimization():
    print("🚀 STARTING WALK-FORWARD OPTIMIZATION", flush=True)
    
    # 1. Identify Data Range
    # We need to read the CSVs to find the dates.
    target_file = os.path.join(DATA_DIR, f"{SYMBOL.replace('/','')}_1h.csv")
    if not os.path.exists(target_file):
        print(f"Target file {target_file} not found.")
        return

    full_df = pd.read_csv(target_file)
    full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])
    
    # Calculate Returns for Entropy Agent (Required by run_backtest)
    full_df['returns'] = np.log(full_df['close'] / full_df['close'].shift(1))
    
    start_date = full_df['timestamp'].min()
    end_date = full_df['timestamp'].max()
    
    print(f"Data Range: {start_date} to {end_date}", flush=True)
    
    current_train_start = start_date
    overall_equity_curve = []
    
    # Results Storage
    results = []
    
    # 2. The Loop
    while True:
        train_end = current_train_start + timedelta(days=TRAIN_WINDOW_DAYS)
        test_end = train_end + timedelta(days=TEST_WINDOW_DAYS)
        
        if test_end > end_date:
            break
            
        print(f"\n🌊 WALK-FORWARD WINDOW: Train({current_train_start.date()} -> {train_end.date()}) | Test({train_end.date()} -> {test_end.date()})" )
        
        # --- A. TRAINING PHASE ---
        # We need to pass a filtered list of data to the training function.
        # But wait, `generate_stacked_dataset` reads files directly.
        # We can't easily injection date filters into `read_csv` without modifying it substantially again.
        # OR, we read everything once, and filter in memory.
        
        # Let's do the In-Memory approach. It's cleaner.
        # We need a custom `generate_dataset_for_window` using the code we refactored.
        
        # HACK: We will load the data inside the loop for now. It's slow but safe.
        # Ideally, we load ALL data once, assign timestamps, and filter.
        
        # Let's use a helper that loads raw data, filters by date, THEN computes indicators.
        # This prevents look-ahead in indicators too (like rolling means!) - CRITICAL.
        
        # Step 1: Get Training Data (Basket)
        train_dfs = []
        top_symbols = [
            'BTCUSD_1h.csv', 'ETHUSDT_1h.csv', 'ADAUSDT_1h.csv', 'XRPUSDT_1h.csv' 
            # Reduced list for speed during initial run
        ]
        
        files_found = 0
        for f in top_symbols:
            f_path = os.path.join(DATA_DIR, f)
            if not os.path.exists(f_path): continue
            
            raw_df = pd.read_csv(f_path)
            raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
            
            # FILTER DATES
            mask = (raw_df['timestamp'] >= current_train_start) & (raw_df['timestamp'] < train_end)
            sliced_df = raw_df.loc[mask].copy()
            
            if len(sliced_df) > 100:
                train_dfs.append(sliced_df)
                files_found += 1
        
        if files_found == 0:
            print("  ⚠️ Not enough data for this window. Skipping.")
            current_train_start += timedelta(days=TEST_WINDOW_DAYS)
            continue
            
        # Combine and Process Features (This computes indicators on the SLICE, mimicking real life)
        # Note: Indicator warmup (first 20 candles) will be NaN and dropped. This is correct.
        
        # We need to reuse the logic from `train_xgboost` but applied to these DFs.
        # Since `generate_stacked_dataset` takes a file list, we should make a variant or just manually call compute.
        
        # Let's manually call compute to be safe and fast.
        processed_dfs = []
        oracle = EntryOracleHolon() # Loads immutable LSTM
        
        for df in train_dfs:
            df = compute_indicators_safe(df)
            
            # LSTM Probing
            probs = [0.5] * len(df)
            step = 10
            for i in range(60, len(df), step):
                window = df['close'].iloc[i-60:i]
                p = oracle.predict_trend_lstm(window)
                for j in range(i, min(i + step, len(df))):
                    probs[j] = p
            df['lstm_prob'] = probs
            
            processed_dfs.append(df.iloc[60:-3].dropna())
            
        if not processed_dfs:
            print("  ⚠️ Data evaporated after feature prep. Skipping.")
            current_train_start += timedelta(days=TEST_WINDOW_DAYS)
            continue
            
        final_train_df = pd.concat(processed_dfs)
        
        # TRAIN MODEL
        print(f"  🧠 Training on {len(final_train_df)} samples...")
        model = train_model(final_train_df)
        
        # --- B. TESTING PHASE ---
        # Inject Model into a fresh Oracle
        test_oracle = EntryOracleHolon()
        test_oracle.set_expert_model(model)
        
        # Prepare Test Data (Target Symbol Only)
        # IMPORTANT: We need "Warmup data" before test_start_date so indicators are valid at second 0.
        # We grab [Train_End - 60 days] to [Test_End]
        warmup_start = train_end - timedelta(days=60)
        
        backtest_data = full_df[(full_df['timestamp'] >= warmup_start) & (full_df['timestamp'] < test_end)].copy()
        
        # Run Backtest
        # Note: internal run_backtest logic filters start_date if provided.
        # We pass the full chunk (with warmup) but tell it to Start trading at train_end.
        
        result = run_backtest(
            symbol=SYMBOL,
            start_date=train_end, # Start TRADING here
            end_date=test_end,
            injected_oracle=test_oracle,
            data_df=backtest_data # Pre-loaded data
        )
        
        if result:
            results.append(result)
            print(f"  📊 Result: PnL ${result['total_pnl']:.2f} | Win Rate {result['win_rate']:.1f}%")
        
        # Move Window
        current_train_start += timedelta(days=TEST_WINDOW_DAYS)

    # 3. Consolidation
    print("\n" + "="*60)
    print("WALK-FORWARD ANALYSIS COMPLETED")
    print("="*60)
    
    total_pnl = sum(r['total_pnl'] for r in results)
    avg_win = np.mean([r['win_rate'] for r in results])
    
    print(f"Total Periods: {len(results)}")
    print(f"Cumulative PnL: ${total_pnl:.2f}")
    print(f"Avg Win Rate:   {avg_win:.1f}%")

def compute_indicators_safe(df):
    from train_xgboost import compute_indicators
    return compute_indicators(df)

if __name__ == "__main__":
    walk_forward_optimization()
