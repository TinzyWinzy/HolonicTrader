
import os
import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
import config

def download_missing_data():
    print("🚀 HISTORICAL DATA DOWNLOADER (DQN WARMUP)")
    print("="*40)
    
    exchange = ccxt.kucoin()
    data_dir = 'market_data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    # Standard timeframe for the bot
    timeframe = '1h'
    
    # Symbols from config
    symbols = config.ALLOWED_ASSETS
    
    # Map for missing files
    symbol_map = {
        'ADA/USDT': 'ADAUSDT_1h.csv',
        'BTC/USDT': 'BTCUSD_1h.csv',
        'DOGE/USDT': 'DOGEUSDT_1h.csv',
        'SUI/USDT': 'SUIUSDT_1h.csv',
        'XRP/USDT': 'XRPUSDT_1h.csv',
        'BNB/USDT': 'BNBUSDT_1h.csv',
        'ETH/USDT': 'ETHUSDT_1h.csv',
        'SOL/USDT': 'SOLUSDT_1h.csv',
        'SHIB/USDT': 'SHIBUSDT_1h.csv',
        'PAXG/USDT': 'PAXGUSDT_1h.csv',
        'LTC/USDT': 'LTCUSDT_1h.csv',
        'LINK/USDT': 'LINKUSDT_1h.csv',
        'XMR/USDT': 'XMRUSDT_1h.csv',
        'ALGO/USDT': 'ALGOUSDT_1h.csv',
        'UNI/USDT': 'UNIUSDT_1h.csv',
        'AAVE/USDT': 'AAVEUSDT_1h.csv'
    }
    
    for symbol in symbols:
        filename = symbol_map.get(symbol)
        if not filename:
            clean_symbol = symbol.replace('/', '').replace(':', '')
            filename = f"{clean_symbol}_1h.csv"
            
        filepath = os.path.join(data_dir, filename)
        
        if os.path.exists(filepath):
            print(f"✅ {symbol} already exists. Skipping.")
            continue
            
        print(f"📥 Downloading {symbol}...", end=" ", flush=True)
        
        try:
            # Fetch last 1000 hours (~41 days)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=1000)
            if not ohlcv:
                print("FAILED (No data returned)")
                continue
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            df.to_csv(filepath, index=False)
            print(f"DONE ({len(df)} rows)")
            
            # Rate limit respect
            time.sleep(1)
            
        except Exception as e:
            print(f"ERR: {e}")

if __name__ == "__main__":
    download_missing_data()
