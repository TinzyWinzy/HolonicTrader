
import ccxt
import os
from dotenv import load_dotenv
import json

# Setup
load_dotenv()
api_key = os.getenv('KRAKEN_FUTURES_API_KEY')
secret = os.getenv('KRAKEN_FUTURES_PRIVATE_KEY')

print(f"Connecting with Key: {api_key[:6]}...")

exchange = ccxt.krakenfutures({
    'apiKey': api_key,
    'secret': secret,
    'enableRateLimit': True,
})

print("Attempting to fetch POSITIONS...")
try:
    # positions = exchange.fetch_positions() # Some exchanges use this
    # Kraken Futures might need fetch_balance()['info']['accounts']['flex'] or specific endpoint
    # CCXT unified: fetch_positions
    positions = exchange.fetch_positions()
    print(f"✅ fetch_positions success! Count: {len(positions)}")
    
    for p in positions:
        # Filter for active positions
        if float(p['contracts']) > 0 or float(p['info'].get('size', 0)) > 0:
            print(f"Found Position: {p['symbol']} Size: {p['contracts']} Price: {p['entryPrice']}")
            print(json.dumps(p, indent=2))
        
except Exception as e:
    print(f"❌ fetch_positions failed: {e}")
