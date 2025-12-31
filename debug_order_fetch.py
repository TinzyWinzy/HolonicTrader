
import ccxt
import os
from dotenv import load_dotenv
import json

# Setup
load_dotenv()
api_key = os.getenv('KRAKEN_FUTURES_API_KEY')
secret = os.getenv('KRAKEN_FUTURES_PRIVATE_KEY')

exchange = ccxt.krakenfutures({
    'apiKey': api_key,
    'secret': secret,
    'enableRateLimit': True,
})

print("Attempting to fetch open orders for XRP/USD:USD...")
try:
    # Use the unified symbol that we know works: 'XRP/USD:USD' or mapped 'PF_XRPUSD'
    # From previous logs: "XRP/USD:USD"
    orders = exchange.fetch_open_orders('XRP/USD:USD')
    print(f"✅ fetch_open_orders success! Count: {len(orders)}")
    if orders:
        print(f"Sample: {orders[0]['id']}")
except Exception as e:
    print(f"❌ fetch_open_orders failed: {e}")

print("\nAttempting to fetch closed orders...")
try:
    closed = exchange.fetch_closed_orders('XRP/USD:USD', limit=5)
    print(f"✅ fetch_closed_orders success! Count: {len(closed)}")
    if closed:
        print(f"Sample: {json.dumps(closed[0], indent=2)}")
except Exception as e:
    print(f"❌ fetch_closed_orders failed: {e}")
