import ccxt
import ccxt.pro
import config
import logging

logging.basicConfig(level=logging.INFO)

exchange = ccxt.krakenfutures({
    'apiKey': config.KRAKEN_FUTURES_API_KEY or config.API_KEY,
    'secret': config.KRAKEN_FUTURES_PRIVATE_KEY or config.API_SECRET,
})

try:
    print("--- Fetching Ledger ---")
    if exchange.has['fetchLedger']:
        ledger = exchange.fetch_ledger(limit=10)
        print(f"Ledger entries: {len(ledger)}")
        for l in ledger:
            print(f"[{l.get('timestamp')}] {l.get('type')} {l.get('amount')} {l.get('currency')} - {l.get('info')}")
    else:
        print("fetchLedger not supported natively by CCXT for krakenfutures.")
        
    print("\n--- Fetching MyTrades ---")
    if exchange.has['fetchMyTrades']:
        trades = exchange.fetch_my_trades(limit=10)
        print(f"Trades retrieved: {len(trades)}")
        for t in trades[:3]:
            print(f"[{t.get('datetime')}] {t.get('symbol')} {t.get('side')} {t.get('amount')} @ {t.get('price')} (Fee: {t.get('fee')})")
            
    print("\n--- Raw Request for Fills/Funding ---")
    try:
        # Kraken Futures specific private history endpoint
        raw_fills = exchange.privateGetFills()
        fills = raw_fills.get('fills', [])
        funding_fills = [f for f in fills if f.get('fillType') == 'funding']
        print(f"Raw Private Fills endpoint retrieved. Total: {len(fills)}. Funding type: {len(funding_fills)}")
        if funding_fills:
            print(f"Sample funding fill: {funding_fills[0]}")
    except Exception as e:
        print(f"Raw fills endpoint error: {e}")
        
except Exception as e:
    print(f"Error: {e}")
