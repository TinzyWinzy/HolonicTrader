// ─── Mock OHLC Data Generator ──────────────────────────────────────────────────
// Generates realistic candlestick data for chart development.

export interface OHLCCandle {
    time: number; // Unix timestamp (seconds)
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}

function randomWalk(base: number, volatility: number): number {
    return base * (1 + (Math.random() - 0.48) * volatility);
}

export function generateMockOHLC(
    symbol: string = 'BTC/USD',
    count: number = 200,
    interval: number = 3600, // 1h candles in seconds
): OHLCCandle[] {
    const candles: OHLCCandle[] = [];

    // Base prices for different symbols
    const basePrices: Record<string, number> = {
        'BTC/USD': 97500,
        'ETH/USD': 2650,
        'SOL/USD': 195,
        'XRP/USD': 2.45,
        'DOGE/USD': 0.265,
        'PAXG/USD': 2870,
    };

    let price = basePrices[symbol] || 100;
    const vol = symbol.includes('BTC') ? 0.008 : symbol.includes('ETH') ? 0.012 : 0.015;
    const now = Math.floor(Date.now() / 1000);
    const startTime = now - count * interval;

    for (let i = 0; i < count; i++) {
        const open = price;
        const close = randomWalk(open, vol);
        const high = Math.max(open, close) * (1 + Math.random() * vol * 0.5);
        const low = Math.min(open, close) * (1 - Math.random() * vol * 0.5);
        const volume = Math.floor(Math.random() * 1000000 + 100000);

        candles.push({
            time: startTime + i * interval,
            open: parseFloat(open.toFixed(2)),
            high: parseFloat(high.toFixed(2)),
            low: parseFloat(low.toFixed(2)),
            close: parseFloat(close.toFixed(2)),
            volume,
        });

        price = close;
    }

    return candles;
}

export function generateMockVolumeData(candles: OHLCCandle[]): { time: number; value: number; color: string }[] {
    return candles.map((c) => ({
        time: c.time,
        value: c.volume,
        color: c.close >= c.open ? 'rgba(0, 230, 118, 0.3)' : 'rgba(255, 23, 68, 0.3)',
    }));
}
