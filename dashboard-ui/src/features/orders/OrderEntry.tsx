import React, { useState } from 'react';
import WidgetPanel from '../../components/ui/WidgetPanel';
import WebSocketManager from '../../services/WebSocketManager';
import { useMarketStore } from '../../store/useMarketStore';

const OrderEntry: React.FC = () => {
    const selectedSymbol = useMarketStore((s) => s.selectedSymbol);
    const prices = useMarketStore((s) => s.prices);
    const equity = useMarketStore((s) => s.equity);

    const [symbol, setSymbol] = useState(selectedSymbol || 'BTC/USDT');
    const [side, setSide] = useState<'BUY' | 'SELL'>('BUY');
    const [orderType, setOrderType] = useState<'market' | 'limit' | 'stop'>('market');
    const [quantity, setQuantity] = useState('');
    const [isUsd, setIsUsd] = useState(true);
    const [leverage, setLeverage] = useState(5);
    const [riskPct, setRiskPct] = useState(2);
    const [submitting, setSubmitting] = useState(false);
    const [result, setResult] = useState<{ ok: boolean; msg: string } | null>(null);

    React.useEffect(() => {
        if (selectedSymbol) setSymbol(selectedSymbol);
    }, [selectedSymbol]);

    const currentPrice = prices[symbol] || 0;
    const qtyNum = parseFloat(quantity) || 0;
    const notional = isUsd ? qtyNum : qtyNum * currentPrice;

    // Auto-calc units from USD
    const estimatedUnits = isUsd && currentPrice > 0 ? qtyNum / currentPrice : qtyNum;

    const handleSubmit = async () => {
        if (!quantity || qtyNum <= 0) return;
        setSubmitting(true);
        setResult(null);

        try {
            const ws = WebSocketManager.getInstance();
            const res = await ws.sendCommand('trade', {
                symbol,
                action: side,
                quantity: qtyNum,
                is_usd: isUsd,
                leverage,
            });
            setResult({ ok: res.status === 'ok', msg: res.message || 'Order placed successfully' });

            // Clear result after 3s
            setTimeout(() => setResult(null), 3000);
        } catch (err: any) {
            setResult({ ok: false, msg: err.message || 'Failed to place order' });
        } finally {
            setSubmitting(false);
        }
    };

    return (
        <WidgetPanel title="DIRECT ACCESS" accent={side === 'BUY' ? 'var(--color-long)' : 'var(--color-short)'}>
            {/* Symbol + Side */}
            <div className="flex gap-2 mb-4">
                <input
                    type="text"
                    value={symbol}
                    onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                    className="flex-1 px-3 py-2 rounded bg-[var(--bg-input)] border border-[var(--border-default)] font-mono text-[11px] focus:outline-none focus:border-[var(--accent-crypto)] text-[var(--text-primary)]"
                    placeholder="SYMBOL"
                />
                <div className="flex rounded overflow-hidden border border-[var(--border-default)]">
                    {(['BUY', 'SELL'] as const).map((s) => (
                        <button
                            key={s}
                            onClick={() => setSide(s)}
                            className={`
                                px-4 py-2 text-[10px] font-mono font-bold transition-all
                                ${side === s ? (s === 'BUY' ? 'bg-[rgba(0,230,118,0.2)] text-[var(--color-long)]' : 'bg-[rgba(255,23,68,0.2)] text-[var(--color-short)]') : 'bg-[var(--bg-input)] text-[var(--text-dim)] hover:text-[var(--text-muted)]'}
                            `}
                        >
                            {s}
                        </button>
                    ))}
                </div>
            </div>

            {/* Order Type */}
            <div className="flex gap-1 mb-4 p-1 rounded bg-[var(--bg-input)]">
                {(['market', 'limit', 'stop'] as const).map((t) => (
                    <button
                        key={t}
                        onClick={() => setOrderType(t)}
                        className={`
                            flex-1 py-1.5 text-[9px] font-mono uppercase rounded transition-all
                            ${orderType === t ? 'bg-[var(--bg-elevated)] text-[var(--text-primary)] shadow-sm' : 'text-[var(--text-dim)] hover:text-[var(--text-muted)]'}
                        `}
                    >
                        {t}
                    </button>
                ))}
            </div>

            {/* Quantity */}
            <div className="mb-4 space-y-2">
                <div className="flex items-center justify-between">
                    <span className="text-[9px] font-mono text-[var(--text-dim)]">SIZE</span>
                    <button
                        onClick={() => setIsUsd(!isUsd)}
                        className="text-[9px] font-mono px-2 py-0.5 rounded bg-[var(--bg-elevated)] text-[var(--accent-crypto)] hover:bg-[var(--bg-hover)]"
                    >
                        {isUsd ? 'USD' : 'UNITS'}
                    </button>
                </div>
                <div className="relative">
                    <input
                        type="number"
                        value={quantity}
                        onChange={(e) => setQuantity(e.target.value)}
                        className="w-full px-3 py-2 rounded bg-[var(--bg-input)] border border-[var(--border-default)] font-mono text-[12px] text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-crypto)]"
                        placeholder={isUsd ? '1000.00' : '0.15'}
                    />
                    <span className="absolute right-3 top-2 text-[10px] text-[var(--text-dim)]">
                        {isUsd ? '$' : 'QTY'}
                    </span>
                </div>
            </div>

            {/* Leverage Slider */}
            <div className="mb-6">
                <div className="flex items-center justify-between mb-2">
                    <span className="text-[9px] font-mono text-[var(--text-dim)]">LEVERAGE</span>
                    <span className="text-[10px] font-mono font-bold text-[var(--accent-info)]">
                        {leverage}x
                    </span>
                </div>
                <input
                    type="range"
                    min={1}
                    max={20}
                    value={leverage}
                    onChange={(e) => setLeverage(parseInt(e.target.value))}
                    className="w-full h-1.5 rounded-lg appearance-none cursor-pointer bg-[var(--bg-elevated)]"
                />
            </div>

            {/* Preview Box */}
            <div className="p-3 rounded mb-4 bg-[var(--bg-elevated)] border border-[var(--border-subtle)]">
                <div className="grid grid-cols-2 gap-y-2 text-[10px] font-mono">
                    <div className="text-[var(--text-dim)]">Notional</div>
                    <div className="text-right text-[var(--text-primary)]">${notional.toLocaleString()}</div>

                    <div className="text-[var(--text-dim)]">Est. Units</div>
                    <div className="text-right text-[var(--text-primary)]">{estimatedUnits.toFixed(6)}</div>

                    <div className="text-[var(--text-dim)]">Margin Req.</div>
                    <div className="text-right text-[var(--text-secondary)]">${(notional / leverage).toFixed(2)}</div>
                </div>
            </div>

            {/* Submit Button */}
            <button
                onClick={handleSubmit}
                disabled={submitting || qtyNum <= 0}
                className={`
                    group w-full py-3 rounded font-mono text-[11px] font-bold uppercase tracking-widest transition-all disabled:opacity-40 disabled:cursor-not-allowed
                    ${side === 'BUY'
                        ? 'bg-[rgba(0,230,118,0.1)] text-[var(--color-long)] border border-[var(--color-long)] hover:bg-[var(--color-long)] hover:text-black'
                        : 'bg-[rgba(255,23,68,0.1)] text-[var(--color-short)] border border-[var(--color-short)] hover:bg-[var(--color-short)] hover:text-white'}
                `}
                style={{
                    boxShadow: submitting ? 'none' : `0 0 10px ${side === 'BUY' ? 'rgba(0,230,118,0.2)' : 'rgba(255,23,68,0.2)'}`
                }}
            >
                {submitting ? (
                    <span className="flex items-center justify-center gap-2">
                        <span className="w-3 h-3 border-2 border-current border-t-transparent rounded-full animate-spin" />
                        SENDING...
                    </span>
                ) : (
                    <span>EXECUTE {side}</span>
                )}
            </button>

            {/* Result Message */}
            {result && (
                <div
                    className={`
                        mt-3 px-3 py-2 rounded text-[10px] font-mono text-center border animate-in fade-in slide-in-from-top-1
                        ${result.ok ? 'bg-[rgba(0,230,118,0.1)] border-[var(--color-positive)] text-[var(--color-positive)]' : 'bg-[rgba(255,23,68,0.1)] border-[var(--color-negative)] text-[var(--color-negative)]'}
                    `}
                >
                    {result.msg}
                </div>
            )}
        </WidgetPanel>
    );
};

export default OrderEntry;
