
import React, { useState } from 'react';
import { Crosshair, ShoppingCart } from 'lucide-react';
import clsx from 'clsx';

const TradePanel = () => {
    const [symbol, setSymbol] = useState('BTC/USDT');
    const [amount, setAmount] = useState('100'); // Default $100
    const [leverage, setLeverage] = useState(5);
    const [side, setSide] = useState('BUY');
    const [loading, setLoading] = useState(false);
    const [status, setStatus] = useState('');

    const executeTrade = async () => {
        if (!amount || isNaN(amount) || Number(amount) <= 0) {
            setStatus('Invalid Amount');
            return;
        }

        setLoading(true);
        setStatus('Sending...');

        try {
            const res = await fetch('/api/trade', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    symbol: symbol.toUpperCase(),
                    action: side,
                    quantity: Number(amount),
                    is_usd: true, // We send USD amount
                    leverage: Number(leverage)
                })
            });
            const data = await res.json();

            if (data.status === 'ok') {
                setStatus('Success!');
                setTimeout(() => setStatus(''), 3000);
            } else {
                setStatus('Error: ' + data.message);
            }
        } catch (e) {
            setStatus('Network Error');
        }
        setLoading(false);
    };

    return (
        <div className="bg-holon-card rounded-xl border border-slate-700/50 p-4 transition-all hover:bg-slate-800/40">
            <div className="flex items-center gap-2 mb-4 text-holon-accent border-b border-slate-700/50 pb-2">
                <Crosshair size={18} />
                <h3 className="font-orbitron text-sm tracking-wider font-bold">MANUAL OVERRIDE</h3>
            </div>

            <div className="space-y-3">
                {/* Symbol Input */}
                <div>
                    <label className="text-[10px] uppercase text-holon-dim font-mono block mb-1">Asset</label>
                    <input
                        type="text"
                        value={symbol}
                        onChange={(e) => setSymbol(e.target.value)}
                        className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs font-mono text-white focus:border-holon-accent focus:outline-none uppercase"
                        placeholder="BTC/USDT"
                    />
                </div>

                {/* Side Toggle */}
                <div className="grid grid-cols-2 gap-2">
                    <button
                        onClick={() => setSide('BUY')}
                        className={clsx(
                            "py-2 rounded font-bold text-xs font-orbitron transition-all",
                            side === 'BUY' ? "bg-emerald-600 text-white shadow-lg" : "bg-slate-800 text-slate-500 hover:bg-slate-700"
                        )}
                    >
                        LONG
                    </button>
                    <button
                        onClick={() => setSide('SELL')}
                        className={clsx(
                            "py-2 rounded font-bold text-xs font-orbitron transition-all",
                            side === 'SELL' ? "bg-red-600 text-white shadow-lg" : "bg-slate-800 text-slate-500 hover:bg-slate-700"
                        )}
                    >
                        SHORT
                    </button>
                </div>

                {/* Amount & Leverage Row */}
                <div className="grid grid-cols-2 gap-2">
                    <div>
                        <label className="text-[10px] uppercase text-holon-dim font-mono block mb-1">Size ($USD)</label>
                        <div className="relative">
                            <span className="absolute left-2 top-1.5 text-slate-500 text-xs">$</span>
                            <input
                                type="number"
                                value={amount}
                                onChange={(e) => setAmount(e.target.value)}
                                className="w-full bg-slate-900 border border-slate-700 rounded pl-5 pr-2 py-1 text-xs font-mono text-white focus:border-holon-accent focus:outline-none"
                            />
                        </div>
                    </div>
                    <div>
                        <label className="text-[10px] uppercase text-holon-dim font-mono block mb-1">Lev (x)</label>
                        <input
                            type="number"
                            min="1" max="50"
                            value={leverage}
                            onChange={(e) => setLeverage(e.target.value)}
                            className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs font-mono text-white focus:border-holon-accent focus:outline-none"
                        />
                    </div>
                </div>

                {/* Confirm Button */}
                <button
                    onClick={executeTrade}
                    disabled={loading}
                    className={clsx(
                        "w-full flex items-center justify-center gap-2 py-3 rounded font-orbitron text-xs font-bold transition-all mt-2",
                        loading
                            ? "bg-slate-700 text-slate-400 cursor-not-allowed"
                            : side === 'BUY'
                                ? "bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-900/40 shadow-lg active:scale-95"
                                : "bg-red-600 hover:bg-red-500 text-white shadow-red-900/40 shadow-lg active:scale-95"
                    )}
                >
                    {loading ? 'EXECUTING...' : <><ShoppingCart size={14} /> EXECUTE {side}</>}
                </button>

                {status && (
                    <p className={clsx(
                        "text-[10px] text-center font-mono mt-1 break-words",
                        status.includes('Error') ? "text-red-400" : "text-emerald-400"
                    )}>
                        {status}
                    </p>
                )}
            </div>
        </div>
    );
};

export default TradePanel;
