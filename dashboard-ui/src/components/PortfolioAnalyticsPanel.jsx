
import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Target, Shield, AlertTriangle, Crosshair, BarChart2 } from 'lucide-react';
import clsx from 'clsx';

export default function PortfolioAnalyticsPanel() {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const fetchData = async () => {
        try {
            const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';
            const res = await fetch(`${API_URL}/api/analyze/positions`);
            if (!res.ok) throw new Error('API Error');
            const json = await res.json();
            setData(json);
            setLoading(false);
        } catch (e) {
            setError(e.message);
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchData();
        const interval = setInterval(fetchData, 5000); // Poll every 5s
        return () => clearInterval(interval);
    }, []);

    if (loading) return <div className="p-4 text-holon-dim animate-pulse">Initializing Analyst...</div>;
    if (error || !data) return <div className="p-4 text-red-400">Analyst Offline</div>;

    const { macro, metrics, positions } = data;
    const isBullish = macro.regime === 'BULLISH';
    const isBearish = macro.regime === 'BEARISH';

    return (
        <div className="flex flex-col gap-4 h-full">
            {/* 1. Macro Regime Banner */}
            <div className={clsx(
                "rounded-xl p-4 border flex items-center justify-between",
                isBullish ? "bg-emerald-900/20 border-emerald-500/30" :
                    isBearish ? "bg-red-900/20 border-red-500/30" :
                        "bg-slate-800/50 border-slate-700"
            )}>
                <div className="flex items-center gap-3">
                    <div className={clsx("p-2 rounded-lg", isBullish ? "bg-emerald-500/20 text-emerald-400" : isBearish ? "bg-red-500/20 text-red-400" : "bg-slate-700 text-slate-400")}>
                        {isBullish ? <TrendingUp size={24} /> : isBearish ? <TrendingDown size={24} /> : <BarChart2 size={24} />}
                    </div>
                    <div>
                        <p className="text-[10px] uppercase tracking-wider text-holon-dim">MACRO REGIME</p>
                        <h2 className={clsx("text-xl font-bold font-orbitron", isBullish ? "text-emerald-400" : isBearish ? "text-red-400" : "text-slate-300")}>
                            {macro.regime}
                        </h2>
                        <p className="text-xs text-slate-500">Bias Score: {macro.bias_score}</p>
                    </div>
                </div>

                {/* Exposure Stats */}
                <div className="flex gap-6 text-right">
                    <div>
                        <p className="text-[10px] text-holon-dim">NET EXPOSURE</p>
                        <p className="text-lg font-mono font-bold text-blue-400">${metrics.net_exposure}</p>
                    </div>
                    <div>
                        <p className="text-[10px] text-holon-dim">LEVERAGE</p>
                        <p className={clsx("text-lg font-mono font-bold", metrics.leverage_ratio > 3 ? "text-orange-400" : "text-slate-300")}>
                            {metrics.leverage_ratio}x
                        </p>
                    </div>
                </div>
            </div>

            {/* 2. Position Health Table */}
            <div className="flex-1 bg-holon-card rounded-xl border border-slate-700/50 overflow-hidden flex flex-col">
                <div className="bg-slate-900/50 p-3 border-b border-slate-700/50 flex justify-between">
                    <span className="font-orbitron text-xs tracking-wider text-holon-accent">STRUCTURAL HEALTH</span>
                </div>
                <div className="flex-1 overflow-auto">
                    <table className="w-full text-left text-sm">
                        <thead className="bg-slate-950/50 text-[10px] uppercase text-holon-dim font-mono sticky top-0">
                            <tr>
                                <th className="p-3">Asset</th>
                                <th className="p-3 text-center">Alignment</th>
                                <th className="p-3 text-right">Size</th>
                                <th className="p-3 text-right">Drift</th>
                                <th className="p-3 text-right">To TP</th>
                                <th className="p-3 text-right">To SL</th>
                                <th className="p-3 text-right">R:R</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800">
                            {positions.map((p, idx) => (
                                <tr key={idx} className="hover:bg-white/5 transition-colors font-mono text-xs">
                                    <td className="p-3 font-bold text-white flex items-center gap-2">
                                        <span className={p.direction === 'BUY' ? "text-emerald-500" : "text-red-500"}>
                                            {p.direction === 'BUY' ? 'L' : 'S'}
                                        </span>
                                        {p.symbol}
                                    </td>
                                    <td className="p-3 text-center">
                                        {p.aligned ?
                                            <span className="text-emerald-400 text-[10px] bg-emerald-900/30 px-2 py-1 rounded">ALIGNED</span> :
                                            <span className="text-orange-400 text-[10px] bg-orange-900/30 px-2 py-1 rounded">DIVERGENT</span>
                                        }
                                    </td>
                                    <td className="p-3 text-right text-slate-300">${p.size_usd}</td>
                                    <td className={clsx("p-3 text-right font-bold", p.drift_pct >= 0 ? "text-emerald-400" : "text-red-400")}>
                                        {p.drift_pct > 0 ? '+' : ''}{p.drift_pct}%
                                    </td>
                                    <td className="p-3 text-right text-blue-300">
                                        {p.tp_dist_pct ? `${p.tp_dist_pct}%` : '-'}
                                    </td>
                                    <td className="p-3 text-right text-orange-300">
                                        {p.sl_dist_pct ? `${p.sl_dist_pct}%` : '-'}
                                    </td>
                                    <td className="p-3 text-right font-bold text-holon-accent">
                                        {p.rr_ratio ? p.rr_ratio : '-'}
                                    </td>
                                </tr>
                            ))}
                            {positions.length === 0 && (
                                <tr><td colSpan="7" className="p-8 text-center text-holon-dim">No active vectors analyzed.</td></tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}
