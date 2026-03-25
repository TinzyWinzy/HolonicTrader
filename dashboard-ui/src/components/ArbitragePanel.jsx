import React from 'react';
import clsx from 'clsx';
import { TrendingUp, TrendingDown, Percent, ArrowUpRight, ArrowDownRight, Zap } from 'lucide-react';

const ArbitragePanel = ({ opportunities = [] }) => {
    // Sort by absolute APY (highest first)
    const sortedOpps = [...opportunities].sort((a, b) => Math.abs(b.funding_apy) - Math.abs(a.funding_apy));

    const getApyColor = (apy) => {
        const absApy = Math.abs(apy);
        if (absApy >= 100) return 'text-yellow-400';
        if (absApy >= 50) return 'text-emerald-400';
        if (absApy >= 20) return 'text-blue-400';
        return 'text-slate-400';
    };

    const getApyBg = (apy) => {
        const absApy = Math.abs(apy);
        if (absApy >= 100) return 'bg-yellow-500/20 border-yellow-500/50';
        if (absApy >= 50) return 'bg-emerald-500/20 border-emerald-500/50';
        if (absApy >= 20) return 'bg-blue-500/20 border-blue-500/50';
        return 'bg-slate-700/30 border-slate-600/50';
    };

    return (
        <div className="bg-holon-card rounded-xl border border-slate-700/50 shadow-xl overflow-hidden">
            {/* Header */}
            <div className="bg-gradient-to-r from-yellow-900/30 to-amber-900/20 p-3 border-b border-slate-700/50 flex justify-between items-center">
                <div className="flex items-center gap-2">
                    <Zap className="text-yellow-400" size={18} />
                    <span className="font-orbitron text-sm tracking-wider text-yellow-400">ARBITRAGE RADAR</span>
                </div>
                <div className="flex items-center gap-2">
                    <span className="text-[10px] bg-yellow-500/20 text-yellow-400 px-2 py-0.5 rounded border border-yellow-500/30">
                        {opportunities.filter(o => o.has_opportunity).length} SIGNALS
                    </span>
                </div>
            </div>

            {/* Content */}
            <div className="p-3">
                {sortedOpps.length === 0 ? (
                    <div className="text-center text-holon-dim py-8">
                        <Percent className="mx-auto mb-2 opacity-30" size={32} />
                        <p className="text-sm">No funding rate data available</p>
                        <p className="text-xs mt-1">Ensure FUTURES mode is enabled</p>
                    </div>
                ) : (
                    <div className="space-y-2">
                        {sortedOpps.map((opp, idx) => (
                            <div
                                key={idx}
                                className={clsx(
                                    "p-3 rounded-lg border transition-all",
                                    opp.has_opportunity
                                        ? getApyBg(opp.funding_apy)
                                        : "bg-slate-800/30 border-slate-700/30"
                                )}
                            >
                                <div className="flex items-center justify-between">
                                    {/* Symbol & Direction */}
                                    <div className="flex items-center gap-2">
                                        <span className="font-semibold text-white">{opp.symbol}</span>
                                        {opp.signal && (
                                            <span className={clsx(
                                                "px-1.5 py-0.5 rounded text-[10px] font-bold flex items-center gap-1",
                                                opp.signal === 'BUY'
                                                    ? "bg-emerald-500/30 text-emerald-400"
                                                    : "bg-red-500/30 text-red-400"
                                            )}>
                                                {opp.signal === 'BUY' ? <ArrowUpRight size={10} /> : <ArrowDownRight size={10} />}
                                                {opp.signal}
                                            </span>
                                        )}
                                    </div>

                                    {/* APY Badge */}
                                    <div className={clsx(
                                        "text-right",
                                        getApyColor(opp.funding_apy)
                                    )}>
                                        <div className="flex items-center gap-1">
                                            {opp.funding_apy > 0 ? (
                                                <TrendingUp size={14} />
                                            ) : (
                                                <TrendingDown size={14} />
                                            )}
                                            <span className="font-bold text-lg">
                                                {opp.funding_apy > 0 ? '+' : ''}{opp.funding_apy.toFixed(1)}%
                                            </span>
                                        </div>
                                        <span className="text-[10px] text-slate-400">APY</span>
                                    </div>
                                </div>

                                {/* Reason row */}
                                {opp.reason && (
                                    <div className="mt-2 pt-2 border-t border-slate-700/50 flex justify-between items-center">
                                        <span className="text-xs text-slate-400">{opp.reason}</span>
                                        <span className="text-xs text-slate-500">
                                            {(opp.confidence * 100).toFixed(0)}% conf
                                        </span>
                                    </div>
                                )}

                                {/* Spread if available */}
                                {(opp.spread_simple !== 0 || opp.spread_long !== undefined) && (
                                    <div className="text-[10px] text-slate-500 mt-1 flex justify-between">
                                        {/* Display Precise Spread based on Signal */}
                                        {opp.signal === 'BUY' && (
                                            <span className="text-emerald-500 font-mono">
                                                Discount: {(opp.spread_long || 0).toFixed(2)}%
                                            </span>
                                        )}
                                        {opp.signal === 'SELL' && (
                                            <span className="text-red-400 font-mono">
                                                Premium: {(opp.spread_short || 0).toFixed(2)}%
                                            </span>
                                        )}
                                        {/* Fallback or Context */}
                                        {!opp.signal && opp.spread_pct && (
                                            <span>Spread: {opp.spread_pct > 0 ? '+' : ''}{opp.spread_pct}%</span>
                                        )}
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Footer Legend */}
            <div className="bg-slate-900/50 px-3 py-2 border-t border-slate-700/50">
                <div className="flex flex-wrap gap-3 text-[9px] text-slate-500">
                    <span className="flex items-center gap-1">
                        <span className="w-2 h-2 rounded bg-yellow-500"></span> &gt;100% APY
                    </span>
                    <span className="flex items-center gap-1">
                        <span className="w-2 h-2 rounded bg-emerald-500"></span> 50-100%
                    </span>
                    <span className="flex items-center gap-1">
                        <span className="w-2 h-2 rounded bg-blue-500"></span> 20-50%
                    </span>
                </div>
            </div>
        </div>
    );
};

export default ArbitragePanel;
