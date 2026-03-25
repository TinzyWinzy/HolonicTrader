import React, { useState } from 'react';
import { formatPrice } from '../utils/formatters';
import { TrendingUp, TrendingDown, Target, Shield, AlertTriangle, ArrowRight, Activity, Percent } from 'lucide-react';
import clsx from 'clsx';

/**
 * PositionsPanel - Real-time positions display with PnL
 * Connected to Governor positions via socket
 */
const PositionsPanel = ({ positions = [], portfolioHealth = {} }) => {
    // Calculate totals
    const totalPnl = positions.reduce((sum, p) => sum + (p.pnl || 0), 0);
    const totalNotional = positions.reduce((sum, p) =>
        sum + (p.quantity * (p.current_price || p.entry_price) * (p.leverage || 1)), 0);

    const getPnlColor = (pnl) => {
        if (pnl > 0) return 'text-emerald-400';
        if (pnl < 0) return 'text-red-400';
        return 'text-slate-400';
    };

    const getDirectionBadge = (direction) => {
        if (direction === 'BUY') {
            return (
                <span className="flex items-center gap-1 text-[10px] bg-emerald-500/10 text-emerald-400 px-1.5 py-0.5 rounded border border-emerald-500/20">
                    <TrendingUp size={10} /> LONG
                </span>
            );
        }
        return (
            <span className="flex items-center gap-1 text-[10px] bg-red-500/10 text-red-400 px-1.5 py-0.5 rounded border border-red-500/20">
                <TrendingDown size={10} /> SHORT
            </span>
        );
    };

    const getStrategyBadge = (strategy) => {
        const s = strategy?.toUpperCase() || 'MANUAL';
        if (s.includes('ARB') || s.includes('FUNDING')) return <span className="text-[9px] bg-purple-500/20 text-purple-300 px-1.5 py-0.5 rounded border border-purple-500/30">⚡ ARB</span>;
        if (s.includes('MOMENTUM') || s.includes('WHALE')) return <span className="text-[9px] bg-blue-500/20 text-blue-300 px-1.5 py-0.5 rounded border border-blue-500/30">🌊 MOMENTUM</span>;
        if (s.includes('MEAN') || s.includes('REV')) return <span className="text-[9px] bg-indigo-500/20 text-indigo-300 px-1.5 py-0.5 rounded border border-indigo-500/30">↩️ MEAN REV</span>;
        return <span className="text-[9px] bg-slate-700/50 text-slate-400 px-1.5 py-0.5 rounded border border-slate-600/30">{s}</span>;
    };

    return (
        <div className="bg-holon-card rounded-xl border border-slate-700/50 flex flex-col shadow-lg overflow-hidden h-full min-h-[300px]">
            {/* Header */}
            <div className="bg-slate-900/50 p-3 border-b border-slate-700/50 flex justify-between items-center">
                <div className="flex items-center gap-2 text-white">
                    <Activity size={16} className="text-blue-400" />
                    <span className="font-orbitron text-xs tracking-wider">ACTIVE POSITIONS</span>
                    <span className="text-[10px] bg-slate-800 text-slate-400 px-1.5 rounded-full">{positions.length}</span>
                </div>
                <div className="flex gap-4 text-xs font-mono">
                    <div className="flex items-center gap-1 text-slate-400">
                        <span>EXP:</span>
                        <span className="text-slate-200 indent-0.5 font-bold">${totalNotional.toFixed(0)}</span>
                    </div>
                    <div className={clsx("flex items-center gap-1 font-bold", getPnlColor(totalPnl))}>
                        <span>PnL:</span>
                        <span>{totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(2)}</span>
                    </div>
                </div>
            </div>

            {/* Portfolio Health Bar */}
            {portfolioHealth.drawdown_pct !== undefined && (
                <div className="px-3 py-2 bg-slate-950/30 border-b border-slate-800/50">
                    <div className="flex justify-between text-[10px] text-slate-500 font-mono mb-1">
                        <span className="flex items-center gap-1"><Shield size={10} /> Margin Util: {((portfolioHealth.margin_utilization || 0) * 100).toFixed(1)}%</span>
                        <span className="flex items-center gap-1">Drawdown: {((portfolioHealth.drawdown_pct || 0) * 100).toFixed(2)}%</span>
                    </div>
                    <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden relative">
                        {/* Safe Zone Marker */}
                        <div className="absolute right-[20%] top-0 bottom-0 w-0.5 bg-slate-600/50 z-10" title="80% Warning" />

                        <div
                            className={clsx("h-full transition-all duration-500 rounded-full",
                                portfolioHealth.margin_utilization > 0.8 ? 'bg-gradient-to-r from-red-600 to-red-500' :
                                    portfolioHealth.margin_utilization > 0.5 ? 'bg-gradient-to-r from-yellow-600 to-yellow-500' :
                                        'bg-gradient-to-r from-emerald-600 to-emerald-500'
                            )}
                            style={{ width: `${Math.min(100, (portfolioHealth.margin_utilization || 0) * 100)}%` }}
                        />
                    </div>
                </div>
            )}

            {/* Positions Table */}
            <div className="flex-1 overflow-auto p-0 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
                <table className="w-full text-left text-sm whitespace-nowrap">
                    <thead className="bg-slate-950/50 text-[9px] uppercase text-holon-dim font-mono sticky top-0 z-10 backdrop-blur-sm">
                        <tr>
                            <th className="p-2 pl-3">Asset / Strat</th>
                            <th className="p-2 text-right">Size (Lev)</th>
                            <th className="p-2 text-right">Entry → Mark</th>
                            <th className="p-2 text-center">Liq Risk</th>
                            <th className="p-2 text-right pr-3">Unrealized PnL</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/50">
                        {positions.length === 0 ? (
                            <tr>
                                <td colSpan="5" className="p-12 text-center text-slate-600 italic text-xs">
                                    <Target className="mx-auto mb-2 opacity-20" size={32} />
                                    No active mandates. Scan running...
                                </td>
                            </tr>
                        ) : (
                            positions.map((pos, idx) => {
                                // Derived Metrics
                                const roe = pos.margin > 0 ? (pos.pnl / pos.margin) * 100 : 0;
                                const distToLiq = pos.liquidation_price ? Math.abs((pos.current_price - pos.liquidation_price) / pos.current_price) * 100 : 100;
                                const isLiqRisk = distToLiq < 5; // <5% to liq

                                return (
                                    <tr key={pos.symbol || idx} className="hover:bg-white/5 transition-colors group">
                                        {/* Asset + Strategy */}
                                        <td className="p-2 pl-3">
                                            <div className="flex flex-col gap-0.5">
                                                <div className="font-bold text-white flex items-center gap-2">
                                                    {pos.symbol?.replace('/USDT', '')}
                                                    {getDirectionBadge(pos.direction)}
                                                </div>
                                                <div className="flex items-center gap-1.5 opacity-80 group-hover:opacity-100 transition-opacity">
                                                    {getStrategyBadge(pos.strategy)}
                                                    {pos.unrealized_pnl_pct && (
                                                        <span className={clsx("text-[9px] font-mono",
                                                            pos.unrealized_pnl_pct > 0 ? "text-emerald-500" : "text-red-500"
                                                        )}>
                                                            {pos.unrealized_pnl_pct.toFixed(2)}%
                                                        </span>
                                                    )}
                                                </div>
                                            </div>
                                        </td>

                                        {/* Size & Leverage */}
                                        <td className="p-2 text-right font-mono text-xs">
                                            <div className="text-slate-200">{Math.abs(pos.quantity || 0).toFixed(4)}</div>
                                            <div className="text-[10px] text-slate-500 bg-slate-800/50 px-1 rounded inline-block mt-0.5">
                                                {pos.leverage}x
                                            </div>
                                        </td>

                                        {/* Price Movement */}
                                        <td className="p-2 text-right font-mono text-xs">
                                            <div className="text-slate-400 text-[10px] line-through decoration-slate-600">{formatPrice(pos.entry_price)}</div>
                                            <div className="flex items-center justify-end gap-1 text-slate-200">
                                                <ArrowRight size={10} className="text-slate-600" />
                                                {formatPrice(pos.current_price)}
                                            </div>
                                        </td>

                                        {/* Liquidation Risk */}
                                        <td className="p-2 text-center align-middle">
                                            {pos.liquidation_price ? (
                                                <div className="flex flex-col items-center justify-center">
                                                    {isLiqRisk && <AlertTriangle size={12} className="text-red-500 animate-pulse mb-0.5" />}
                                                    <span className={clsx("text-[10px] font-mono",
                                                        distToLiq < 2 ? "text-red-500 font-bold" :
                                                            distToLiq < 10 ? "text-amber-500" : "text-emerald-500"
                                                    )}>
                                                        {distToLiq.toFixed(1)}%
                                                    </span>
                                                    <div className="w-12 h-1 bg-slate-800 rounded-full mt-0.5 overflow-hidden">
                                                        <div
                                                            className={clsx("h-full", distToLiq < 10 ? "bg-red-500" : "bg-emerald-500")}
                                                            style={{ width: `${Math.max(0, 100 - distToLiq)}%` }}
                                                        />
                                                    </div>
                                                </div>
                                            ) : (
                                                <span className="text-[10px] text-slate-600">—</span>
                                            )}
                                        </td>

                                        {/* PnL & ROE */}
                                        <td className="p-2 text-right pr-3 font-mono">
                                            <div className={clsx("font-bold text-sm", getPnlColor(pos.pnl))}>
                                                {pos.pnl >= 0 ? '+' : ''}{pos.pnl?.toFixed(2)}
                                            </div>
                                            {roe !== 0 && (
                                                <span className={clsx("text-[9px] px-1 py-0.5 rounded",
                                                    roe > 0 ? "bg-emerald-500/10 text-emerald-400" : "bg-red-500/10 text-red-400"
                                                )}>
                                                    ROE {roe > 0 ? '+' : ''}{roe.toFixed(1)}%
                                                </span>
                                            )}
                                        </td>
                                    </tr>
                                );
                            })
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default PositionsPanel;
