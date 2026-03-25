import React, { useState } from 'react';
import { Radar, Target, ChevronDown, ChevronUp, Crosshair, Zap, TrendingUp, Shield, Clock } from 'lucide-react';
import clsx from 'clsx';

const SignalRow = ({ item, isExpanded, onToggle }) => {
    const strategy = item.metadata?.strategy || item.reason?.split(' |')[0] || 'SCAN';
    const hps = item.hps_score || 0;
    const conviction = item.conviction || 0;
    const hitProb = item.hit_probability || 0;
    const expectedYield = item.expected_yield || 0;
    const quality = item.quality || item.status || 'MEDIUM';
    const exec = item.execution_details || {};

    const qualityColor = {
        'HIGH': 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
        'MEDIUM': 'bg-blue-500/20 text-blue-400 border-blue-500/30',
        'LOW': 'bg-amber-500/20 text-amber-400 border-amber-500/30',
        'VETOED': 'bg-red-500/20 text-red-400 border-red-500/30',
    }[quality] || 'bg-slate-500/20 text-slate-400 border-slate-500/30';

    return (
        <>
            <tr
                className="hover:bg-white/5 transition-colors font-mono text-xs cursor-pointer"
                onClick={onToggle}
            >
                {/* Symbol + Direction */}
                <td className="p-2 pl-3 font-bold text-white">
                    <div className="flex items-center gap-2">
                        <Target size={12} className="text-blue-500/50 flex-shrink-0" />
                        <div className="flex flex-col">
                            <span>{item.symbol}</span>
                            {item.direction && (
                                <span className={clsx("text-[9px]",
                                    item.direction === 'BUY' ? "text-emerald-500" : "text-red-500"
                                )}>
                                    {item.direction === 'BUY' ? '▲ LONG' : '▼ SHORT'}
                                </span>
                            )}
                        </div>
                    </div>
                </td>

                {/* Strategy Tag */}
                <td className="p-2">
                    <span className="text-[9px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 whitespace-nowrap">
                        {strategy.replace(/_/g, ' ')}
                    </span>
                </td>

                {/* HPS Score */}
                <td className="p-2 text-center">
                    <div className="flex items-center justify-center gap-1">
                        <div className="flex gap-0.5">
                            {[1, 2, 3, 4, 5].map(i => (
                                <div
                                    key={i}
                                    className={clsx(
                                        "w-1.5 h-3 rounded-sm",
                                        i <= hps ? "bg-blue-400" : "bg-slate-700"
                                    )}
                                />
                            ))}
                        </div>
                        <span className="text-blue-300 text-[10px] ml-1">{hps}/5</span>
                    </div>
                </td>

                {/* Hit Probability */}
                <td className="p-2 text-right">
                    <span className={clsx("font-bold",
                        hitProb >= 80 ? "text-emerald-400" : hitProb >= 50 ? "text-amber-400" : "text-red-400"
                    )}>
                        {hitProb.toFixed(0)}%
                    </span>
                </td>

                {/* Quality Badge */}
                <td className="p-2 text-center">
                    <span className={clsx("px-1.5 py-0.5 rounded text-[9px] font-bold border", qualityColor)}>
                        {quality}
                    </span>
                </td>

                {/* Expand Toggle */}
                <td className="p-2 pr-3 text-center text-slate-500">
                    {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                </td>
            </tr>

            {/* Expanded Detail Row */}
            {isExpanded && (
                <tr className="bg-slate-900/60">
                    <td colSpan="6" className="p-0">
                        <div className="px-4 py-3 border-t border-slate-800/50">
                            {/* Metrics Row */}
                            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-3">
                                <div className="bg-slate-800/50 rounded-lg p-2 border border-slate-700/30">
                                    <p className="text-[9px] text-slate-500 uppercase tracking-wider flex items-center gap-1">
                                        <Zap size={10} /> Conviction
                                    </p>
                                    <p className="text-sm font-bold text-white">{(conviction * 100).toFixed(0)}%</p>
                                </div>
                                <div className="bg-slate-800/50 rounded-lg p-2 border border-slate-700/30">
                                    <p className="text-[9px] text-slate-500 uppercase tracking-wider flex items-center gap-1">
                                        <TrendingUp size={10} /> Exp. Yield
                                    </p>
                                    <p className={clsx("text-sm font-bold", expectedYield >= 0 ? "text-emerald-400" : "text-red-400")}>
                                        {expectedYield >= 0 ? '+' : ''}{expectedYield.toFixed(2)}%
                                    </p>
                                </div>
                                <div className="bg-slate-800/50 rounded-lg p-2 border border-slate-700/30">
                                    <p className="text-[9px] text-slate-500 uppercase tracking-wider flex items-center gap-1">
                                        <Shield size={10} /> Leverage
                                    </p>
                                    <p className="text-sm font-bold text-blue-300">{exec.leverage?.toFixed(1) || '—'}x</p>
                                </div>
                                <div className="bg-slate-800/50 rounded-lg p-2 border border-slate-700/30">
                                    <p className="text-[9px] text-slate-500 uppercase tracking-wider flex items-center gap-1">
                                        <Clock size={10} /> Horizon
                                    </p>
                                    <p className="text-sm font-bold text-purple-300">{item.optimal_horizon || '—'}h</p>
                                </div>
                            </div>

                            {/* Price Levels & Execution */}
                            <div className="grid grid-cols-2 gap-3">
                                {/* Price Levels */}
                                <div className="bg-slate-800/40 rounded-lg p-2 border border-slate-700/20">
                                    <p className="text-[9px] text-slate-500 uppercase tracking-wider mb-1.5">Price Levels</p>
                                    <div className="space-y-1 font-mono text-[11px]">
                                        <div className="flex justify-between">
                                            <span className="text-slate-400">Entry</span>
                                            <span className="text-white font-bold">${item.price?.toFixed(4) || '—'}</span>
                                        </div>
                                        {item.tp && (
                                            <div className="flex justify-between">
                                                <span className="text-emerald-500">TP</span>
                                                <span className="text-emerald-400">${item.tp?.toFixed(4)}</span>
                                            </div>
                                        )}
                                        {item.sl && (
                                            <div className="flex justify-between">
                                                <span className="text-red-500">SL</span>
                                                <span className="text-red-400">${item.sl?.toFixed(4)}</span>
                                            </div>
                                        )}
                                        {item.pips_potential > 0 && (
                                            <div className="flex justify-between">
                                                <span className="text-slate-400">Pips</span>
                                                <span className="text-blue-300">+{item.pips_potential?.toFixed(4)}</span>
                                            </div>
                                        )}
                                    </div>
                                </div>

                                {/* Execution Details */}
                                <div className="bg-slate-800/40 rounded-lg p-2 border border-slate-700/20">
                                    <p className="text-[9px] text-slate-500 uppercase tracking-wider mb-1.5">Execution</p>
                                    <div className="space-y-1 font-mono text-[11px]">
                                        <div className="flex justify-between">
                                            <span className="text-slate-400">Type</span>
                                            <span className="text-white">{exec.order_type || '—'}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-slate-400">Qty</span>
                                            <span className="text-white">{exec.quantity?.toFixed(4) || '—'}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-slate-400">Decay</span>
                                            <span className={clsx(
                                                (item.decay_score || 0) >= 0.8 ? "text-emerald-400" :
                                                    (item.decay_score || 0) >= 0.5 ? "text-amber-400" : "text-red-400"
                                            )}>
                                                {((item.decay_score || 0) * 100).toFixed(0)}%
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* HPS Pillars */}
                            {item.hps_pillars?.length > 0 && (
                                <div className="flex gap-1.5 mt-2.5 flex-wrap">
                                    {item.hps_pillars.map((pillar, i) => (
                                        <span key={i} className="text-[9px] px-1.5 py-0.5 rounded bg-blue-900/30 text-blue-300 border border-blue-800/30">
                                            {pillar}
                                        </span>
                                    ))}
                                    {item.metadata?.is_whale && (
                                        <span className="text-[9px] px-1.5 py-0.5 rounded bg-purple-900/30 text-purple-300 border border-purple-800/30">
                                            🐋 WHALE
                                        </span>
                                    )}
                                </div>
                            )}

                            {/* Reason / Notes */}
                            {item.reason && (
                                <p className="text-[10px] text-slate-400 mt-2 truncate" title={item.reason}>
                                    {item.reason}
                                </p>
                            )}
                        </div>
                    </td>
                </tr>
            )}
        </>
    );
};

const RadarPanel = ({ items = [] }) => {
    const [expandedIdx, setExpandedIdx] = useState(null);

    const toggleExpand = (idx) => {
        setExpandedIdx(prev => prev === idx ? null : idx);
    };

    // Sort by quality (HIGH first), then by HPS score
    const qualityOrder = { HIGH: 0, MEDIUM: 1, LOW: 2, VETOED: 3 };
    const sorted = [...items].sort((a, b) => {
        const qa = qualityOrder[a.quality] ?? 99;
        const qb = qualityOrder[b.quality] ?? 99;
        if (qa !== qb) return qa - qb;
        return (b.hps_score || 0) - (a.hps_score || 0);
    });

    return (
        <div className="bg-holon-card rounded-xl border border-slate-700/50 flex flex-col shadow-lg overflow-hidden h-full">
            <div className="bg-slate-900/50 p-3 border-b border-slate-700/50 flex justify-between items-center">
                <div className="flex items-center gap-2 text-blue-400">
                    <Radar size={16} />
                    <span className="font-orbitron text-xs tracking-wider">SIGNAL RADAR</span>
                </div>
                <div className="flex gap-2 items-center">
                    <span className="text-[10px] font-mono text-holon-dim">
                        {items.length} TARGETS
                    </span>
                    <span className="text-[10px] bg-emerald-900/30 text-emerald-400 px-2 py-0.5 rounded border border-emerald-800/30">
                        {items.filter(s => s.quality === 'HIGH').length} HIGH
                    </span>
                </div>
            </div>

            <div className="flex-1 overflow-auto p-0">
                <table className="w-full text-left text-sm">
                    <thead className="bg-slate-950/50 text-[10px] uppercase text-holon-dim font-mono sticky top-0">
                        <tr>
                            <th className="p-2 pl-3">Asset</th>
                            <th className="p-2">Strategy</th>
                            <th className="p-2 text-center">HPS</th>
                            <th className="p-2 text-right">Hit %</th>
                            <th className="p-2 text-center">Quality</th>
                            <th className="p-2 pr-3 w-8"></th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800">
                        {sorted.length === 0 ? (
                            <tr>
                                <td colSpan="6" className="p-8 text-center text-holon-dim italic text-xs">
                                    <Crosshair className="mx-auto mb-2 opacity-30" size={24} />
                                    Scanning sector alpha...
                                </td>
                            </tr>
                        ) : (
                            sorted.map((item, idx) => (
                                <SignalRow
                                    key={idx}
                                    item={item}
                                    isExpanded={expandedIdx === idx}
                                    onToggle={() => toggleExpand(idx)}
                                />
                            ))
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default RadarPanel;
