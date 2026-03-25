import React from 'react';
import { Signal, Target, TrendingUp, TrendingDown, Shield, Zap, AlertTriangle, Clock, Percent, Activity } from 'lucide-react';
import clsx from 'clsx';
import { formatPrice } from '../utils/formatters';

const SignalCard = ({ signal }) => {
    const isLong = signal.direction === 'BUY';
    const exec = signal.execution_details || {};
    const account = signal.account_context || {};

    const qualityColors = {
        'HIGH': 'border-emerald-500/50 bg-emerald-500/5',
        'MEDIUM': 'border-blue-500/50 bg-blue-500/5',
        'VETOED': 'border-red-500/50 bg-red-500/5',
    };

    const qualityBadge = {
        'HIGH': 'bg-emerald-500/20 text-emerald-400',
        'MEDIUM': 'bg-blue-500/20 text-blue-400',
        'VETOED': 'bg-red-500/20 text-red-400',
    };

    // Monte Carlo anticipation data
    const expectedYield = signal.expected_yield || 0;
    const hitProbability = signal.hit_probability || 0;
    const decayScore = signal.decay_score || 1;
    const optimalHorizon = signal.optimal_horizon || 24;
    const pipsPotential = signal.pips_potential || 0;

    const getDecayColor = (score) => {
        if (score >= 0.8) return 'text-emerald-400';
        if (score >= 0.5) return 'text-yellow-400';
        return 'text-red-400';
    };

    const hpsScore = signal.hps_score || 0;
    const hpsPillars = signal.hps_pillars || [];

    const getHpsStars = (score) => {
        return Array(5).fill(0).map((_, i) => (
            <span key={i} className={i < score ? "text-amber-400 text-xs shadow-glow-gold" : "text-slate-700 text-xs"}>★</span>
        ));
    };

    return (
        <div className={clsx(
            "rounded-lg border p-3 transition-all hover:scale-[1.01] relative overflow-hidden",
            qualityColors[signal.quality] || 'border-slate-700/50 bg-slate-800/30'
        )}>
            {/* HPS Background Glow */}
            {hpsScore >= 4 && <div className="absolute top-0 right-0 w-32 h-32 bg-amber-500/10 blur-3xl rounded-full -mr-16 -mt-16 pointer-events-none" />}

            {/* Header: Symbol + Direction + Quality */}
            <div className="flex items-center justify-between mb-3 relative z-10">
                <div className="flex items-center gap-2">
                    <div className={clsx(
                        "w-8 h-8 rounded-lg flex items-center justify-center",
                        signal.direction === 'BUY' ? "bg-emerald-500/20" :
                            signal.direction === 'NEUTRAL' ? "bg-slate-500/20" : "bg-red-500/20"
                    )}>
                        {signal.direction === 'BUY' ? <TrendingUp className="text-emerald-400" size={18} /> :
                            signal.direction === 'NEUTRAL' ? <Signal className="text-slate-400" size={18} /> :
                                <TrendingDown className="text-red-400" size={18} />}
                    </div>
                    <div>
                        <div className="flex items-center gap-2">
                            <h3 className="font-bold text-white font-mono">{signal.symbol}</h3>
                            {/* HPS Star Rating */}
                            <div className="flex -gap-0.5 ml-1" title={`HPS Score: ${hpsScore}/5`}>
                                {getHpsStars(hpsScore)}
                            </div>
                        </div>
                        <span className={clsx("text-[10px] font-bold",
                            signal.direction === 'BUY' ? "text-emerald-400" :
                                signal.direction === 'NEUTRAL' ? "text-slate-400" : "text-red-400")}>
                            {signal.direction === 'BUY' ? '▲ LONG' :
                                signal.direction === 'NEUTRAL' ? '• NEUTRAL' : '▼ SHORT'}
                        </span>
                    </div>
                </div>
                <div className="flex flex-col items-end gap-1">
                    <span className={clsx("px-2 py-0.5 rounded text-[10px] font-bold", qualityBadge[signal.quality] || qualityBadge['MEDIUM'])}>
                        {signal.quality || 'MEDIUM'}
                    </span>
                    {hpsScore >= 3 && (
                        <span className="text-[8px] text-amber-400 font-mono border border-amber-500/30 bg-amber-500/10 px-1 rounded">
                            SNIPER
                        </span>
                    )}
                </div>
            </div>

            {/* === MONTE CARLO ANTICIPATION === */}
            <div className="grid grid-cols-4 gap-1.5 mb-3 bg-gradient-to-r from-purple-900/30 to-blue-900/30 rounded-lg p-2 border border-purple-500/20">
                <div className="text-center">
                    <p className="text-[8px] text-purple-400 uppercase font-bold">Yield</p>
                    <p className={clsx("text-sm font-mono font-bold", expectedYield > 0 ? "text-emerald-400" : "text-gray-400")}>
                        {expectedYield > 0 ? '+' : ''}{expectedYield.toFixed(1)}%
                    </p>
                </div>
                <div className="text-center">
                    <p className="text-[8px] text-blue-400 uppercase font-bold">Hit %</p>
                    <p className="text-sm font-mono font-bold text-blue-400">{hitProbability.toFixed(0)}%</p>
                </div>
                <div className="text-center">
                    <p className="text-[8px] text-amber-400 uppercase font-bold">Fresh</p>
                    <p className={clsx("text-sm font-mono font-bold", getDecayColor(decayScore))}>
                        {(decayScore * 100).toFixed(0)}%
                    </p>
                </div>
                <div className="text-center">
                    <p className="text-[8px] text-cyan-400 uppercase font-bold">Window</p>
                    <p className="text-sm font-mono font-bold text-cyan-400">{optimalHorizon}h</p>
                </div>
            </div>

            {/* Price Levels */}
            <div className="grid grid-cols-3 gap-2 mb-3 text-center">
                <div className="bg-slate-900/50 rounded p-2">
                    <p className="text-[9px] text-slate-500 uppercase">Entry</p>
                    <p className="text-sm font-mono text-white font-bold">${formatPrice(signal.price)}</p>
                </div>
                <div className="bg-emerald-900/20 rounded p-2 border border-emerald-500/20">
                    <p className="text-[9px] text-emerald-500 uppercase">Take Profit</p>
                    <p className="text-sm font-mono text-emerald-400 font-bold">${formatPrice(signal.tp)}</p>
                </div>
                <div className="bg-red-900/20 rounded p-2 border border-red-500/20">
                    <p className="text-[9px] text-red-500 uppercase">Stop Loss</p>
                    <p className="text-sm font-mono text-red-400 font-bold">${formatPrice(signal.sl)}</p>
                </div>
            </div>

            {/* Execution Details */}
            <div className="grid grid-cols-3 gap-2 mb-3 text-xs">
                <div className="flex items-center gap-1 text-slate-400">
                    <Zap size={12} className="text-yellow-500" />
                    <span>Lev: <span className="text-white font-bold">{exec.leverage?.toFixed(1)}x</span></span>
                </div>
                <div className="flex items-center gap-1 text-slate-400">
                    <Target size={12} className="text-blue-500" />
                    <span>Qty: <span className="text-white font-bold">{exec.quantity?.toFixed(4)}</span></span>
                </div>
                <div className="flex items-center gap-1 text-slate-400">
                    <Shield size={12} className="text-purple-500" />
                    <span>{exec.order_type}</span>
                </div>
            </div>

            {/* Conviction Bar */}
            <div className="mb-3">
                <div className="flex justify-between text-[10px] mb-1">
                    <span className="text-slate-500">CONVICTION</span>
                    <span className="text-white font-bold">{((signal.conviction || 0) * 100).toFixed(1)}%</span>
                </div>
                <div className="h-1.5 bg-slate-700/50 rounded-full overflow-hidden">
                    <div
                        className={clsx("h-full rounded-full transition-all", {
                            'bg-emerald-500': (signal.conviction || 0) >= 0.7,
                            'bg-blue-500': (signal.conviction || 0) >= 0.5 && (signal.conviction || 0) < 0.7,
                            'bg-amber-500': (signal.conviction || 0) < 0.5
                        })}
                        style={{ width: `${(signal.conviction || 0) * 100}%` }}
                    />
                </div>
            </div>

            {/* Status Tags */}
            <div className="flex flex-wrap gap-1 mb-2">
                <span className="px-2 py-0.5 bg-purple-500/20 text-purple-400 text-[9px] rounded font-mono">
                    {signal.regime}
                </span>
                <span className="px-2 py-0.5 bg-cyan-500/20 text-cyan-400 text-[9px] rounded font-mono">
                    {signal.tda_status}
                </span>
                {signal.metadata?.is_whale && (
                    <span className="px-2 py-0.5 bg-amber-500/20 text-amber-400 text-[9px] rounded font-mono">
                        🐋 WHALE
                    </span>
                )}
                {pipsPotential > 0 && (
                    <span className="px-2 py-0.5 bg-green-500/20 text-green-400 text-[9px] rounded font-mono">
                        ${pipsPotential.toFixed(2)} pips
                    </span>
                )}
            </div>

            {/* Reason */}
            <div className="text-[10px] text-slate-400 font-mono border-t border-slate-700/50 pt-2 mt-2">
                <span className="text-slate-500">REASON:</span> {signal.reason}
            </div>

            {/* Timestamp */}
            <div className="text-[9px] text-slate-600 mt-1 font-mono">
                {new Date(signal.timestamp).toLocaleTimeString()}
            </div>
        </div>
    );
};

const SignalsPanel = ({ signals = [], lastScan }) => {
    return (
        <div className="bg-holon-card rounded-xl border border-slate-700/50 p-4 h-full flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <Signal className="text-blue-500" size={18} />
                    <h2 className="font-orbitron text-sm text-white font-bold tracking-wide">LIVE SIGNALS</h2>
                </div>
                <div className="text-[10px] text-slate-500 font-mono">
                    {signals.length} active
                </div>
            </div>

            {/* Last Scan */}
            {lastScan && (
                <div className="text-[9px] text-slate-500 font-mono mb-3 flex items-center gap-1">
                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                    Last Scan: {new Date(lastScan).toLocaleTimeString()}
                </div>
            )}

            {/* Signals List */}
            <div className="flex-1 overflow-y-auto space-y-3 pr-1">
                {signals.length > 0 ? (
                    signals.map((signal, idx) => (
                        <SignalCard key={`${signal.symbol}-${idx}`} signal={signal} />
                    ))
                ) : (
                    <div className="flex flex-col items-center justify-center h-full text-slate-500">
                        <AlertTriangle size={24} className="mb-2 opacity-50" />
                        <p className="text-xs">No active signals</p>
                        <p className="text-[10px]">Waiting for signal scan...</p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default SignalsPanel;
