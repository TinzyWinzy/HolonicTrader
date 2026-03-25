import React, { useState, useEffect, useCallback } from 'react';
import clsx from 'clsx';
import {
    FileText, RefreshCw, TrendingUp, TrendingDown, Shield,
    Zap, Target, AlertTriangle, DollarSign, BarChart3,
    ArrowUpRight, ArrowDownRight, Activity
} from 'lucide-react';

const SitrepPanel = () => {
    const [sitrep, setSitrep] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [lastFetch, setLastFetch] = useState(null);

    const fetchSitrep = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const res = await fetch('/api/sitrep');
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            setSitrep(data);
            setLastFetch(new Date().toLocaleTimeString());
        } catch (e) {
            setError(e.message);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchSitrep();
        const interval = setInterval(fetchSitrep, 30000); // Refresh every 30s
        return () => clearInterval(interval);
    }, [fetchSitrep]);

    if (error) {
        return (
            <div className="bg-holon-card rounded-xl border border-red-700/50 p-6 text-center">
                <AlertTriangle className="mx-auto mb-2 text-red-400" size={32} />
                <p className="text-red-400 text-sm font-mono">SITREP FAILED: {error}</p>
                <button onClick={fetchSitrep} className="mt-3 text-xs text-blue-400 hover:text-blue-300 underline">
                    Retry
                </button>
            </div>
        );
    }

    if (!sitrep) {
        return (
            <div className="bg-holon-card rounded-xl border border-slate-700/50 p-8 text-center animate-pulse">
                <Activity className="mx-auto mb-2 text-slate-500" size={32} />
                <p className="text-holon-dim text-sm">Loading SITREP...</p>
            </div>
        );
    }

    const { overview, signals, positions, arbitrage, chart } = sitrep;

    const getSetupColor = (type) => {
        const colors = {
            'FUNDING CARRY': 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30',
            'SPATIAL ARBITRAGE': 'text-purple-400 bg-purple-500/10 border-purple-500/30',
            'WHALE SIGNAL': 'text-blue-400 bg-blue-500/10 border-blue-500/30',
            'MOMENTUM': 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30',
            'MEAN REVERSION': 'text-cyan-400 bg-cyan-500/10 border-cyan-500/30',
            'CRISIS HEDGE': 'text-red-400 bg-red-500/10 border-red-500/30',
        };
        return colors[type] || 'text-slate-400 bg-slate-500/10 border-slate-500/30';
    };

    const getQualityBadge = (quality) => {
        const map = {
            'A+': 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40',
            'A': 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
            'B': 'bg-blue-500/15 text-blue-300 border-blue-500/30',
            'C': 'bg-yellow-500/15 text-yellow-300 border-yellow-500/30',
            'VETO': 'bg-red-500/15 text-red-400 border-red-500/30',
        };
        return map[quality] || 'bg-slate-500/15 text-slate-300 border-slate-500/30';
    };

    return (
        <div className="flex flex-col gap-4">
            {/* === HEADER === */}
            <div className="bg-holon-card rounded-xl border border-slate-700/50 overflow-hidden">
                <div className="bg-gradient-to-r from-indigo-900/40 to-violet-900/30 p-3 border-b border-slate-700/50 flex justify-between items-center">
                    <div className="flex items-center gap-2">
                        <FileText className="text-indigo-400" size={18} />
                        <span className="font-orbitron text-sm tracking-wider text-indigo-400">SITUATION REPORT</span>
                    </div>
                    <div className="flex items-center gap-3">
                        {lastFetch && <span className="text-[9px] text-slate-500 font-mono">{lastFetch}</span>}
                        <button
                            onClick={fetchSitrep}
                            disabled={loading}
                            className="text-slate-400 hover:text-white transition-colors"
                        >
                            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
                        </button>
                    </div>
                </div>

                {/* System Overview Strip */}
                <div className="p-3 grid grid-cols-2 md:grid-cols-4 gap-3">
                    <MiniStat label="Balance" value={`$${overview.balance.toFixed(2)}`} color="text-emerald-400" />
                    <MiniStat label="Risk Budget" value={`$${overview.risk_budget.toFixed(2)}`} color="text-blue-400" />
                    <MiniStat label="Metabolism" value={overview.metabolism} color={overview.metabolism === 'PREDATOR' ? 'text-orange-400' : 'text-cyan-400'} />
                    <MiniStat label="Regime" value={overview.regime} color="text-purple-400" />
                    {overview.drawdown_locked && (
                        <div className="col-span-2 md:col-span-4 bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-1.5 flex items-center gap-2">
                            <AlertTriangle size={14} className="text-red-400" />
                            <span className="text-[10px] text-red-400 font-mono uppercase tracking-wider">DRAWDOWN LOCK ACTIVE</span>
                        </div>
                    )}
                </div>
            </div>

            {/* === ACTIVE SIGNALS === */}
            <div className="bg-holon-card rounded-xl border border-slate-700/50 overflow-hidden">
                <div className="bg-slate-900/50 p-3 border-b border-slate-700/50 flex justify-between items-center">
                    <div className="flex items-center gap-2">
                        <Target className="text-cyan-400" size={16} />
                        <span className="font-orbitron text-xs tracking-wider text-holon-accent">ACTIVE SIGNALS</span>
                    </div>
                    <span className="text-[10px] bg-cyan-500/20 text-cyan-400 px-2 py-0.5 rounded border border-cyan-500/30">
                        {signals.count} ACTIVE
                    </span>
                </div>

                <div className="p-3">
                    {signals.count === 0 ? (
                        <div className="text-center py-6">
                            <Target className="mx-auto mb-2 opacity-20 text-slate-500" size={28} />
                            <p className="text-sm text-holon-dim">No active signals. Scanning...</p>
                        </div>
                    ) : (
                        <div className="space-y-3">
                            {signals.items.map((sig, idx) => (
                                <SignalCard key={idx} signal={sig} getSetupColor={getSetupColor} getQualityBadge={getQualityBadge} />
                            ))}
                        </div>
                    )}
                </div>
            </div>

            {/* === POSITIONS === */}
            {positions.count > 0 && (
                <div className="bg-holon-card rounded-xl border border-slate-700/50 overflow-hidden">
                    <div className="bg-slate-900/50 p-3 border-b border-slate-700/50 flex justify-between items-center">
                        <div className="flex items-center gap-2">
                            <BarChart3 className="text-emerald-400" size={16} />
                            <span className="font-orbitron text-xs tracking-wider text-holon-accent">POSITIONS</span>
                        </div>
                        <span className={clsx("text-xs font-mono font-bold", positions.unrealized_pnl >= 0 ? 'text-emerald-400' : 'text-red-400')}>
                            {positions.unrealized_pnl >= 0 ? '+' : ''}${positions.unrealized_pnl.toFixed(2)}
                        </span>
                    </div>
                    <div className="divide-y divide-slate-800">
                        {positions.items.map((pos, idx) => (
                            <div key={idx} className="p-3 flex items-center justify-between hover:bg-white/5 transition-colors">
                                <div className="flex items-center gap-3">
                                    <div className={clsx("w-1.5 h-8 rounded-full", pos.direction === 'BUY' ? 'bg-emerald-500' : 'bg-red-500')} />
                                    <div>
                                        <span className="text-sm font-bold text-white font-mono">{pos.symbol}</span>
                                        <div className="flex items-center gap-2 mt-0.5">
                                            <span className={clsx("text-[9px] font-mono", pos.direction === 'BUY' ? 'text-emerald-500' : 'text-red-500')}>
                                                {pos.direction === 'BUY' ? 'LONG' : 'SHORT'}
                                            </span>
                                            <span className="text-[9px] text-slate-500 font-mono">{pos.strategy}</span>
                                        </div>
                                    </div>
                                </div>
                                <div className="text-right">
                                    <span className={clsx("text-sm font-bold font-mono", pos.pnl >= 0 ? 'text-emerald-400' : 'text-red-400')}>
                                        {pos.pnl >= 0 ? '+' : ''}${pos.pnl.toFixed(2)}
                                    </span>
                                    <p className={clsx("text-[10px] font-mono", pos.pnl_pct >= 0 ? 'text-emerald-500/70' : 'text-red-500/70')}>
                                        {pos.pnl_pct >= 0 ? '+' : ''}{pos.pnl_pct.toFixed(2)}%
                                    </p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* === ARBITRAGE OPPORTUNITIES === */}
            {arbitrage.count > 0 && (
                <div className="bg-holon-card rounded-xl border border-slate-700/50 overflow-hidden">
                    <div className="bg-slate-900/50 p-3 border-b border-slate-700/50 flex justify-between items-center">
                        <div className="flex items-center gap-2">
                            <Zap className="text-yellow-400" size={16} />
                            <span className="font-orbitron text-xs tracking-wider text-yellow-400">FUNDING & SPREADS</span>
                        </div>
                        <span className="text-[10px] text-yellow-400 font-mono">{arbitrage.count} ASSETS</span>
                    </div>
                    <div className="p-3 grid grid-cols-2 md:grid-cols-3 gap-2">
                        {arbitrage.items.slice(0, 12).map((arb, idx) => (
                            <div key={idx} className="bg-slate-900/60 rounded-lg border border-slate-700/40 p-2.5">
                                <div className="flex justify-between items-center">
                                    <span className="text-[10px] font-mono text-white font-bold">{arb.symbol.replace('/USDT', '')}</span>
                                    <span className={clsx("text-[10px] font-mono font-bold",
                                        Math.abs(arb.funding_apy) >= 50 ? 'text-yellow-400' :
                                            Math.abs(arb.funding_apy) >= 20 ? 'text-emerald-400' : 'text-slate-400'
                                    )}>
                                        {arb.funding_apy > 0 ? '+' : ''}{arb.funding_apy.toFixed(1)}%
                                    </span>
                                </div>
                                {arb.spread_pct !== 0 && (
                                    <div className="mt-1 text-[9px] text-slate-500 font-mono">
                                        Spread: {arb.spread_pct.toFixed(3)}%
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* === MINI EQUITY CHART === */}
            {chart && chart.points > 0 && (
                <div className="bg-holon-card rounded-xl border border-slate-700/50 overflow-hidden p-4">
                    <div className="flex items-center gap-2 mb-3">
                        <TrendingUp className="text-emerald-400" size={14} />
                        <span className="font-orbitron text-xs tracking-wider text-holon-accent">EQUITY TREND</span>
                        <span className="text-[9px] text-slate-500 font-mono ml-auto">{chart.points} pts</span>
                    </div>
                    <MiniSparkline data={chart.equity_history} />
                </div>
            )}
        </div>
    );
};

/* --- Sub-Components --- */

const MiniStat = ({ label, value, color }) => (
    <div className="bg-slate-900/40 rounded-lg border border-slate-800/50 p-2.5">
        <p className="text-[9px] font-mono text-holon-dim uppercase tracking-wider">{label}</p>
        <p className={clsx("text-sm font-bold font-orbitron mt-0.5", color)}>{value}</p>
    </div>
);

const SignalCard = ({ signal, getSetupColor, getQualityBadge }) => {
    const [expanded, setExpanded] = useState(false);
    const isLong = signal.direction === 'BUY';

    return (
        <div
            className="bg-slate-900/60 rounded-lg border border-slate-700/40 overflow-hidden cursor-pointer hover:border-slate-600/60 transition-all"
            onClick={() => setExpanded(!expanded)}
        >
            {/* Signal Header */}
            <div className="p-3 flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <div className={clsx("p-1.5 rounded-md", isLong ? 'bg-emerald-500/15' : 'bg-red-500/15')}>
                        {isLong ? <ArrowUpRight size={16} className="text-emerald-400" /> : <ArrowDownRight size={16} className="text-red-400" />}
                    </div>
                    <div>
                        <div className="flex items-center gap-2">
                            <span className="text-sm font-bold text-white font-mono">{signal.symbol}</span>
                            <span className={clsx("text-[9px] px-1.5 py-0.5 rounded border font-mono", getSetupColor(signal.setup_type))}>
                                {signal.setup_type}
                            </span>
                        </div>
                        <p className="text-[10px] text-slate-400 mt-0.5 max-w-xs truncate">{signal.rationale}</p>
                    </div>
                </div>
                <div className="text-right flex items-center gap-3">
                    <span className={clsx("text-[10px] px-1.5 py-0.5 rounded border font-mono font-bold", getQualityBadge(signal.quality))}>
                        {signal.quality}
                    </span>
                    <div>
                        <p className="text-xs font-mono text-white">${signal.price.toFixed(2)}</p>
                        <p className="text-[9px] text-slate-500 font-mono">
                            Conv: {(signal.conviction * 100).toFixed(0)}%
                        </p>
                    </div>
                </div>
            </div>

            {/* Expanded Details */}
            {expanded && (
                <div className="border-t border-slate-700/40 p-3 bg-slate-950/30">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        {/* Sizing */}
                        <div>
                            <p className="text-[9px] text-holon-dim uppercase font-mono mb-1">Sizing</p>
                            <p className="text-xs text-white font-mono">${signal.sizing.notional_usd.toFixed(2)}</p>
                            <p className="text-[9px] text-slate-500 font-mono">
                                Margin: ${signal.sizing.margin_usd.toFixed(2)} @ {signal.sizing.leverage.toFixed(1)}x
                            </p>
                        </div>
                        {/* Returns */}
                        <div>
                            <p className="text-[9px] text-holon-dim uppercase font-mono mb-1">Target</p>
                            <p className="text-xs text-emerald-400 font-mono">+{signal.returns.tp_pct.toFixed(2)}%</p>
                            <p className="text-[9px] text-emerald-500/60 font-mono">
                                +${signal.returns.expected_return_usd.toFixed(2)}
                            </p>
                        </div>
                        <div>
                            <p className="text-[9px] text-holon-dim uppercase font-mono mb-1">Stop</p>
                            <p className="text-xs text-red-400 font-mono">-{signal.returns.sl_pct.toFixed(2)}%</p>
                            <p className="text-[9px] text-red-500/60 font-mono">
                                -${signal.returns.max_loss_usd.toFixed(2)}
                            </p>
                        </div>
                        <div>
                            <p className="text-[9px] text-holon-dim uppercase font-mono mb-1">R:R Ratio</p>
                            <p className={clsx("text-xs font-mono font-bold",
                                signal.returns.rr_ratio >= 2 ? 'text-emerald-400' :
                                    signal.returns.rr_ratio >= 1 ? 'text-yellow-400' : 'text-red-400'
                            )}>
                                {signal.returns.rr_ratio.toFixed(2)}:1
                            </p>
                        </div>
                    </div>
                    {/* Raw Reason */}
                    <div className="mt-2 pt-2 border-t border-slate-800/50">
                        <p className="text-[9px] text-slate-600 font-mono">RAW: {signal.raw_reason}</p>
                    </div>
                </div>
            )}
        </div>
    );
};

const MiniSparkline = ({ data = [] }) => {
    if (data.length < 2) return <p className="text-xs text-holon-dim text-center py-2">Insufficient data</p>;

    const values = data.map(d => d.y || 0);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    const h = 60;
    const w = 100; // percentage

    const points = values.map((v, i) => {
        const x = (i / (values.length - 1)) * 100;
        const y = h - ((v - min) / range) * h;
        return `${x},${y}`;
    }).join(' ');

    const isUp = values[values.length - 1] >= values[0];

    return (
        <svg viewBox={`0 0 100 ${h}`} className="w-full h-16" preserveAspectRatio="none">
            <defs>
                <linearGradient id="sparkGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={isUp ? '#10b981' : '#ef4444'} stopOpacity="0.3" />
                    <stop offset="100%" stopColor={isUp ? '#10b981' : '#ef4444'} stopOpacity="0.0" />
                </linearGradient>
            </defs>
            <polygon
                points={`0,${h} ${points} 100,${h}`}
                fill="url(#sparkGrad)"
            />
            <polyline
                points={points}
                fill="none"
                stroke={isUp ? '#10b981' : '#ef4444'}
                strokeWidth="1.5"
                vectorEffect="non-scaling-stroke"
            />
        </svg>
    );
};

export default SitrepPanel;
