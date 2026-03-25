import React from 'react';
import clsx from 'clsx';
import { Shield, AlertTriangle, TrendingDown, TrendingUp, Minus } from 'lucide-react';

const ACTION_CONFIG = {
    URGENT_CLOSE: { icon: AlertTriangle, color: 'text-red-500', bg: 'bg-red-500/10', border: 'border-red-500/30', label: 'URGENT CLOSE', pulse: true },
    CLOSE: { icon: TrendingDown, color: 'text-orange-400', bg: 'bg-orange-500/10', border: 'border-orange-500/30', label: 'CLOSE' },
    REDUCE: { icon: Minus, color: 'text-amber-400', bg: 'bg-amber-500/10', border: 'border-amber-500/30', label: 'REDUCE' },
    STACK: { icon: TrendingUp, color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', label: 'STACK' },
};

export default function PositionHealthPanel({ managementSignals = [] }) {
    if (!managementSignals || managementSignals.length === 0) {
        return (
            <div className="bg-holon-card rounded-xl border border-slate-700/50 p-4">
                <div className="flex items-center gap-2 mb-3">
                    <Shield size={14} className="text-emerald-400" />
                    <span className="text-[10px] font-orbitron tracking-wider text-emerald-400 uppercase">Position Health</span>
                </div>
                <div className="flex items-center justify-center py-4">
                    <div className="flex items-center gap-2 text-emerald-400/60">
                        <div className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
                        <span className="text-xs font-mono">ALL POSITIONS HEALTHY</span>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="bg-holon-card rounded-xl border border-slate-700/50 overflow-hidden">
            <div className="bg-slate-900/50 p-3 border-b border-slate-700/50 flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Shield size={14} className="text-amber-400" />
                    <span className="text-[10px] font-orbitron tracking-wider text-amber-400 uppercase">Position Health Alerts</span>
                </div>
                <span className="text-[10px] font-mono text-red-400 animate-pulse">{managementSignals.length} ALERT{managementSignals.length > 1 ? 'S' : ''}</span>
            </div>

            <div className="divide-y divide-slate-800">
                {managementSignals.map((sig, idx) => {
                    const action = sig.direction || 'CLOSE';
                    const cfg = ACTION_CONFIG[action] || ACTION_CONFIG.CLOSE;
                    const Icon = cfg.icon;
                    const urgency = sig.conviction || 0;

                    return (
                        <div key={idx} className={clsx("p-3 transition-colors hover:bg-white/5", cfg.bg)}>
                            <div className="flex items-center justify-between mb-2">
                                <div className="flex items-center gap-2">
                                    <div className={clsx("p-1.5 rounded-md", cfg.bg, cfg.border, "border", cfg.pulse && "animate-pulse")}>
                                        <Icon size={12} className={cfg.color} />
                                    </div>
                                    <span className="font-bold text-white text-sm font-mono">{sig.symbol}</span>
                                </div>
                                <span className={clsx("text-[10px] font-orbitron tracking-wider px-2 py-0.5 rounded", cfg.bg, cfg.color, "border", cfg.border)}>
                                    {cfg.label}
                                </span>
                            </div>

                            {/* Urgency Bar */}
                            <div className="flex items-center gap-2 mb-1.5">
                                <span className="text-[9px] text-holon-dim font-mono w-12">URGENCY</span>
                                <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                    <div
                                        className={clsx("h-full rounded-full transition-all duration-500",
                                            urgency > 0.8 ? "bg-red-500" : urgency > 0.5 ? "bg-amber-500" : "bg-emerald-500"
                                        )}
                                        style={{ width: `${urgency * 100}%` }}
                                    />
                                </div>
                                <span className={clsx("text-[10px] font-mono font-bold", cfg.color)}>
                                    {(urgency * 100).toFixed(0)}%
                                </span>
                            </div>

                            {/* Reason */}
                            <p className="text-[10px] text-slate-400 font-mono leading-relaxed">
                                {sig.reason?.replace('[AI MANAGER] ', '')}
                            </p>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
