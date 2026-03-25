import React from 'react';
import { BarChart2, Layers, ArrowUpRight, ArrowDownRight, Activity } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import clsx from 'clsx';

const OrderFlowPanel = ({ data = {} }) => {
    if (!data || Object.keys(data).length === 0) {
        return (
            <div className="bg-holon-card rounded-xl border border-slate-700/50 shadow-xl p-4 flex items-center justify-center min-h-[200px]">
                <div className="text-center text-slate-500">
                    <Layers className="mx-auto mb-2 opacity-50" size={32} />
                    <p className="text-xs font-mono">WAITING FOR ORDER FLOW DATA...</p>
                </div>
            </div>
        );
    }

    const { current_cvd = 0, imbalance_ratio = 0, cvd_history = [] } = data;

    // Format chart data
    const chartData = cvd_history.map((val, idx) => ({
        idx,
        cvd: val
    }));

    const isBullish = current_cvd > 0;
    const imbalanceColor = imbalance_ratio > 1.0 ? 'text-emerald-400' : (imbalance_ratio < 1.0 ? 'text-rose-400' : 'text-slate-400');

    return (
        <div className="bg-holon-card rounded-xl border border-slate-700/50 shadow-xl overflow-hidden flex flex-col h-full">
            {/* Header */}
            <div className="bg-gradient-to-r from-blue-900/30 to-cyan-900/20 p-3 border-b border-slate-700/50 flex justify-between items-center">
                <div className="flex items-center gap-2">
                    <BarChart2 className="text-cyan-400" size={18} />
                    <span className="font-orbitron text-sm tracking-wider text-cyan-400">ORDER FLOW</span>
                </div>
                <div className={clsx("flex items-center gap-1 text-xs font-mono px-2 py-0.5 rounded border bg-opacity-20", isBullish ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/30" : "bg-rose-500/20 text-rose-400 border-rose-500/30")}>
                    {isBullish ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}
                    CVD {current_cvd > 0 ? '+' : ''}{current_cvd.toFixed(2)}
                </div>
            </div>

            {/* Metrics */}
            <div className="grid grid-cols-2 gap-2 p-3 border-b border-slate-700/50 bg-slate-900/20">
                <div className="bg-slate-800/40 p-2 rounded border border-slate-700/30">
                    <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-0.5">Imbalance Ratio</p>
                    <p className={clsx("text-lg font-mono", imbalanceColor)}>{imbalance_ratio.toFixed(2)}x</p>
                </div>
                <div className="bg-slate-800/40 p-2 rounded border border-slate-700/30">
                    <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-0.5">Flow Regime</p>
                    <p className="text-lg font-mono text-slate-300">{imbalance_ratio > 1.5 ? 'ACCUMULATION' : (imbalance_ratio < 0.6 ? 'DISTRIBUTION' : 'NEUTRAL')}</p>
                </div>
            </div>

            {/* CVD Chart */}
            <div className="flex-1 min-h-[150px] p-2">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData}>
                        <defs>
                            <linearGradient id="colorCvd" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3} />
                                <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <ReferenceLine y={0} stroke="#475569" strokeDasharray="3 3" />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '12px' }}
                            itemStyle={{ color: '#22d3ee' }}
                            labelStyle={{ display: 'none' }}
                            formatter={(value) => [value.toFixed(2), 'CVD']}
                        />
                        <Area type="monotone" dataKey="cvd" stroke="#06b6d4" fillOpacity={1} fill="url(#colorCvd)" />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default OrderFlowPanel;
