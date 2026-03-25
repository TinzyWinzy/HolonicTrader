import React from 'react';
import { Spline, TrendingUp, HelpCircle } from 'lucide-react';
import { ComposedChart, Line, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import clsx from 'clsx';

const MonteCarloPanel = ({ data = {} }) => {
    if (!data || Object.keys(data).length === 0) {
        return (
            <div className="bg-holon-card rounded-xl border border-slate-700/50 shadow-xl p-4 flex items-center justify-center min-h-[200px]">
                <div className="text-center text-slate-500">
                    <Spline className="mx-auto mb-2 opacity-50" size={32} />
                    <p className="text-xs font-mono">WAITING FOR MONTE CARLO RESULTS...</p>
                </div>
            </div>
        );
    }

    const { paths = [], p50 = [], p95_upper = [], p95_lower = [], horizon = 0, current_price = 0 } = data;

    // Format chart data (assuming arrays are same length)
    const chartData = p50.map((val, i) => ({
        step: i,
        p50: val,
        p95_upper: p95_upper[i] || val,
        p95_lower: p95_lower[i] || val,
        base: current_price
    }));

    const projectedReturn = p50.length > 0 ? ((p50[p50.length - 1] - current_price) / current_price) * 100 : 0;
    const isPositive = projectedReturn >= 0;

    return (
        <div className="bg-holon-card rounded-xl border border-slate-700/50 shadow-xl overflow-hidden flex flex-col h-full">
            {/* Header */}
            <div className="bg-gradient-to-r from-orange-900/30 to-amber-900/20 p-3 border-b border-slate-700/50 flex justify-between items-center">
                <div className="flex items-center gap-2">
                    <Spline className="text-orange-400" size={18} />
                    <span className="font-orbitron text-sm tracking-wider text-orange-400">MONTE CARLO PROJECTION</span>
                </div>
                <div className="flex items-center gap-2">
                    <span className={clsx("text-xs font-mono font-bold", isPositive ? "text-emerald-400" : "text-rose-400")}>
                        {isPositive ? '+' : ''}{projectedReturn.toFixed(2)}%
                    </span>
                    <span className="text-[10px] text-slate-500">({horizon}H)</span>
                </div>
            </div>

            {/* Quick Stats */}
            <div className="flex gap-4 p-2 px-3 text-[10px] font-mono text-slate-400 border-b border-slate-700/30 bg-slate-900/10">
                <span>PATHS: {paths}</span>
                <span>CONFIDENCE: 95%</span>
            </div>

            {/* Projection Chart */}
            <div className="flex-1 min-h-[150px] p-2">
                <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} vertical={false} />
                        <XAxis dataKey="step" hide />
                        <YAxis domain={['auto', 'auto']} hide />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '12px' }}
                            itemStyle={{ color: '#fb923c' }}
                            labelStyle={{ display: 'none' }}
                            formatter={(value) => [value.toFixed(2), 'Price']}
                        />
                        {/* Confidence Interval Area */}
                        <Area type="monotone" dataKey="p95_upper" stroke="none" fill="#fb923c" fillOpacity={0.1} />
                        <Area type="monotone" dataKey="p95_lower" stroke="none" fill="#fb923c" fillOpacity={0.1} />

                        {/* Median Path */}
                        <Line type="monotone" dataKey="p50" stroke="#fb923c" strokeWidth={2} dot={false} />

                        {/* Baseline */}
                        <Line type="monotone" dataKey="base" stroke="#94a3b8" strokeDasharray="3 3" strokeWidth={1} dot={false} />
                    </ComposedChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default MonteCarloPanel;
