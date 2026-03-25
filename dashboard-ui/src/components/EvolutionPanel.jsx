import React from 'react';
import { Activity, Zap, Dna, GitBranch, Crosshair } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import clsx from 'clsx';

const EvolutionPanel = ({ data = {} }) => {
    // Check if data exists
    if (!data || Object.keys(data).length === 0) {
        return (
            <div className="bg-holon-card rounded-xl border border-slate-700/50 shadow-xl p-4 flex items-center justify-center min-h-[200px]">
                <div className="text-center text-slate-500">
                    <Dna className="mx-auto mb-2 opacity-50" size={32} />
                    <p className="text-xs font-mono">WAITING FOR EVOLUTION DATA...</p>
                </div>
            </div>
        );
    }

    const { generation = 0, best_fitness = 0, avg_fitness = 0, mutation_rate = 0, population_size = 0, history = [] } = data;

    // Format history for chart
    const chartData = history.slice(-20).map((pt, idx) => ({
        gen: pt.gen || idx,
        best: pt.best || 0,
        avg: pt.avg || 0
    }));

    return (
        <div className="bg-holon-card rounded-xl border border-slate-700/50 shadow-xl overflow-hidden flex flex-col h-full">
            {/* Header */}
            <div className="bg-gradient-to-r from-purple-900/30 to-indigo-900/20 p-3 border-b border-slate-700/50 flex justify-between items-center">
                <div className="flex items-center gap-2">
                    <Dna className="text-purple-400" size={18} />
                    <span className="font-orbitron text-sm tracking-wider text-purple-400">EVOLUTION ENGINE</span>
                </div>
                <div className="flex items-center gap-3 text-xs font-mono text-slate-400">
                    <span className="flex items-center gap-1">
                        <GitBranch size={12} /> GEN {generation}
                    </span>
                    <span className="flex items-center gap-1 text-purple-300">
                        <Zap size={12} /> FIT {best_fitness.toFixed(4)}
                    </span>
                </div>
            </div>

            {/* Metrics Grid */}
            <div className="grid grid-cols-2 gap-2 p-3 border-b border-slate-700/50 bg-slate-900/20">
                <div className="bg-slate-800/40 p-2 rounded border border-slate-700/30">
                    <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-0.5">Mutation Rate</p>
                    <p className="text-lg font-mono text-indigo-300">{(mutation_rate * 100).toFixed(1)}%</p>
                </div>
                <div className="bg-slate-800/40 p-2 rounded border border-slate-700/30">
                    <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-0.5">Population</p>
                    <p className="text-lg font-mono text-indigo-300">{population_size}</p>
                </div>
            </div>

            {/* Fitness Chart */}
            <div className="flex-1 min-h-[150px] p-2">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} vertical={false} />
                        <XAxis dataKey="gen" hide />
                        <YAxis hide domain={['auto', 'auto']} />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '12px' }}
                            itemStyle={{ color: '#e2e8f0' }}
                            labelStyle={{ display: 'none' }}
                        />
                        <Line type="monotone" dataKey="best" stroke="#c084fc" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="avg" stroke="#6366f1" strokeWidth={1} strokeDasharray="5 5" dot={false} />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default EvolutionPanel;
