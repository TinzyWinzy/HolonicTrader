
import React, { useState, useEffect } from 'react';
import { Settings, Save, RefreshCw } from 'lucide-react';
import clsx from 'clsx';
import { useSocket } from '../context/SocketContext'; // Ensure context is available if needed, though we use REST here

const ConfigPanel = () => {
    const [allocation, setAllocation] = useState(0.20); // Default to safe fallback
    const [leverage, setLeverage] = useState(5.0);
    const [loading, setLoading] = useState(false);
    const [initialLoading, setInitialLoading] = useState(true);
    const [status, setStatus] = useState('');

    // Fetch config on mount
    const fetchConfig = async () => {
        try {
            const res = await fetch('http://localhost:5000/api/config');
            const data = await res.json();
            if (data.status === 'ok' && data.config) {
                setAllocation(data.config.max_allocation);
                setLeverage(data.config.leverage_cap);
            }
        } catch (e) {
            console.error("Config fetch error:", e);
            setStatus('Fetch Error');
        } finally {
            setInitialLoading(false);
        }
    };

    useEffect(() => {
        fetchConfig();
        // Optional: Poll every 10s to stay in sync if changed elsewhere
        const interval = setInterval(fetchConfig, 10000);
        return () => clearInterval(interval);
    }, []);

    const handleSave = async () => {
        setLoading(true);
        setStatus('Saving...');
        try {
            const res = await fetch('http://localhost:5000/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    max_allocation: Number(allocation),
                    leverage_cap: Number(leverage)
                })
            });
            const data = await res.json();
            if (data.status === 'ok') {
                setStatus('Saved!');
                // Update local state with confirmed server values
                if (data.updates) {
                    if (data.updates.max_allocation) setAllocation(data.updates.max_allocation);
                    if (data.updates.leverage_cap) setLeverage(data.updates.leverage_cap);
                }
                setTimeout(() => setStatus(''), 2000);
            } else {
                setStatus('Error: ' + data.message);
            }
        } catch (e) {
            setStatus('Network Error');
        }
        setLoading(false);
    };

    if (initialLoading) {
        return (
            <div className="bg-holon-card rounded-xl border border-slate-700/50 p-4 h-[200px] flex items-center justify-center animate-pulse">
                <div className="text-slate-500 font-orbitron text-xs">LOADING CONFIG...</div>
            </div>
        );
    }

    return (
        <div className="bg-holon-card rounded-xl border border-slate-700/50 p-4 transition-all hover:bg-slate-800/40 relative group">
            {/* Hover Refresh Button */}
            <button
                onClick={fetchConfig}
                className="absolute top-4 right-4 text-slate-500 opacity-0 group-hover:opacity-100 hover:text-holon-accent transition-all"
                title="Refresh Config"
            >
                <RefreshCw size={14} />
            </button>

            <div className="flex items-center gap-2 mb-4 text-holon-accent border-b border-slate-700/50 pb-2">
                <Settings size={18} />
                <h3 className="font-orbitron text-sm tracking-wider font-bold">SYSTEM CONFIG</h3>
            </div>

            <div className="space-y-6">
                {/* Allocation Slider */}
                <div>
                    <div className="flex justify-between text-xs text-slate-400 mb-2">
                        <span>Max Position Size</span>
                        <span className={clsx("font-mono font-bold", allocation > 0.3 ? "text-amber-400" : "text-emerald-400")}>
                            {(allocation * 100).toFixed(0)}%
                        </span>
                    </div>
                    <input
                        type="range"
                        min="0.01"
                        max="0.50"
                        step="0.01"
                        value={allocation}
                        onChange={(e) => setAllocation(Number(e.target.value))}
                        className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-holon-accent hover:accent-orange-400 transition-all"
                    />
                    <div className="flex justify-between text-[10px] text-slate-600 px-1 mt-1">
                        <span>1%</span>
                        <span>25%</span>
                        <span>50%</span>
                    </div>
                </div>

                {/* Leverage Input */}
                <div>
                    <div className="flex justify-between text-xs text-slate-400 mb-2">
                        <span>Global Leverage Cap</span>
                        <span className={clsx("font-mono font-bold", leverage > 10 ? "text-red-400" : "text-blue-400")}>
                            {leverage}x
                        </span>
                    </div>
                    <input
                        type="range"
                        min="1"
                        max="20"
                        step="1"
                        value={leverage}
                        onChange={(e) => setLeverage(Number(e.target.value))}
                        className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500 hover:accent-blue-400 transition-all"
                    />
                    <div className="flex justify-between text-[10px] text-slate-600 px-1 mt-1">
                        <span>1x</span>
                        <span>10x</span>
                        <span>20x</span>
                    </div>
                </div>

                {/* Save Button */}
                <button
                    onClick={handleSave}
                    disabled={loading}
                    className={clsx(
                        "w-full flex items-center justify-center gap-2 py-3 rounded font-orbitron text-xs font-bold transition-all border border-transparent",
                        loading
                            ? "bg-slate-800 text-slate-500 cursor-not-allowed"
                            : "bg-holon-accent/10 text-holon-accent hover:bg-holon-accent hover:text-white border-holon-accent/20 hover:border-holon-accent hover:shadow-[0_0_15px_rgba(249,115,22,0.4)] active:scale-95"
                    )}
                >
                    {loading ? (
                        <RefreshCw size={14} className="animate-spin" />
                    ) : (
                        <Save size={14} />
                    )}
                    {loading ? 'SYNCING...' : 'APPLY CONFIGURATION'}
                </button>

                {status && (
                    <div className={clsx(
                        "text-[10px] text-center font-mono py-1 rounded bg-slate-900/50 border",
                        status.includes('Error') ? "text-red-400 border-red-900/30" : "text-emerald-400 border-emerald-900/30"
                    )}>
                        {status}
                    </div>
                )}
            </div>
        </div>
    );
};

export default ConfigPanel;
