import React, { useEffect, useRef, useState } from 'react';
import clsx from 'clsx';
import { Terminal, Filter, XCircle, AlertTriangle, CheckCircle, Info } from 'lucide-react';

const LiveLog = ({ logs = [] }) => {
    const containerRef = useRef(null);
    const [filter, setFilter] = useState('ALL'); // ALL, ERROR, WARN, TRADE

    // Auto-scroll only within the log container
    useEffect(() => {
        if (containerRef.current) {
            containerRef.current.scrollTop = containerRef.current.scrollHeight;
        }
    }, [logs, filter]);

    const filteredLogs = logs.filter(log => {
        if (filter === 'ALL') return true;
        const txt = (log.msg || '').toUpperCase();
        if (filter === 'ERROR') return txt.includes('ERROR') || txt.includes('FATAL') || txt.includes('CRITICAL');
        if (filter === 'WARN') return txt.includes('WARNING') || txt.includes('ALERT');
        if (filter === 'TRADE') return txt.includes('PROFIT') || txt.includes('ORDER') || txt.includes('FILLED');
        return true;
    });

    return (
        <div className="bg-holon-card rounded-xl border border-slate-700/50 flex flex-col shadow-xl overflow-hidden h-full min-h-[300px] flex-1">
            {/* Header */}
            <div className="bg-slate-900/50 p-2 border-b border-slate-700/50 flex justify-between items-center">
                <div className="flex items-center gap-2 text-holon-accent pl-2">
                    <Terminal size={14} />
                    <span className="font-orbitron text-[10px] tracking-wider">SYSTEM KERNEL</span>
                </div>

                {/* Filter Controls */}
                <div className="flex gap-1 pr-1">
                    {['ALL', 'TRADE', 'WARN', 'ERROR'].map(f => (
                        <button
                            key={f}
                            onClick={() => setFilter(f)}
                            className={clsx(
                                "text-[9px] px-2 py-0.5 rounded font-mono transition-colors border",
                                filter === f
                                    ? "bg-slate-700 text-white border-slate-600"
                                    : "text-slate-500 border-transparent hover:bg-slate-800 hover:text-slate-400"
                            )}
                        >
                            {f}
                        </button>
                    ))}
                </div>
            </div>

            {/* Log Output */}
            <div
                ref={containerRef}
                className="flex-1 overflow-y-auto p-2 font-mono text-[10px] space-y-0.5 bg-black/40 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent"
            >
                {filteredLogs.length === 0 && (
                    <div className="text-center text-slate-600 py-12 italic">
                        No active kernel logs...
                    </div>
                )}

                {filteredLogs.map((log, i) => {
                    const txt = log.msg || '';
                    const upper = txt.toUpperCase();
                    const isErr = upper.includes('ERROR') || upper.includes('FATAL') || upper.includes('CRITICAL');
                    const isWarn = upper.includes('WARNING') || upper.includes('ALERT');
                    const isSuccess = upper.includes('SUCCESS') || upper.includes('PROFIT') || upper.includes('FILLED');
                    const isTrade = upper.includes('ORDER') || upper.includes('TRADE');

                    return (
                        <div key={i} className="flex gap-2 items-start hover:bg-white/5 px-1 rounded-sm">
                            <span className="text-slate-600 shrink-0 select-none w-14 text-right opacity-50">
                                {log.time?.split(' ')[1] || log.time}
                            </span>

                            <div className="flex items-start gap-1.5 break-all">
                                {isErr ? <XCircle size={10} className="text-red-500 mt-0.5 shrink-0" /> :
                                    isWarn ? <AlertTriangle size={10} className="text-amber-500 mt-0.5 shrink-0" /> :
                                        isSuccess ? <CheckCircle size={10} className="text-emerald-500 mt-0.5 shrink-0" /> :
                                            isTrade ? <Terminal size={10} className="text-blue-400 mt-0.5 shrink-0" /> :
                                                <span className="w-2.5 inline-block" />}

                                <span className={clsx("leading-snug", {
                                    'text-red-400 font-bold': isErr,
                                    'text-amber-400': isWarn,
                                    'text-emerald-400': isSuccess,
                                    'text-blue-300': isTrade,
                                    'text-slate-400': !isErr && !isWarn && !isSuccess && !isTrade
                                })}>
                                    {txt}
                                </span>
                            </div>
                        </div>
                    );
                })}

                {/* Scroll Anchor */}
                <div className="h-1" />
            </div>
        </div>
    );
};

export default LiveLog;
