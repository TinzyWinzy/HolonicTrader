import React, { useState, useEffect, useRef } from 'react';
import { useMarketStore } from '../../store/useMarketStore';
import type { TabId } from '../../types/market';

const CommandPalette: React.FC = () => {
    const isOpen = useMarketStore((s) => s.commandPaletteOpen);
    const toggle = useMarketStore((s) => s.toggleCommandPalette);
    const setTab = useMarketStore((s) => s.setActiveTab);
    const setSymbol = useMarketStore((s) => s.setSelectedSymbol);
    const prices = useMarketStore((s) => s.prices);

    const [query, setQuery] = useState('');
    const inputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        if (isOpen) {
            setQuery('');
            setTimeout(() => inputRef.current?.focus(), 50);
        }
    }, [isOpen]);

    if (!isOpen) return null;

    // Build command list
    type Command = { label: string; hint: string; action: () => void };
    const commands: Command[] = [
        { label: 'Market Overview', hint: 'Tab', action: () => { setTab('market'); toggle(); } },
        { label: 'Positions', hint: 'Tab', action: () => { setTab('positions'); toggle(); } },
        { label: 'Signals', hint: 'Tab', action: () => { setTab('signals'); toggle(); } },
        { label: 'Arbitrage', hint: 'Tab', action: () => { setTab('arbitrage'); toggle(); } },
        { label: 'Risk', hint: 'Tab', action: () => { setTab('risk'); toggle(); } },
        { label: 'Logs', hint: 'Tab', action: () => { setTab('logs'); toggle(); } },
        // Symbol commands from live prices
        ...Object.keys(prices).map((sym) => ({
            label: sym,
            hint: `$${prices[sym]?.toFixed(2)}`,
            action: () => { setSymbol(sym); setTab('market'); toggle(); },
        })),
    ];

    const filtered = query
        ? commands.filter((c) => c.label.toLowerCase().includes(query.toLowerCase()))
        : commands;

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Escape') toggle();
        if (e.key === 'Enter' && filtered.length > 0) {
            filtered[0].action();
        }
    };

    return (
        <div
            className="fixed inset-0 z-50 flex items-start justify-center pt-[15vh]"
            onClick={toggle}
            style={{ background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(4px)' }}
        >
            <div
                className="w-full max-w-lg rounded-lg overflow-hidden"
                onClick={(e) => e.stopPropagation()}
                style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-accent)' }}
            >
                {/* Input */}
                <div className="flex items-center px-4 border-b" style={{ borderColor: 'var(--border-default)' }}>
                    <span className="text-[var(--accent-crypto)] font-mono text-sm mr-2">›</span>
                    <input
                        ref={inputRef}
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Search symbols, commands..."
                        className="flex-1 bg-transparent py-3 text-sm text-[var(--text-primary)] font-mono focus:outline-none placeholder:text-[var(--text-dim)]"
                    />
                    <kbd className="px-1.5 py-0.5 text-[9px] font-mono bg-[var(--bg-card)] rounded border border-[var(--border-accent)] text-[var(--text-dim)]">
                        ESC
                    </kbd>
                </div>

                {/* Results */}
                <div className="max-h-72 overflow-auto bb-scrollbar">
                    {filtered.length === 0 ? (
                        <div className="px-4 py-8 text-center text-[var(--text-dim)] text-sm font-mono">
                            No results
                        </div>
                    ) : (
                        filtered.slice(0, 15).map((cmd, i) => (
                            <button
                                key={i}
                                onClick={cmd.action}
                                className="w-full flex items-center justify-between px-4 py-2 text-left text-sm font-mono hover:bg-[var(--bg-hover)] transition-colors"
                                style={{ color: 'var(--text-primary)' }}
                            >
                                <span>{cmd.label}</span>
                                <span className="text-[10px] text-[var(--text-dim)]">{cmd.hint}</span>
                            </button>
                        ))
                    )}
                </div>
            </div>
        </div>
    );
};

export default CommandPalette;
