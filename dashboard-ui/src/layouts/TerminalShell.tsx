import React, { useEffect } from 'react';
import hotkeys from 'hotkeys-js';
import { useMarketStore } from '../store/useMarketStore';
import WebSocketManager from '../services/WebSocketManager';
import StatusBar from '../components/ui/StatusBar';
import CommandPalette from '../components/ui/CommandPalette';
import DataCell from '../components/ui/DataCell';
import type { TabId } from '../types/market';

const TABS: { id: TabId; label: string; key: string }[] = [
    { id: 'market', label: 'MARKET', key: '1' },
    { id: 'positions', label: 'POSITIONS', key: '2' },
    { id: 'signals', label: 'SIGNALS', key: '3' },
    { id: 'arbitrage', label: 'ARBITRAGE', key: '4' },
    { id: 'risk', label: 'RISK', key: '5' },
    { id: 'logs', label: 'LOGS', key: '6' },
];

interface TerminalShellProps {
    children: React.ReactNode;
}

const TerminalShell: React.FC<TerminalShellProps> = ({ children }) => {
    const activeTab = useMarketStore((s) => s.activeTab);
    const setActiveTab = useMarketStore((s) => s.setActiveTab);
    const togglePalette = useMarketStore((s) => s.toggleCommandPalette);
    const isConnected = useMarketStore((s) => s.isConnected);
    const equity = useMarketStore((s) => s.equity);
    const pnl = useMarketStore((s) => s.pnl);
    const regime = useMarketStore((s) => s.regime);
    const systemStatus = useMarketStore((s) => s.systemStatus);
    const healthScore = useMarketStore((s) => s.healthScore);
    const doomsday = useMarketStore((s) => s.doomsday);

    // Keyboard shortcuts
    useEffect(() => {
        hotkeys('ctrl+k, command+k', (e) => {
            e.preventDefault();
            togglePalette();
        });

        TABS.forEach((tab) => {
            hotkeys(tab.key, (e) => {
                // Don't trigger if typing in an input
                if ((e.target as HTMLElement).tagName === 'INPUT') return;
                e.preventDefault();
                setActiveTab(tab.id);
            });
        });

        return () => {
            hotkeys.unbind('ctrl+k, command+k');
            TABS.forEach((t) => hotkeys.unbind(t.key));
        };
    }, [setActiveTab, togglePalette]);

    const defconColor = (() => {
        const lvl = doomsday.defcon_level;
        if (lvl <= 1) return 'var(--defcon-1)';
        if (lvl <= 2) return 'var(--defcon-2)';
        if (lvl <= 3) return 'var(--defcon-3)';
        if (lvl <= 4) return 'var(--defcon-4)';
        return 'var(--defcon-5)';
    })();

    return (
        <div className="h-screen flex flex-col" style={{ background: 'var(--bg-primary)', color: 'var(--text-primary)' }}>
            {/* ─── Top Header Bar ─────────────────────────────────────── */}
            <div
                className="flex items-center justify-between px-4 h-10 border-b shrink-0"
                style={{ background: 'var(--bg-header)', borderColor: 'var(--border-default)' }}
            >
                {/* Left: System Identity */}
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                        <div
                            className={`w-2 h-2 rounded-full ${isConnected ? 'bg-[var(--color-positive)]' : 'bg-[var(--color-negative)]'}`}
                            style={isConnected ? { boxShadow: '0 0 6px var(--color-positive)' } : undefined}
                        />
                        <span className="font-mono text-[11px] font-bold tracking-wider" style={{ color: 'var(--accent-crypto)' }}>
                            HOLONIC TERMINAL
                        </span>
                    </div>

                    <div className="h-4 w-px" style={{ background: 'var(--border-accent)' }} />

                    <span
                        className="font-mono text-[10px] uppercase tracking-wider px-2 py-0.5 rounded"
                        style={{
                            color: systemStatus === 'SOLVENT' || systemStatus === 'RUNNING' ? 'var(--color-positive)' : 'var(--color-warning)',
                            background: systemStatus === 'SOLVENT' || systemStatus === 'RUNNING' ? 'rgba(0,230,118,0.1)' : 'rgba(255,145,0,0.1)',
                        }}
                    >
                        {systemStatus}
                    </span>

                    {/* DEFCON */}
                    <div className="flex items-center gap-1">
                        <span className="text-[9px] font-mono" style={{ color: 'var(--text-dim)' }}>DEF</span>
                        <span
                            className="font-mono text-[11px] font-bold"
                            style={{ color: defconColor }}
                        >
                            {doomsday.defcon_level}
                        </span>
                    </div>
                </div>

                {/* Center: KPIs */}
                <div className="flex items-center gap-6">
                    <div className="flex items-center gap-2">
                        <span className="text-[9px] font-mono" style={{ color: 'var(--text-dim)' }}>EQUITY</span>
                        <DataCell value={equity} format="usd" decimals={2} />
                    </div>
                    <div className="flex items-center gap-2">
                        <span className="text-[9px] font-mono" style={{ color: 'var(--text-dim)' }}>PNL</span>
                        <DataCell value={pnl} format="pnl" decimals={2} />
                    </div>
                    <div className="flex items-center gap-2">
                        <span className="text-[9px] font-mono" style={{ color: 'var(--text-dim)' }}>REGIME</span>
                        <span className="font-mono text-[11px] font-medium" style={{ color: 'var(--accent-forex)' }}>
                            {regime}
                        </span>
                    </div>
                    <div className="flex items-center gap-2">
                        <span className="text-[9px] font-mono" style={{ color: 'var(--text-dim)' }}>HP</span>
                        <div className="w-16 h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--bg-elevated)' }}>
                            <div
                                className="h-full rounded-full transition-all duration-500"
                                style={{
                                    width: `${Math.min(100, healthScore)}%`,
                                    background: healthScore > 70 ? 'var(--color-positive)' : healthScore > 40 ? 'var(--color-warning)' : 'var(--color-negative)',
                                }}
                            />
                        </div>
                        <span className="font-mono text-[10px]" style={{ color: 'var(--text-muted)' }}>
                            {healthScore.toFixed(0)}%
                        </span>
                    </div>
                </div>

                {/* Right: Controls & Clock */}
                <div className="flex items-center gap-3">
                    <button
                        onClick={() => {
                            if (window.confirm('🚨 EMERGENCY: ARE YOU SURE YOU WANT TO PANIC? THIS WILL ATTEMPT TO CLOSE ALL POSITIONS.')) {
                                WebSocketManager.getInstance().sendCommand('panic');
                            }
                        }}
                        className="px-3 py-1 rounded text-[10px] font-mono font-bold border hover:bg-[var(--color-negative)] hover:text-white transition-all animate-pulse"
                        style={{
                            borderColor: 'var(--color-negative)',
                            color: 'var(--color-negative)',
                            boxShadow: '0 0 8px rgba(255, 23, 68, 0.2)'
                        }}
                    >
                        🚨 PANIC ESTOP
                    </button>

                    <button
                        onClick={togglePalette}
                        className="px-2 py-1 rounded text-[10px] font-mono border hover:bg-[var(--bg-hover)] transition-colors"
                        style={{ borderColor: 'var(--border-accent)', color: 'var(--text-muted)' }}
                    >
                        ⌘K
                    </button>
                    <ClockDisplay />
                </div>
            </div>

            {/* ─── Tab Bar ────────────────────────────────────────────── */}
            <div
                className="flex items-center gap-0 px-2 h-8 border-b shrink-0"
                style={{ background: 'var(--bg-primary)', borderColor: 'var(--border-default)' }}
            >
                {TABS.map((tab) => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className="relative px-4 h-full text-[10px] font-mono font-semibold tracking-widest transition-colors"
                        style={{
                            color: activeTab === tab.id ? 'var(--accent-crypto)' : 'var(--text-dim)',
                            background: activeTab === tab.id ? 'var(--bg-card)' : 'transparent',
                            borderBottom: activeTab === tab.id ? '2px solid var(--accent-crypto)' : '2px solid transparent',
                        }}
                    >
                        <span className="text-[8px] mr-1" style={{ color: 'var(--text-dim)' }}>{tab.key}</span>
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* ─── Content Area ───────────────────────────────────────── */}
            <div className="flex-1 overflow-hidden">
                {children}
            </div>

            {/* ─── Status Bar ─────────────────────────────────────────── */}
            <StatusBar />

            {/* ─── Command Palette Overlay ──────────────────────────── */}
            <CommandPalette />
        </div>
    );
};

// ─── Clock Component ───────────────────────────────────────────────────────────

const ClockDisplay: React.FC = () => {
    const [time, setTime] = React.useState(new Date());

    useEffect(() => {
        const interval = setInterval(() => setTime(new Date()), 1000);
        return () => clearInterval(interval);
    }, []);

    return (
        <span className="font-mono text-[11px] tabular-nums" style={{ color: 'var(--text-secondary)' }}>
            {time.toLocaleTimeString('en-US', { hour12: false })}
        </span>
    );
};

export default TerminalShell;
