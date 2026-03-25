import React from 'react';
import { useMarketStore } from '../../store/useMarketStore';

const StatusBar: React.FC = () => {
    const isConnected = useMarketStore((s) => s.isConnected);
    const scanning = useMarketStore((s) => s.scanning);
    const lastScan = useMarketStore((s) => s.lastScan);
    const lastUpdate = useMarketStore((s) => s.lastUpdate);

    const formatTime = (ts: number) => {
        if (!ts) return '—';
        const d = new Date(ts * 1000);
        return d.toLocaleTimeString('en-US', { hour12: false });
    };

    return (
        <div
            className="flex items-center justify-between px-4 h-6 border-t text-[10px] font-mono"
            style={{
                background: 'var(--bg-header)',
                borderColor: 'var(--border-default)',
                color: 'var(--text-muted)',
            }}
        >
            {/* Left: Connection */}
            <div className="flex items-center gap-3">
                <div className="flex items-center gap-1.5">
                    <div
                        className={`w-1.5 h-1.5 rounded-full ${isConnected ? 'bg-[var(--color-positive)]' : 'bg-[var(--color-negative)]'}`}
                        style={isConnected ? { boxShadow: '0 0 4px var(--color-positive)' } : undefined}
                    />
                    <span>{isConnected ? 'CONNECTED' : 'DISCONNECTED'}</span>
                </div>

                {scanning && (
                    <div className="flex items-center gap-1 text-[var(--accent-crypto)]">
                        <span className="animate-pulse">●</span>
                        <span>SCANNING</span>
                    </div>
                )}
            </div>

            {/* Center: Hotkey Hints */}
            <div className="flex items-center gap-4 text-[var(--text-dim)]">
                <span>
                    <kbd className="px-1 py-0.5 bg-[var(--bg-elevated)] rounded text-[9px] border border-[var(--border-accent)]">Ctrl+K</kbd>
                    {' '}Command
                </span>
                <span>
                    <kbd className="px-1 py-0.5 bg-[var(--bg-elevated)] rounded text-[9px] border border-[var(--border-accent)]">1-6</kbd>
                    {' '}Tabs
                </span>
            </div>

            {/* Right: Timestamps */}
            <div className="flex items-center gap-4">
                {lastScan && <span>SCAN: {new Date(lastScan).toLocaleTimeString('en-US', { hour12: false })}</span>}
                <span>UPD: {formatTime(lastUpdate)}</span>
                <span className="text-[var(--text-dim)]">{new Date().toLocaleTimeString('en-US', { hour12: false })}</span>
            </div>
        </div>
    );
};

export default StatusBar;
