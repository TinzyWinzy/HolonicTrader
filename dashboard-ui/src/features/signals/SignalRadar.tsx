import React from 'react';
import WidgetPanel from '../../components/ui/WidgetPanel';
import Badge from '../../components/ui/Badge';
import DataCell from '../../components/ui/DataCell';
import { useMarketStore } from '../../store/useMarketStore';

const SignalRadar: React.FC = () => {
    const signals = useMarketStore((s) => s.radar);

    return (
        <WidgetPanel
            title="SIGNAL RADAR"
            accent="var(--accent-crypto)"
            flush
            rightContent={
                <div className="flex items-center gap-2">
                    <span className="text-[9px] font-mono" style={{ color: 'var(--text-dim)' }}>
                        SCANNING
                    </span>
                    <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-positive)] animate-pulse-fast" />
                </div>
            }
        >
            {signals.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-[var(--text-dim)] font-mono space-y-2 py-12">
                    <div className="w-12 h-12 border border-[var(--border-subtle)] rounded-full flex items-center justify-center">
                        <div className="w-1 h-1 bg-[var(--text-dim)] rounded-full animate-ping" />
                    </div>
                    <span className="text-xs">LISTENING THE VOID...</span>
                </div>
            ) : (
                <table className="bb-table">
                    <thead>
                        <tr>
                            <th>Asset</th>
                            <th>Side</th>
                            <th>Quality</th>
                            <th className="text-right">Conviction</th>
                            <th className="text-right">Entry</th>
                            <th className="text-right">Target</th>
                            <th className="text-right">Stop</th>
                            <th>Thesis</th>
                        </tr>
                    </thead>
                    <tbody>
                        {signals.map((sig, idx) => {
                            const conv = sig.conviction ?? (sig.score ? sig.score / 100 : 0);
                            const isHighConviction = conv >= 0.8;

                            return (
                                <tr
                                    key={`${sig.symbol}-${idx}`}
                                    className={`
                                        transition-colors hover:bg-[var(--bg-hover)]
                                        ${isHighConviction ? 'bg-[rgba(0,230,118,0.05)]' : ''}
                                    `}
                                >
                                    <td className="font-bold font-mono">
                                        <div className="flex items-center gap-2">
                                            {isHighConviction && <span className="text-[8px]">🔥</span>}
                                            <span style={{ color: isHighConviction ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
                                                {sig.symbol?.replace('/USDT', '').replace('/USD', '')}
                                            </span>
                                        </div>
                                    </td>
                                    <td>
                                        <Badge variant={sig.direction === 'BUY' ? 'long' : sig.direction === 'SELL' ? 'short' : 'info'}>
                                            {sig.direction === 'BUY' ? 'LONG' : 'SHORT'}
                                        </Badge>
                                    </td>
                                    <td>
                                        <div className="flex items-center gap-1">
                                            <div
                                                className={`w-1.5 h-1.5 rounded-full ${isHighConviction ? 'animate-pulse' : ''}`}
                                                style={{
                                                    background: sig.quality === 'HIGH' ? 'var(--color-positive)' :
                                                        sig.quality === 'MEDIUM' ? 'var(--color-warning)' : 'var(--text-dim)'
                                                }}
                                            />
                                            <span className="text-[10px] opacity-80">{sig.quality}</span>
                                        </div>
                                    </td>
                                    <td className="text-right">
                                        <div className="flex items-center justify-end gap-2">
                                            <div className="w-12 h-1.5 rounded-sm overflow-hidden bg-[var(--bg-elevated)]">
                                                <div
                                                    className="h-full rounded-sm transition-all duration-500"
                                                    style={{
                                                        width: `${Math.min(100, conv * 100)}%`,
                                                        background: conv > 0.7 ? 'var(--color-positive)' : conv > 0.4 ? 'var(--color-warning)' : 'var(--text-dim)',
                                                        boxShadow: conv > 0.8 ? '0 0 8px var(--color-positive)' : 'none'
                                                    }}
                                                />
                                            </div>
                                            <span className="text-[10px] font-mono font-bold" style={{ color: 'var(--text-primary)' }}>
                                                {(conv * 100).toFixed(0)}%
                                            </span>
                                        </div>
                                    </td>
                                    <td className="text-right">
                                        <DataCell value={sig.price || 0} format="price" decimals={2} flashOnChange={false} />
                                    </td>
                                    <td className="text-right" style={{ color: 'var(--color-positive)' }}>
                                        {sig.tp ? <DataCell value={sig.tp} format="price" decimals={2} flashOnChange={false} /> : <span className="opacity-20">—</span>}
                                    </td>
                                    <td className="text-right" style={{ color: 'var(--color-negative)' }}>
                                        {sig.sl ? <DataCell value={sig.sl} format="price" decimals={2} flashOnChange={false} /> : <span className="opacity-20">—</span>}
                                    </td>
                                    <td>
                                        <span
                                            className="text-[9px] font-mono uppercase tracking-wide opacity-60 truncate block max-w-[150px]"
                                            title={sig.reason}
                                        >
                                            {sig.reason || 'Wait'}
                                        </span>
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            )}
        </WidgetPanel>
    );
};

export default SignalRadar;
