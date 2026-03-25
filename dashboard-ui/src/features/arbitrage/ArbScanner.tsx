import React from 'react';
import WidgetPanel from '../../components/ui/WidgetPanel';
import Badge from '../../components/ui/Badge';
import { useMarketStore } from '../../store/useMarketStore';

const ArbScanner: React.FC = () => {
    const arbitrage = useMarketStore((s) => s.arbitrage);

    // Sort: opportunities first, then by absolute funding APY
    const sorted = [...arbitrage].sort((a, b) => {
        if (a.has_opportunity && !b.has_opportunity) return -1;
        if (!a.has_opportunity && b.has_opportunity) return 1;
        return Math.abs(b.funding_apy) - Math.abs(a.funding_apy);
    });

    const getFundingColor = (apy: number): string => {
        if (Math.abs(apy) > 30) return 'var(--color-negative)';
        if (Math.abs(apy) > 10) return 'var(--color-warning)';
        if (apy > 0) return 'var(--color-positive)';
        return 'var(--text-muted)';
    };

    return (
        <WidgetPanel
            title="Arbitrage Radar"
            accent="var(--color-arb)"
            flush
            rightContent={
                <span className="text-[9px] font-mono" style={{ color: 'var(--text-dim)' }}>
                    {arbitrage.filter((a) => a.has_opportunity).length} OPPORTUNITIES
                </span>
            }
        >
            {sorted.length === 0 ? (
                <div className="flex items-center justify-center h-24 text-[var(--text-dim)] text-xs font-mono">
                    No arbitrage data
                </div>
            ) : (
                <table className="bb-table">
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th className="text-right">Funding APY</th>
                            <th className="text-right">Spread</th>
                            <th>Signal</th>
                            <th>Confidence</th>
                            <th>Reason</th>
                        </tr>
                    </thead>
                    <tbody>
                        {sorted.map((opp, idx) => (
                            <tr
                                key={opp.symbol || idx}
                                style={opp.has_opportunity ? { background: 'rgba(171,71,188,0.05)' } : undefined}
                            >
                                <td>
                                    <span className="font-semibold">
                                        {opp.symbol?.replace('/USDT', '').replace('/USD', '')}
                                    </span>
                                </td>
                                <td className="text-right">
                                    <span
                                        className="font-mono text-[11px] font-semibold"
                                        style={{ color: getFundingColor(opp.funding_apy) }}
                                    >
                                        {opp.funding_apy > 0 ? '+' : ''}{opp.funding_apy.toFixed(1)}%
                                    </span>
                                </td>
                                <td className="text-right">
                                    <span
                                        className="font-mono text-[11px]"
                                        style={{ color: Math.abs(opp.spread_pct) > 0.1 ? 'var(--accent-crypto)' : 'var(--text-muted)' }}
                                    >
                                        {opp.spread_pct.toFixed(2)}%
                                    </span>
                                </td>
                                <td>
                                    {opp.signal ? (
                                        <Badge variant={opp.signal === 'BUY' ? 'long' : 'short'}>{opp.signal}</Badge>
                                    ) : (
                                        <span className="text-[10px]" style={{ color: 'var(--text-dim)' }}>—</span>
                                    )}
                                </td>
                                <td>
                                    {opp.confidence > 0 ? (
                                        <div className="flex items-center gap-1">
                                            <div className="w-8 h-1 rounded-full overflow-hidden" style={{ background: 'var(--bg-elevated)' }}>
                                                <div
                                                    className="h-full rounded-full"
                                                    style={{
                                                        width: `${Math.min(100, opp.confidence * 100)}%`,
                                                        background: 'var(--color-arb)',
                                                    }}
                                                />
                                            </div>
                                            <span className="text-[9px] font-mono" style={{ color: 'var(--text-dim)' }}>
                                                {(opp.confidence * 100).toFixed(0)}
                                            </span>
                                        </div>
                                    ) : (
                                        <span className="text-[10px]" style={{ color: 'var(--text-dim)' }}>—</span>
                                    )}
                                </td>
                                <td>
                                    <span
                                        className="text-[10px] font-mono truncate block max-w-[100px]"
                                        style={{ color: 'var(--text-muted)' }}
                                        title={opp.reason}
                                    >
                                        {opp.reason || '—'}
                                    </span>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            )}
        </WidgetPanel>
    );
};

export default ArbScanner;
