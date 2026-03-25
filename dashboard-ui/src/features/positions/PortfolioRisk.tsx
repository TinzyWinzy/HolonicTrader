import React from 'react';
import WidgetPanel from '../../components/ui/WidgetPanel';
import DataCell from '../../components/ui/DataCell';
import { useMarketStore } from '../../store/useMarketStore';

const PortfolioRisk: React.FC = () => {
    const portfolioHealth = useMarketStore((s) => s.portfolioHealth);
    const positions = useMarketStore((s) => s.positions);
    const equity = useMarketStore((s) => s.equity);
    const doomsday = useMarketStore((s) => s.doomsday);

    const totalPnl = positions.reduce((s, p) => s + (p.pnl || 0), 0);
    const longCount = positions.filter((p) => p.direction === 'BUY').length;
    const shortCount = positions.filter((p) => p.direction === 'SELL').length;
    const arbCount = positions.filter((p) =>
        p.strategy?.includes('ARB') || p.strategy?.includes('FUNDING')
    ).length;

    const metrics = [
        {
            label: 'EQUITY',
            value: equity,
            format: 'usd' as const,
            color: 'var(--text-primary)',
        },
        {
            label: 'DRAWDOWN',
            value: (portfolioHealth.drawdown_pct || 0) * 100,
            format: 'pct' as const,
            color: portfolioHealth.drawdown_pct > 0.1 ? 'var(--color-negative)' : 'var(--color-positive)',
        },
        {
            label: 'MARGIN USED',
            value: (portfolioHealth.margin_utilization || 0) * 100,
            format: 'pct' as const,
            color: portfolioHealth.margin_utilization > 0.7 ? 'var(--color-negative)' : 'var(--text-primary)',
        },
        {
            label: 'RISK BUDGET',
            value: portfolioHealth.risk_budget || 0,
            format: 'usd' as const,
            color: 'var(--text-primary)',
        },
        {
            label: 'SESSION PNL',
            value: totalPnl,
            format: 'pnl' as const,
            color: totalPnl >= 0 ? 'var(--color-positive)' : 'var(--color-negative)',
        },
    ];

    const defconColor = (() => {
        const lvl = doomsday.defcon_level;
        if (lvl <= 1) return 'var(--defcon-1)';
        if (lvl <= 2) return 'var(--defcon-2)';
        if (lvl <= 3) return 'var(--defcon-3)';
        if (lvl <= 4) return 'var(--defcon-4)';
        return 'var(--defcon-5)';
    })();

    return (
        <WidgetPanel title="Portfolio & Risk" accent="var(--color-negative)">
            {/* DEFCON Banner */}
            <div
                className="flex items-center justify-between p-2 rounded mb-3"
                style={{ background: 'var(--bg-elevated)', border: `1px solid ${defconColor}33` }}
            >
                <span className="text-[9px] font-mono uppercase" style={{ color: 'var(--text-dim)' }}>
                    THREAT LEVEL
                </span>
                <div className="flex items-center gap-2">
                    <span className="font-mono text-lg font-bold" style={{ color: defconColor }}>
                        DEFCON {doomsday.defcon_level}
                    </span>
                    {doomsday.crisis_active && (
                        <span className="animate-pulse text-[9px] font-mono px-1 rounded" style={{ background: 'rgba(255,23,68,0.2)', color: 'var(--color-negative)' }}>
                            CRISIS
                        </span>
                    )}
                </div>
            </div>

            {/* Risk Metrics Grid */}
            <div className="grid grid-cols-2 gap-2 mb-3">
                {metrics.map((m) => (
                    <div
                        key={m.label}
                        className="p-2 rounded"
                        style={{ background: 'var(--bg-elevated)' }}
                    >
                        <div className="text-[8px] font-mono uppercase tracking-wider mb-1" style={{ color: 'var(--text-dim)' }}>
                            {m.label}
                        </div>
                        <DataCell value={m.value} format={m.format} decimals={2} className="text-sm" />
                    </div>
                ))}
            </div>

            {/* Exposure Breakdown */}
            <div className="p-2 rounded" style={{ background: 'var(--bg-elevated)' }}>
                <div className="text-[8px] font-mono uppercase tracking-wider mb-2" style={{ color: 'var(--text-dim)' }}>
                    EXPOSURE
                </div>
                <div className="flex items-center gap-3 text-[10px] font-mono">
                    <div className="flex items-center gap-1">
                        <div className="w-2 h-2 rounded-full bg-[var(--color-long)]" />
                        <span style={{ color: 'var(--text-muted)' }}>LONG</span>
                        <span style={{ color: 'var(--color-long)' }}>{longCount}</span>
                    </div>
                    <div className="flex items-center gap-1">
                        <div className="w-2 h-2 rounded-full bg-[var(--color-short)]" />
                        <span style={{ color: 'var(--text-muted)' }}>SHORT</span>
                        <span style={{ color: 'var(--color-short)' }}>{shortCount}</span>
                    </div>
                    <div className="flex items-center gap-1">
                        <div className="w-2 h-2 rounded-full bg-[var(--color-arb)]" />
                        <span style={{ color: 'var(--text-muted)' }}>ARB</span>
                        <span style={{ color: 'var(--color-arb)' }}>{arbCount}</span>
                    </div>
                </div>
            </div>
        </WidgetPanel>
    );
};

export default PortfolioRisk;
