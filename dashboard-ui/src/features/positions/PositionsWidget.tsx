import React from 'react';
import WidgetPanel from '../../components/ui/WidgetPanel';
import Badge from '../../components/ui/Badge';
import DataCell from '../../components/ui/DataCell';
import { useMarketStore } from '../../store/useMarketStore';

const PositionsWidget: React.FC = () => {
    const positions = useMarketStore((s) => s.positions);
    const portfolioHealth = useMarketStore((s) => s.portfolioHealth);

    const totalPnl = positions.reduce((s, p) => s + (p.pnl || 0), 0);
    const totalNotional = positions.reduce((s, p) => s + Math.abs(p.quantity * p.current_price), 0);

    return (
        <WidgetPanel
            title="Open Positions"
            accent="var(--color-info)"
            flush
            rightContent={
                <div className="flex items-center gap-3 text-[10px] font-mono">
                    <span style={{ color: 'var(--text-dim)' }}>{positions.length} OPEN</span>
                    <span style={{ color: 'var(--text-dim)' }}>NOT: </span>
                    <DataCell value={totalNotional} format="usd" decimals={0} />
                    <DataCell value={totalPnl} format="pnl" decimals={2} />
                </div>
            }
        >
            {/* Margin Bar */}
            {portfolioHealth.margin_utilization > 0 && (
                <div className="px-3 py-2 border-b" style={{ borderColor: 'var(--border-subtle)' }}>
                    <div className="flex items-center justify-between text-[9px] font-mono mb-1">
                        <span style={{ color: 'var(--text-dim)' }}>MARGIN</span>
                        <span style={{ color: 'var(--text-muted)' }}>
                            {(portfolioHealth.margin_utilization * 100).toFixed(1)}%
                        </span>
                    </div>
                    <div className="h-1 rounded-full overflow-hidden" style={{ background: 'var(--bg-elevated)' }}>
                        <div
                            className="h-full rounded-full transition-all duration-500"
                            style={{
                                width: `${Math.min(100, portfolioHealth.margin_utilization * 100)}%`,
                                background:
                                    portfolioHealth.margin_utilization > 0.8
                                        ? 'var(--color-negative)'
                                        : portfolioHealth.margin_utilization > 0.5
                                            ? 'var(--color-warning)'
                                            : 'var(--color-positive)',
                            }}
                        />
                    </div>
                </div>
            )}

            {/* Positions Table */}
            {positions.length === 0 ? (
                <div className="flex items-center justify-center h-24 text-[var(--text-dim)] text-xs font-mono">
                    No active positions
                </div>
            ) : (
                <table className="bb-table">
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Dir</th>
                            <th>Type</th>
                            <th className="text-right">Qty</th>
                            <th className="text-right">Entry</th>
                            <th className="text-right">Mark</th>
                            <th className="text-right">PnL</th>
                            <th className="text-right">%</th>
                        </tr>
                    </thead>
                    <tbody>
                        {positions.map((pos, idx) => {
                            const isArb = pos.strategy?.includes('ARB') || pos.strategy?.includes('FUNDING') || pos.strategy?.includes('BASIS');
                            return (
                                <tr key={pos.symbol || idx}>
                                    <td>
                                        <span className="font-semibold">{pos.symbol?.replace('/USDT', '').replace('/USD', '')}</span>
                                        {pos.leverage > 1 && (
                                            <span className="ml-1 text-[9px]" style={{ color: 'var(--accent-crypto)' }}>
                                                {pos.leverage}×
                                            </span>
                                        )}
                                    </td>
                                    <td>
                                        <Badge variant={pos.direction === 'BUY' ? 'long' : 'short'}>
                                            {pos.direction === 'BUY' ? 'LONG' : 'SHORT'}
                                        </Badge>
                                    </td>
                                    <td>
                                        <Badge variant={isArb ? 'arb' : 'dir'}>{isArb ? 'ARB' : 'DIR'}</Badge>
                                    </td>
                                    <td className="text-right">{pos.quantity?.toFixed(4)}</td>
                                    <td className="text-right" style={{ color: 'var(--text-muted)' }}>
                                        <DataCell value={pos.entry_price} format="price" decimals={2} flashOnChange={false} />
                                    </td>
                                    <td className="text-right">
                                        <DataCell value={pos.current_price} format="price" decimals={2} />
                                    </td>
                                    <td className="text-right">
                                        <DataCell value={pos.pnl} format="pnl" decimals={2} />
                                    </td>
                                    <td className="text-right">
                                        <DataCell value={pos.pnl_pct} format="pct" decimals={2} />
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

export default PositionsWidget;
