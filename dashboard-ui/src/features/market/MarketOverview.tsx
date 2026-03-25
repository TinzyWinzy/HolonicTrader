import React from 'react';
import WidgetPanel from '../../components/ui/WidgetPanel';
import DataCell from '../../components/ui/DataCell';
import { useMarketStore } from '../../store/useMarketStore';

const MarketOverview: React.FC = () => {
    const prices = useMarketStore((s) => s.prices);
    const setSymbol = useMarketStore((s) => s.setSelectedSymbol);
    const selectedSymbol = useMarketStore((s) => s.selectedSymbol);

    const symbols = Object.entries(prices);

    // Group by type
    const crypto = symbols.filter(([s]) => !s.includes('EUR') && !s.includes('GBP') && !s.includes('JPY') && !s.includes('XAU'));
    const forex = symbols.filter(([s]) => s.includes('EUR') || s.includes('GBP') || s.includes('JPY') || s.includes('XAU'));

    const renderGroup = (items: [string, number][], label: string, accentColor: string) => (
        <div className="mb-4">
            <div className="flex items-center gap-2 mb-2 px-1">
                <div className="w-1.5 h-1.5 rounded-full" style={{ background: accentColor, boxShadow: `0 0 8px ${accentColor}` }} />
                <span className="text-[10px] font-mono font-bold uppercase tracking-widest" style={{ color: 'var(--text-secondary)' }}>
                    {label}
                </span>
                <span className="text-[9px] font-mono" style={{ color: 'var(--text-dim)' }}>
                    // {items.length}
                </span>
            </div>
            <div className="grid grid-cols-2 gap-2">
                {items.map(([symbol, price]) => {
                    const isSelected = selectedSymbol === symbol;
                    return (
                        <button
                            key={symbol}
                            onClick={() => setSymbol(symbol)}
                            className={`
                                relative flex flex-col justify-between px-3 py-2 rounded border transition-all duration-200 group
                                ${isSelected ? 'bg-[var(--bg-hover)] border-[var(--color-info)]' : 'bg-[var(--bg-elevated)] border-transparent hover:border-[var(--border-subtle)]'}
                            `}
                            style={{
                                boxShadow: isSelected ? '0 0 12px rgba(0, 229, 255, 0.15)' : 'none'
                            }}
                        >
                            <div className="flex items-center justify-between w-full mb-1">
                                <span
                                    className="font-mono text-[11px] font-bold tracking-tight"
                                    style={{ color: isSelected ? accentColor : 'var(--text-primary)' }}
                                >
                                    {symbol.replace('/USDT', '').replace('/USD', '')}
                                </span>
                                {isSelected && <div className="w-1.5 h-1.5 rounded-full bg-[var(--color-info)] animate-pulse" />}
                            </div>

                            <div className="flex w-full items-end justify-between">
                                <DataCell
                                    value={price}
                                    format="price"
                                    decimals={price < 1 ? 6 : price < 100 ? 4 : 2}
                                    className={`text-[12px] font-mono ${isSelected ? 'text-glow' : ''}`}
                                />
                            </div>
                        </button>
                    );
                })}
            </div>
        </div>
    );

    return (
        <WidgetPanel
            title="MARKET FEED"
            accent="var(--accent-crypto)"
            rightContent={
                <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-[var(--color-positive)] animate-pulse" />
                    <span className="text-[9px] font-mono" style={{ color: 'var(--text-dim)' }}>LIVE</span>
                </div>
            }
        >
            {symbols.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-48 space-y-4">
                    <div className="w-8 h-8 border-2 border-[var(--border-default)] border-t-[var(--accent-crypto)] rounded-full animate-spin" />
                    <span className="text-[var(--text-dim)] text-xs font-mono tracking-widest">INITIALIZING FEED...</span>
                </div>
            ) : (
                <div className="p-2 space-y-6">
                    {crypto.length > 0 && renderGroup(crypto, 'Crypto Assets', 'var(--accent-crypto)')}
                    {forex.length > 0 && renderGroup(forex, 'Forex & Comm', 'var(--accent-forex)')}
                </div>
            )}
        </WidgetPanel>
    );
};

export default MarketOverview;
