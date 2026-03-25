import React from 'react';
import WidgetPanel from '../../components/ui/WidgetPanel';
import { useMarketStore } from '../../store/useMarketStore';

const DefconWidget: React.FC = () => {
    const doomsday = useMarketStore((s) => s.doomsday);

    const levels = [
        { level: 1, label: 'NUCLEAR WAR', desc: 'Maximum threat — all assets at risk', color: '#ff1744' },
        { level: 2, label: 'SEVERE', desc: 'Critical market conditions detected', color: '#ff5722' },
        { level: 3, label: 'ELEVATED', desc: 'Heightened volatility and risk', color: '#ff9100' },
        { level: 4, label: 'GUARDED', desc: 'Some concerns, monitoring closely', color: '#ffb300' },
        { level: 5, label: 'NORMAL', desc: 'Standard market conditions', color: '#00e676' },
    ];

    const current = levels.find((l) => l.level === doomsday.defcon_level) || levels[4];

    return (
        <WidgetPanel title="Doomsday Monitor" accent={current.color}>
            {/* DEFCON Display */}
            <div className="flex items-center justify-center mb-4">
                <div
                    className="relative w-20 h-20 rounded-full flex items-center justify-center"
                    style={{
                        background: `${current.color}15`,
                        border: `2px solid ${current.color}`,
                        boxShadow: `0 0 20px ${current.color}22`,
                    }}
                >
                    <div className="text-center">
                        <div className="text-[9px] font-mono" style={{ color: current.color }}>
                            DEFCON
                        </div>
                        <div
                            className="text-2xl font-mono font-bold"
                            style={{ color: current.color }}
                        >
                            {doomsday.defcon_level}
                        </div>
                    </div>
                    {doomsday.crisis_active && (
                        <div
                            className="absolute -top-1 -right-1 w-3 h-3 rounded-full animate-pulse"
                            style={{ background: 'var(--color-negative)', boxShadow: '0 0 8px var(--color-negative)' }}
                        />
                    )}
                </div>
            </div>

            {/* Level Label */}
            <div className="text-center mb-4">
                <div className="text-[11px] font-mono font-bold" style={{ color: current.color }}>
                    {current.label}
                </div>
                <div className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>
                    {current.desc}
                </div>
            </div>

            {/* Level Indicators */}
            <div className="flex items-center gap-1">
                {levels.map((l) => (
                    <div
                        key={l.level}
                        className="flex-1 h-1.5 rounded-full transition-all duration-500"
                        style={{
                            background: doomsday.defcon_level <= l.level ? l.color : 'var(--bg-elevated)',
                            opacity: doomsday.defcon_level <= l.level ? 1 : 0.3,
                        }}
                    />
                ))}
            </div>
        </WidgetPanel>
    );
};

export default DefconWidget;
