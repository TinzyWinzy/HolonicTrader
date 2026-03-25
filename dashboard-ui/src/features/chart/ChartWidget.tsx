import React, { useEffect, useRef, useMemo, useCallback } from 'react';
import { createChart, IChartApi, ColorType, CandlestickSeries, HistogramSeries } from 'lightweight-charts';
import { generateMockOHLC, generateMockVolumeData } from '../../services/mockFeed';
import WidgetPanel from '../../components/ui/WidgetPanel';
import { useMarketStore } from '../../store/useMarketStore';

const TIMEFRAMES = ['1m', '5m', '15m', '1H', '4H', '1D'] as const;

const ChartWidget: React.FC = () => {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const selectedSymbol = useMarketStore((s) => s.selectedSymbol);
    const [timeframe, setTimeframe] = React.useState<string>('1H');

    const symbol = selectedSymbol || 'BTC/USD';

    const { candles, volumes } = useMemo(() => {
        const intervalMap: Record<string, number> = {
            '1m': 60, '5m': 300, '15m': 900, '1H': 3600, '4H': 14400, '1D': 86400,
        };
        const c = generateMockOHLC(symbol, 200, intervalMap[timeframe] || 3600);
        const v = generateMockVolumeData(c);
        return { candles: c, volumes: v };
    }, [symbol, timeframe]);

    useEffect(() => {
        const container = chartContainerRef.current;
        if (!container) return;

        const chart = createChart(container, {
            layout: {
                background: { type: ColorType.Solid, color: '#0a0a0a' },
                textColor: '#666',
                fontSize: 10,
                fontFamily: 'JetBrains Mono, monospace',
            },
            grid: {
                vertLines: { color: '#1a1a1a' },
                horzLines: { color: '#1a1a1a' },
            },
            crosshair: {
                mode: 0,
                vertLine: { color: '#333', width: 1, style: 2, labelBackgroundColor: '#222' },
                horzLine: { color: '#333', width: 1, style: 2, labelBackgroundColor: '#222' },
            },
            rightPriceScale: {
                borderColor: '#222',
                scaleMargins: { top: 0.1, bottom: 0.25 },
            },
            timeScale: {
                borderColor: '#222',
                timeVisible: true,
                secondsVisible: false,
            },
            handleScroll: { mouseWheel: true, pressedMouseMove: true },
            handleScale: { mouseWheel: true, pinch: true },
        });

        chartRef.current = chart;

        const candleSeries = chart.addSeries(CandlestickSeries, {
            upColor: '#00e676',
            downColor: '#ff1744',
            borderUpColor: '#00e676',
            borderDownColor: '#ff1744',
            wickUpColor: '#00e67688',
            wickDownColor: '#ff174488',
        });

        candleSeries.setData(candles as any[]);

        const volumeSeries = chart.addSeries(HistogramSeries, {
            priceScaleId: 'volume',
            priceFormat: { type: 'volume' },
        });

        chart.priceScale('volume').applyOptions({
            scaleMargins: { top: 0.8, bottom: 0 },
        });

        volumeSeries.setData(volumes as any[]);
        chart.timeScale().fitContent();

        const resizeObserver = new ResizeObserver((entries) => {
            if (chartRef.current) {
                const { width, height } = entries[0].contentRect;
                chart.applyOptions({ width, height });
            }
        });

        resizeObserver.observe(container);

        return () => {
            resizeObserver.disconnect();
            chartRef.current = null;
            chart.remove();
        };
    }, [candles, volumes]);

    return (
        <WidgetPanel
            title={`${symbol} — ${timeframe}`}
            accent="var(--accent-crypto)"
            flush
            rightContent={
                <div className="flex items-center gap-1">
                    {TIMEFRAMES.map((tf) => (
                        <button
                            key={tf}
                            onClick={() => setTimeframe(tf)}
                            className="px-2 py-0.5 text-[9px] font-mono rounded transition-colors"
                            style={{
                                background: timeframe === tf ? 'var(--bg-hover)' : 'transparent',
                                color: timeframe === tf ? 'var(--accent-crypto)' : 'var(--text-dim)',
                                border: timeframe === tf ? '1px solid var(--border-accent)' : '1px solid transparent',
                            }}
                        >
                            {tf}
                        </button>
                    ))}
                </div>
            }
        >
            <div ref={chartContainerRef} className="w-full h-full min-h-[300px]" />
        </WidgetPanel>
    );
};

export default ChartWidget;
