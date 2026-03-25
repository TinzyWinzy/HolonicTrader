import React, { useEffect, useRef } from 'react';
import { createChart, IChartApi, ColorType, AreaSeries } from 'lightweight-charts';
import WidgetPanel from '../../components/ui/WidgetPanel';
import { useMarketStore } from '../../store/useMarketStore';

const EquityChart: React.FC = () => {
    const equityHistory = useMarketStore((s) => s.equityHistory);
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);

    useEffect(() => {
        const container = chartContainerRef.current;
        if (!container) return;

        const chart = createChart(container, {
            layout: {
                background: { type: ColorType.Solid, color: '#0a0a0a' },
                textColor: '#666',
                fontSize: 9,
                fontFamily: 'JetBrains Mono, monospace',
            },
            grid: {
                vertLines: { color: '#1a1a1a' },
                horzLines: { color: '#1a1a1a' },
            },
            rightPriceScale: {
                borderColor: '#222',
            },
            timeScale: {
                borderColor: '#222',
                visible: false,
            },
            crosshair: {
                vertLine: { visible: false },
                horzLine: { color: '#333', width: 1, style: 2, labelBackgroundColor: '#222' },
            },
            handleScroll: false,
            handleScale: false,
        });

        chartRef.current = chart;

        const areaSeries = chart.addSeries(AreaSeries, {
            lineColor: '#00e676',
            topColor: 'rgba(0, 230, 118, 0.15)',
            bottomColor: 'rgba(0, 230, 118, 0.01)',
            lineWidth: 2,
            priceFormat: { type: 'custom', formatter: (val: number) => `$${val.toFixed(0)}` },
        });

        if (equityHistory.length > 0) {
            const now = Math.floor(Date.now() / 1000);
            const data = equityHistory.map((pt, idx) => ({
                time: (now - (equityHistory.length - idx) * 60) as any,
                value: pt.y,
            }));
            areaSeries.setData(data);
            chart.timeScale().fitContent();
        }

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
    }, [equityHistory]);

    return (
        <WidgetPanel title="Equity Curve" accent="var(--color-positive)" flush>
            <div ref={chartContainerRef} className="w-full h-full min-h-[120px]" />
        </WidgetPanel>
    );
};

export default EquityChart;
