import React, { useEffect } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import TerminalShell from './layouts/TerminalShell';
import WebSocketManager from './services/WebSocketManager';
import { useMarketStore } from './store/useMarketStore';

// Feature Widgets
import MarketOverview from './features/market/MarketOverview';
import ChartWidget from './features/chart/ChartWidget';
import EquityChart from './features/chart/EquityChart';
import PositionsWidget from './features/positions/PositionsWidget';
import PortfolioRisk from './features/positions/PortfolioRisk';
import SignalRadar from './features/signals/SignalRadar';
import ArbScanner from './features/arbitrage/ArbScanner';
import OrderEntry from './features/orders/OrderEntry';
import DefconWidget from './features/risk/DefconWidget';
import SystemLog from './features/logs/SystemLog';

const queryClient = new QueryClient({
    defaultOptions: {
        queries: { refetchOnWindowFocus: false, retry: 2 },
    },
});

// ─── Tab Content Layouts ───────────────────────────────────────────────────────

const MarketTab: React.FC = () => (
    <div className="h-full grid grid-cols-12 gap-[2px] p-[2px]" style={{ background: 'var(--bg-primary)' }}>
        {/* Left: Market Overview + Order Entry */}
        <div className="col-span-3 flex flex-col gap-[2px] overflow-auto bb-scrollbar">
            <MarketOverview />
            <OrderEntry />
        </div>

        {/* Center: Chart + Equity */}
        <div className="col-span-6 flex flex-col gap-[2px]">
            <div className="flex-[3] min-h-0">
                <ChartWidget />
            </div>
            <div className="flex-1 min-h-0">
                <EquityChart />
            </div>
        </div>

        {/* Right: Positions + Signals */}
        <div className="col-span-3 flex flex-col gap-[2px] overflow-auto bb-scrollbar">
            <PositionsWidget />
            <SignalRadar />
        </div>
    </div>
);

const PositionsTab: React.FC = () => (
    <div className="h-full grid grid-cols-12 gap-[2px] p-[2px]" style={{ background: 'var(--bg-primary)' }}>
        <div className="col-span-8 flex flex-col gap-[2px] overflow-auto bb-scrollbar">
            <PositionsWidget />
            <EquityChart />
        </div>
        <div className="col-span-4 flex flex-col gap-[2px] overflow-auto bb-scrollbar">
            <PortfolioRisk />
            <OrderEntry />
        </div>
    </div>
);

const SignalsTab: React.FC = () => (
    <div className="h-full grid grid-cols-12 gap-[2px] p-[2px]" style={{ background: 'var(--bg-primary)' }}>
        <div className="col-span-8 overflow-auto bb-scrollbar">
            <SignalRadar />
        </div>
        <div className="col-span-4 flex flex-col gap-[2px] overflow-auto bb-scrollbar">
            <ChartWidget />
            <OrderEntry />
        </div>
    </div>
);

const ArbitrageTab: React.FC = () => (
    <div className="h-full grid grid-cols-12 gap-[2px] p-[2px]" style={{ background: 'var(--bg-primary)' }}>
        <div className="col-span-8 overflow-auto bb-scrollbar">
            <ArbScanner />
        </div>
        <div className="col-span-4 flex flex-col gap-[2px] overflow-auto bb-scrollbar">
            <DefconWidget />
            <OrderEntry />
        </div>
    </div>
);

const RiskTab: React.FC = () => (
    <div className="h-full grid grid-cols-12 gap-[2px] p-[2px]" style={{ background: 'var(--bg-primary)' }}>
        <div className="col-span-4 flex flex-col gap-[2px] overflow-auto bb-scrollbar">
            <DefconWidget />
            <PortfolioRisk />
        </div>
        <div className="col-span-8 flex flex-col gap-[2px] overflow-auto bb-scrollbar">
            <PositionsWidget />
            <EquityChart />
        </div>
    </div>
);

const LogsTab: React.FC = () => (
    <div className="h-full p-[2px]" style={{ background: 'var(--bg-primary)' }}>
        <SystemLog />
    </div>
);

// ─── Dashboard Content ─────────────────────────────────────────────────────────

const DashboardContent: React.FC = () => {
    const activeTab = useMarketStore((s) => s.activeTab);

    return (
        <div className="h-full">
            {activeTab === 'market' && <MarketTab />}
            {activeTab === 'positions' && <PositionsTab />}
            {activeTab === 'signals' && <SignalsTab />}
            {activeTab === 'arbitrage' && <ArbitrageTab />}
            {activeTab === 'risk' && <RiskTab />}
            {activeTab === 'logs' && <LogsTab />}
        </div>
    );
};

// ─── App Root ──────────────────────────────────────────────────────────────────

const App: React.FC = () => {
    // Initialize WebSocket on mount
    useEffect(() => {
        const ws = WebSocketManager.getInstance();
        ws.connect();
        return () => ws.disconnect();
    }, []);

    return (
        <QueryClientProvider client={queryClient}>
            <TerminalShell>
                <DashboardContent />
            </TerminalShell>
        </QueryClientProvider>
    );
};

export default App;
