import { create } from 'zustand';
import type {
    HubState, Position, Signal, ArbitrageOpportunity,
    PortfolioHealth, DoomsdayStatus, EquityPoint, LogEntry,
    EvolutionData, OrderFlowData, MonteCarloData, TabId
} from '../types/market';

// ─── Store Shape ───────────────────────────────────────────────────────────────

interface MarketStore {
    // Connection
    isConnected: boolean;
    lastUpdate: number;

    // Market Data
    equity: number;
    pnl: number;
    regime: string;
    systemStatus: string;
    healthScore: number;
    scanning: boolean;
    lastScan: string | null;

    // Collections
    positions: Position[];
    prices: Record<string, number>;
    equityHistory: EquityPoint[];
    radar: Signal[];
    arbitrage: ArbitrageOpportunity[];
    logs: LogEntry[];

    // Risk
    portfolioHealth: PortfolioHealth;
    doomsday: DoomsdayStatus;

    // Subsystems
    evolution: EvolutionData;
    orderFlow: OrderFlowData;
    monteCarlo: MonteCarloData;

    // UI State
    activeTab: TabId;
    selectedSymbol: string | null;
    commandPaletteOpen: boolean;

    // Actions
    updateFromHubState: (data: Partial<HubState>) => void;
    setConnected: (connected: boolean) => void;
    setActiveTab: (tab: TabId) => void;
    setSelectedSymbol: (symbol: string | null) => void;
    toggleCommandPalette: () => void;
}

// ─── Store ─────────────────────────────────────────────────────────────────────

export const useMarketStore = create<MarketStore>((set) => ({
    // Connection
    isConnected: false,
    lastUpdate: 0,

    // Market Data
    equity: 0,
    pnl: 0,
    regime: 'UNKNOWN',
    systemStatus: 'DISCONNECTED',
    healthScore: 0,
    scanning: false,
    lastScan: null,

    // Collections
    positions: [],
    prices: {},
    equityHistory: [],
    radar: [],
    arbitrage: [],
    logs: [],

    // Risk
    portfolioHealth: { drawdown_pct: 0, margin_utilization: 0, risk_budget: 0, equity: 0 },
    doomsday: { defcon_level: 5, crisis_active: false },

    // Subsystems
    evolution: {},
    orderFlow: {},
    monteCarlo: {},

    // UI State
    activeTab: 'market',
    selectedSymbol: null,
    commandPaletteOpen: false,

    // ── Actions ──────────────────────────────────────────────────────────────

    updateFromHubState: (data) =>
        set((state) => ({
            equity: data.equity ?? state.equity,
            pnl: data.pnl ?? state.pnl,
            regime: data.regime ?? state.regime,
            systemStatus: data.system_status ?? state.systemStatus,
            healthScore: data.health_score ?? state.healthScore,
            scanning: data.scanning ?? state.scanning,
            lastScan: data.last_scan ?? state.lastScan,

            positions: data.positions ?? state.positions,
            prices: data.prices ?? state.prices,
            equityHistory: data.equity_history ?? state.equityHistory,
            radar: data.radar ?? state.radar,
            arbitrage: data.arbitrage ?? state.arbitrage,
            logs: data.logs ?? state.logs,

            portfolioHealth: data.portfolio_health ?? state.portfolioHealth,
            doomsday: data.doomsday ?? state.doomsday,

            evolution: data.evolution ?? state.evolution,
            orderFlow: data.order_flow ?? state.orderFlow,
            monteCarlo: data.monte_carlo ?? state.monteCarlo,

            lastUpdate: data.timestamp ?? Date.now() / 1000,
        })),

    setConnected: (connected) => set({ isConnected: connected }),
    setActiveTab: (tab) => set({ activeTab: tab }),
    setSelectedSymbol: (symbol) => set({ selectedSymbol: symbol }),
    toggleCommandPalette: () => set((s) => ({ commandPaletteOpen: !s.commandPaletteOpen })),
}));

// ─── Selectors (for memoized access) ───────────────────────────────────────

export const usePositions = () => useMarketStore((s) => s.positions);
export const usePrices = () => useMarketStore((s) => s.prices);
export const useSignals = () => useMarketStore((s) => s.radar);
export const useArbitrage = () => useMarketStore((s) => s.arbitrage);
export const useEquity = () => useMarketStore((s) => s.equity);
export const usePnl = () => useMarketStore((s) => s.pnl);
export const useRegime = () => useMarketStore((s) => s.regime);
export const useSystemStatus = () => useMarketStore((s) => s.systemStatus);
export const useActiveTab = () => useMarketStore((s) => s.activeTab);
export const useSelectedSymbol = () => useMarketStore((s) => s.selectedSymbol);
export const useLogs = () => useMarketStore((s) => s.logs);
export const useDoomsday = () => useMarketStore((s) => s.doomsday);
export const usePortfolioHealth = () => useMarketStore((s) => s.portfolioHealth);
