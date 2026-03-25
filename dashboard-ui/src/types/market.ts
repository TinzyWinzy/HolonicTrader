// ─── Domain Types ──────────────────────────────────────────────────────────────
// Aligned with signal_server.py hub_state payload shape

export interface Position {
    symbol: string;
    direction: 'BUY' | 'SELL';
    quantity: number;
    entry_price: number;
    current_price: number;
    leverage: number;
    pnl: number;
    pnl_pct: number;
    strategy: string;
}

export interface Signal {
    symbol: string;
    direction: 'BUY' | 'SELL' | 'NEUTRAL';
    conviction: number;
    score?: number;
    quality: 'HIGH' | 'MEDIUM' | 'LOW';
    reason: string;
    price: number;
    tp?: number;
    sl?: number;
    timestamp: string;
    regime?: string;
    tda_status?: string;
    metadata?: Record<string, unknown>;
    pips_potential?: number;
    expected_yield?: number;
    hit_probability?: number;
    decay_score?: number;
    optimal_horizon?: string;
    execution_details?: Record<string, unknown>;
}

export interface ArbitrageOpportunity {
    symbol: string;
    funding_apy: number;
    spread_pct: number;
    signal: string | null;
    reason: string;
    confidence: number;
    has_opportunity: boolean;
}

export interface PortfolioHealth {
    drawdown_pct: number;
    margin_utilization: number;
    risk_budget: number;
    equity: number;
}

export interface DoomsdayStatus {
    defcon_level: number;
    crisis_active: boolean;
}

export interface EquityPoint {
    t: string;
    y: number;
}

export interface LogEntry {
    time: string;
    msg: string;
    level: string;
}

export interface EvolutionData {
    generation?: number;
    best_fitness?: number;
    population_size?: number;
    [key: string]: unknown;
}

export interface OrderFlowData {
    [key: string]: unknown;
}

export interface MonteCarloData {
    [key: string]: unknown;
}

export interface HubState {
    status: string;
    system_status: string;
    health_score: number;
    equity: number;
    pnl: number;
    regime: string;

    scanning: boolean;
    last_scan: string | null;

    equity_history: EquityPoint[];
    radar: Signal[];
    positions: Position[];
    prices: Record<string, number>;
    portfolio_health: PortfolioHealth;
    doomsday: DoomsdayStatus;
    arbitrage: ArbitrageOpportunity[];

    evolution: EvolutionData;
    order_flow: OrderFlowData;
    monte_carlo: MonteCarloData;
    logs: LogEntry[];
    timestamp: number;
}

export type TabId = 'market' | 'positions' | 'signals' | 'arbitrage' | 'risk' | 'logs';
