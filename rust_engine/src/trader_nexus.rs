/// Phase 4: TraderNexus Rust Core
/// Complete orchestration layer migrated from Python to Rust

use std::collections::HashMap;
use std::time::{Instant, Duration};
use pyo3::prelude::*;
use crate::agents::{governor, oracle, executor};
use crate::execution_algorithms::{TwapExecutor, VwapExecutor};
use crate::arbitrage::{ArbScanner, ExchangePrice};

/// Configuration for the trading loop
#[derive(Clone, Debug)]
#[pyclass]
pub struct NexusConfig {
    #[pyo3(get, set)]
    pub initial_capital: f64,
    #[pyo3(get, set)]
    pub fee_rate: f64,
    #[pyo3(get, set)]
    pub leverage: f64,
    #[pyo3(get, set)]
    pub stop_loss_pct: f64,
    #[pyo3(get, set)]
    pub take_profit_pct: f64,
    #[pyo3(get, set)]
    pub trailing_stop_active: f64,
    #[pyo3(get, set)]
    pub trailing_stop_dist: f64,
    #[pyo3(get, set)]
    pub max_positions: usize,
    #[pyo3(get, set)]
    pub risk_per_trade: f64,
    #[pyo3(get, set)]
    pub cycle_interval_ms: u64,
}

impl Default for NexusConfig {
    fn default() -> Self {
        NexusConfig {
            initial_capital: 100.0,
            fee_rate: 0.002,
            leverage: 5.0,
            stop_loss_pct: 0.02,
            take_profit_pct: 0.04,
            trailing_stop_active: 0.02,
            trailing_stop_dist: 0.01,
            max_positions: 8,
            risk_per_trade: 0.02,
            cycle_interval_ms: 60000, // 1 minute
        }
    }
}

/// Position state
#[derive(Clone, Debug)]
pub struct Position {
    pub symbol: String,
    pub side: String, // "BUY" or "SELL"
    pub quantity: f64,
    pub entry_price: f64,
    pub entry_time: u64,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub trailing_stop: Option<f64>,
    pub highest_price: f64,
    pub lowest_price: f64,
    pub strategy: String,
}

impl Position {
    pub fn new(
        symbol: &str,
        side: &str,
        quantity: f64,
        entry_price: f64,
        stop_loss: Option<f64>,
        take_profit: Option<f64>,
    ) -> Self {
        Position {
            symbol: symbol.to_string(),
            side: side.to_string(),
            quantity,
            entry_price,
            entry_time: current_timestamp_ms(),
            stop_loss,
            take_profit,
            trailing_stop: None,
            highest_price: entry_price,
            lowest_price: entry_price,
            strategy: "DIRECTIONAL".to_string(),
        }
    }

    pub fn unrealized_pnl(&self, current_price: f64) -> f64 {
        if self.side == "BUY" {
            (current_price - self.entry_price) * self.quantity
        } else {
            (self.entry_price - current_price) * self.quantity
        }
    }

    pub fn unrealized_pnl_pct(&self, current_price: f64) -> f64 {
        if self.side == "BUY" {
            (current_price - self.entry_price) / self.entry_price
        } else {
            (self.entry_price - current_price) / self.entry_price
        }
    }

    pub fn update_extremes(&mut self, price: f64) {
        if price > self.highest_price {
            self.highest_price = price;
        }
        if price < self.lowest_price {
            self.lowest_price = price;
        }

        // Update trailing stop
        if self.side == "BUY" && price > self.highest_price {
            if let Some(dist) = self.trailing_stop {
                self.stop_loss = Some(price - dist);
            }
        }
    }
}

/// Market data snapshot
#[derive(Clone, Debug)]
pub struct MarketSnapshot {
    pub symbol: String,
    pub timestamp: u64,
    pub bid: f64,
    pub ask: f64,
    pub last: f64,
    pub bid_size: f64,
    pub ask_size: f64,
    pub volume_24h: f64,
}

/// Trading signal
#[derive(Clone, Debug)]
pub struct TradingSignal {
    pub symbol: String,
    pub action: String, // "BUY", "SELL", "CLOSE", "HOLD"
    pub confidence: f64,
    pub size: f64,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub reason: String,
    pub strategy: String,
}

/// Performance metrics
#[derive(Clone, Debug, Default)]
pub struct PerformanceMetrics {
    pub total_trades: u64,
    pub winning_trades: u64,
    pub losing_trades: u64,
    pub total_pnl: f64,
    pub gross_profit: f64,
    pub gross_loss: f64,
    pub largest_win: f64,
    pub largest_loss: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub profit_factor: f64,
    pub win_rate: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub current_drawdown: f64,
}

/// Main TraderNexus state
#[pyclass]
pub struct TraderNexus {
    config: NexusConfig,
    governor: governor::GovernorState,
    executor: executor::ExecutorState,
    positions: HashMap<String, Position>,
    snapshots: HashMap<String, MarketSnapshot>,
    signals: Vec<TradingSignal>,
    metrics: PerformanceMetrics,
    twap_executors: HashMap<String, TwapExecutor>,
    vwap_executors: HashMap<String, VwapExecutor>,
    arb_scanner: ArbScanner,
    cycle_count: u64,
    last_cycle_time: Instant,
    is_running: bool,
}

impl TraderNexus {
    /// Create new TraderNexus
    pub fn new(config: NexusConfig) -> Self {
        let gov_config = governor::GovernorConfig::default();
        
        TraderNexus {
            config: config.clone(),
            governor: governor::GovernorState::new(gov_config),
            executor: executor::ExecutorState::new(config.initial_capital),
            positions: HashMap::new(),
            snapshots: HashMap::new(),
            signals: Vec::new(),
            metrics: PerformanceMetrics::default(),
            twap_executors: HashMap::new(),
            vwap_executors: HashMap::new(),
            arb_scanner: ArbScanner::new(0.1, 5000),
            cycle_count: 0,
            last_cycle_time: Instant::now(),
            is_running: false,
        }
    }

    /// Get current equity
    pub fn get_equity(&self) -> f64 {
        let mut equity = self.executor.balance_usd;
        
        for (symbol, position) in &self.positions {
            if let Some(snapshot) = self.snapshots.get(symbol) {
                equity += position.unrealized_pnl(snapshot.last);
            }
        }
        
        equity
    }

    /// Update market snapshot
    pub fn update_snapshot(&mut self, snapshot: MarketSnapshot) {
        // Update for arb scanner
        self.arb_scanner.update_price(ExchangePrice {
            exchange: "primary".to_string(),
            symbol: snapshot.symbol.clone(),
            bid: snapshot.bid,
            ask: snapshot.ask,
            bid_size: snapshot.bid_size,
            ask_size: snapshot.ask_size,
            timestamp: snapshot.timestamp,
        });

        // Update position extremes
        if let Some(position) = self.positions.get_mut(&snapshot.symbol) {
            position.update_extremes(snapshot.last);
        }

        self.snapshots.insert(snapshot.symbol.clone(), snapshot);
    }

    /// Generate trading signal from oracle
    pub fn generate_signal(
        &self,
        symbol: &str,
        closes: &[f64],
        rsi: &[f64],
        bb_lower: &[f64],
        bb_upper: &[f64],
        obv_slope: f64,
        entropy_score: f64,
        metabolism_state: &str,
    ) -> TradingSignal {
        let signal = oracle::OracleEngine::analyze_for_entry(
            closes, rsi, bb_lower, bb_upper, obv_slope, metabolism_state,
        );

        let action = match signal.direction {
            oracle::SignalDirection::Buy => "BUY",
            oracle::SignalDirection::Sell => "SELL",
            oracle::SignalDirection::Hold => "HOLD",
        };

        // Calculate position size
        let current_price = self.snapshots.get(symbol)
            .map(|s| s.last)
            .unwrap_or(*closes.last().unwrap_or(&0.0));

        let mut size = self.governor.calculate_position_size(
            current_price,
            0.0, // ATR
            signal.confidence,
        );

        // Apply entropy-based adjustment
        if entropy_score > 2.0 {
            size *= 0.5; // Reduce in chaos
        }

        TradingSignal {
            symbol: symbol.to_string(),
            action: action.to_string(),
            confidence: signal.confidence,
            size,
            stop_loss: None,
            take_profit: None,
            reason: signal.reason,
            strategy: "ORACLE".to_string(),
        }
    }

    /// Execute signal
    pub fn execute_signal(&mut self, signal: &TradingSignal) -> bool {
        if signal.action == "HOLD" || signal.size <= 0.0 {
            return false;
        }

        // Check position limits
        if self.positions.len() >= self.config.max_positions {
            return false;
        }

        // Check if already have position
        if self.positions.contains_key(&signal.symbol) {
            return false;
        }

        // Get current price
        let current_price = self.snapshots.get(&signal.symbol)
            .map(|s| s.last)
            .unwrap_or(0.0);

        if current_price <= 0.0 {
            return false;
        }

        // Calculate stop loss and take profit
        let stop_loss = if signal.action == "BUY" {
            Some(current_price * (1.0 - self.config.stop_loss_pct))
        } else {
            Some(current_price * (1.0 + self.config.stop_loss_pct))
        };

        let take_profit = if signal.action == "BUY" {
            Some(current_price * (1.0 + self.config.take_profit_pct))
        } else {
            Some(current_price * (1.0 - self.config.take_profit_pct))
        };

        // Create position
        let quantity = signal.size / current_price;
        let position = Position::new(
            &signal.symbol,
            &signal.action,
            quantity,
            current_price,
            stop_loss,
            take_profit,
        );

        // Update executor state
        self.executor.held_assets.insert(signal.symbol.clone(), quantity);
        self.executor.entry_prices.insert(signal.symbol.clone(), current_price);

        // Deduct margin
        let margin = signal.size / self.config.leverage;
        self.executor.balance_usd -= margin;

        self.positions.insert(signal.symbol.clone(), position);

        self.metrics.total_trades += 1;

        true
    }

    /// Check exits for all positions
    pub fn check_exits(&mut self) -> Vec<(String, f64, String)> {
        let mut exits = Vec::new();

        for (symbol, position) in &mut self.positions {
            if let Some(snapshot) = self.snapshots.get(symbol) {
                let current_price = snapshot.last;

                // Check stop loss
                if let Some(sl) = position.stop_loss {
                    if (position.side == "BUY" && current_price <= sl) ||
                       (position.side == "SELL" && current_price >= sl) {
                        exits.push((symbol.clone(), current_price, "STOP_LOSS".to_string()));
                        continue;
                    }
                }

                // Check take profit
                if let Some(tp) = position.take_profit {
                    if (position.side == "BUY" && current_price >= tp) ||
                       (position.side == "SELL" && current_price <= tp) {
                        exits.push((symbol.clone(), current_price, "TAKE_PROFIT".to_string()));
                        continue;
                    }
                }

                // Check trailing stop
                if let Some(trail) = position.trailing_stop {
                    if (position.side == "BUY" && current_price <= trail) ||
                       (position.side == "SELL" && current_price >= trail) {
                        exits.push((symbol.clone(), current_price, "TRAILING_STOP".to_string()));
                        continue;
                    }
                }
            }
        }

        // Execute exits (clone to avoid move)
        for (symbol, exit_price, reason) in exits.clone() {
            self.close_position(&symbol, exit_price, &reason);
        }

        exits
    }

    /// Close a position
    pub fn close_position(&mut self, symbol: &str, exit_price: f64, _reason: &str) -> f64 {
        if let Some(position) = self.positions.remove(symbol) {
            let pnl = position.unrealized_pnl(exit_price);
            
            // Update metrics
            self.metrics.total_pnl += pnl;
            
            if pnl > 0.0 {
                self.metrics.winning_trades += 1;
                self.metrics.gross_profit += pnl;
                if pnl > self.metrics.largest_win {
                    self.metrics.largest_win = pnl;
                }
            } else {
                self.metrics.losing_trades += 1;
                self.metrics.gross_loss += pnl.abs();
                if pnl.abs() > self.metrics.largest_loss {
                    self.metrics.largest_loss = pnl.abs();
                }
            }

            // Return margin to balance
            let margin = (position.quantity * position.entry_price) / self.config.leverage;
            self.executor.balance_usd += margin + pnl;

            // Remove from executor state
            self.executor.held_assets.remove(symbol);
            self.executor.entry_prices.remove(symbol);

            return pnl;
        }

        0.0
    }

    /// Scan for arbitrage opportunities
    pub fn scan_arbitrage(&self) -> Vec<crate::arbitrage::SpatialArbOpportunity> {
        self.arb_scanner.scan_spatial_arb()
    }

    /// Start TWAP execution
    pub fn start_twap(
        &mut self,
        symbol: &str,
        side: &str,
        total_qty: f64,
        duration_minutes: u64,
        num_slices: usize,
    ) {
        let twap = TwapExecutor::new(symbol, side, total_qty, duration_minutes, num_slices);
        self.twap_executors.insert(symbol.to_string(), twap);
    }

    /// Start VWAP execution
    pub fn start_vwap(
        &mut self,
        symbol: &str,
        side: &str,
        total_qty: f64,
        volume_profile: Vec<f64>,
    ) {
        let vwap = VwapExecutor::new(symbol, side, total_qty, volume_profile);
        self.vwap_executors.insert(symbol.to_string(), vwap);
    }

    /// Run single trading cycle
    pub fn run_cycle(
        &mut self,
        market_data: &HashMap<String, Vec<f64>>,
        indicators: &HashMap<String, Vec<f64>>,
    ) -> Vec<TradingSignal> {
        self.cycle_count += 1;
        self.last_cycle_time = Instant::now();

        let mut signals = Vec::new();

        // Process each symbol
        for (symbol, closes) in market_data {
            if closes.is_empty() {
                continue;
            }

            // Get indicators for this symbol
            let rsi = indicators.get(&format!("{}_rsi", symbol))
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            
            let bb_lower = indicators.get(&format!("{}_bb_lower", symbol))
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            
            let bb_upper = indicators.get(&format!("{}_bb_upper", symbol))
                .map(|v| v.as_slice())
                .unwrap_or(&[]);

            let obv_slope = indicators.get(&format!("{}_obv", symbol))
                .and_then(|v| v.last())
                .copied()
                .unwrap_or(0.0);

            let entropy_score = indicators.get(&format!("{}_entropy", symbol))
                .and_then(|v| v.last())
                .copied()
                .unwrap_or(1.0);

            // Generate signal
            let signal = self.generate_signal(
                symbol,
                closes,
                rsi,
                bb_lower,
                bb_upper,
                obv_slope,
                entropy_score,
                "NORMAL",
            );

            // Execute signal
            if self.execute_signal(&signal) {
                signals.push(signal);
            }
        }

        // Check exits
        self.check_exits();

        // Update metrics
        self.update_metrics();

        signals
    }

    /// Update performance metrics
    pub fn update_metrics(&mut self) {
        let equity = self.get_equity();
        
        // Calculate drawdown
        let peak_equity = self.config.initial_capital.max(
            self.metrics.total_pnl + self.config.initial_capital
        );
        
        self.metrics.current_drawdown = (peak_equity - equity) / peak_equity;
        
        if self.metrics.current_drawdown > self.metrics.max_drawdown {
            self.metrics.max_drawdown = self.metrics.current_drawdown;
        }

        // Win rate
        if self.metrics.total_trades > 0 {
            self.metrics.win_rate = self.metrics.winning_trades as f64 / self.metrics.total_trades as f64;
        }

        // Profit factor
        if self.metrics.gross_loss > 0.0 {
            self.metrics.profit_factor = self.metrics.gross_profit / self.metrics.gross_loss;
        }

        // Average win/loss
        if self.metrics.winning_trades > 0 {
            self.metrics.avg_win = self.metrics.gross_profit / self.metrics.winning_trades as f64;
        }
        if self.metrics.losing_trades > 0 {
            self.metrics.avg_loss = self.metrics.gross_loss / self.metrics.losing_trades as f64;
        }
    }

    /// Get status
    pub fn get_status(&self) -> HashMap<String, String> {
        let mut status = HashMap::new();
        
        status.insert("cycle_count".to_string(), self.cycle_count.to_string());
        status.insert("equity".to_string(), format!("{:.2}", self.get_equity()));
        status.insert("balance".to_string(), format!("{:.2}", self.executor.balance_usd));
        status.insert("positions".to_string(), self.positions.len().to_string());
        status.insert("total_trades".to_string(), self.metrics.total_trades.to_string());
        status.insert("win_rate".to_string(), format!("{:.2}%", self.metrics.win_rate * 100.0));
        status.insert("total_pnl".to_string(), format!("{:.2}", self.metrics.total_pnl));
        status.insert("max_drawdown".to_string(), format!("{:.2}%", self.metrics.max_drawdown * 100.0));
        
        status
    }

    /// Start the trading loop
    pub fn start(&mut self) {
        self.is_running = true;
        self.last_cycle_time = Instant::now();
    }

    /// Stop the trading loop
    pub fn stop(&mut self) {
        self.is_running = false;
    }

    /// Check if should run next cycle
    pub fn should_run_cycle(&self) -> bool {
        if !self.is_running {
            return false;
        }

        self.last_cycle_time.elapsed() >= Duration::from_millis(self.config.cycle_interval_ms)
    }
}

// === PHASE 4: PYTHON BINDINGS ===

#[pymethods]
impl NexusConfig {
    #[new]
    #[pyo3(signature = (
        initial_capital=100.0,
        fee_rate=0.002,
        leverage=5.0,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        trailing_stop_active=0.02,
        trailing_stop_dist=0.01,
        max_positions=8,
        risk_per_trade=0.02,
        cycle_interval_ms=60000,
    ))]
    fn py_new(
        initial_capital: f64,
        fee_rate: f64,
        leverage: f64,
        stop_loss_pct: f64,
        take_profit_pct: f64,
        trailing_stop_active: f64,
        trailing_stop_dist: f64,
        max_positions: usize,
        risk_per_trade: f64,
        cycle_interval_ms: u64,
    ) -> Self {
        NexusConfig {
            initial_capital,
            fee_rate,
            leverage,
            stop_loss_pct,
            take_profit_pct,
            trailing_stop_active,
            trailing_stop_dist,
            max_positions,
            risk_per_trade,
            cycle_interval_ms,
        }
    }
}

#[pymethods]
impl TraderNexus {
    #[new]
    fn py_new(config: NexusConfig) -> Self {
        TraderNexus::new(config)
    }

    /// Get cycle count
    fn cycle_count(&self) -> u64 {
        self.cycle_count
    }

    /// Get is_running status
    fn is_running(&self) -> bool {
        self.is_running
    }

    /// Get number of positions
    fn num_positions(&self) -> usize {
        self.positions.len()
    }

    /// Get current equity
    fn get_equity_py(&self) -> f64 {
        self.get_equity()
    }

    /// Get status summary
    fn status(&self) -> std::collections::HashMap<String, String> {
        self.get_status()
    }

    /// Start the trading loop
    fn start_py(&mut self) {
        self.start()
    }

    /// Stop the trading loop
    fn stop_py(&mut self) {
        self.stop()
    }

    /// Run single trading cycle (internal Rust use only)
    /// Python should use the high-level run_cycle_with_data method
    #[pyo3(name = "run_cycle")]
    fn run_cycle_py(
        &mut self,
        _market_data: std::collections::HashMap<String, Vec<f64>>,
        _indicators: std::collections::HashMap<String, Vec<f64>>,
    ) -> usize {
        // Simplified: just increment cycle count for now
        // Full implementation would require TradingSignal to be #[pyclass]
        self.cycle_count += 1;
        self.cycle_count as usize
    }

    /// Start TWAP execution
    #[pyo3(name = "start_twap")]
    fn start_twap_py(
        &mut self,
        symbol: &str,
        side: &str,
        total_qty: f64,
        duration_minutes: u64,
        num_slices: usize,
    ) {
        self.start_twap(symbol, side, total_qty, duration_minutes, num_slices)
    }

    /// Start VWAP execution
    #[pyo3(name = "start_vwap")]
    fn start_vwap_py(
        &mut self,
        symbol: &str,
        side: &str,
        total_qty: f64,
        volume_profile: Vec<f64>,
    ) {
        self.start_vwap(symbol, side, total_qty, volume_profile)
    }

    /// Scan for arbitrage opportunities (returns count for now)
    #[pyo3(name = "scan_arbitrage")]
    fn scan_arbitrage_py(&self) -> usize {
        // Simplified: return count only
        // Full implementation would require SpatialArbOpportunity to be #[pyclass]
        let opportunities = self.scan_arbitrage();
        opportunities.len()
    }

    /// Get performance metrics
    fn get_metrics(&self) -> std::collections::HashMap<String, f64> {
        let mut m = std::collections::HashMap::new();
        m.insert("total_trades".to_string(), self.metrics.total_trades as f64);
        m.insert("winning_trades".to_string(), self.metrics.winning_trades as f64);
        m.insert("losing_trades".to_string(), self.metrics.losing_trades as f64);
        m.insert("total_pnl".to_string(), self.metrics.total_pnl);
        m.insert("win_rate".to_string(), self.metrics.win_rate);
        m.insert("max_drawdown".to_string(), self.metrics.max_drawdown);
        m.insert("current_drawdown".to_string(), self.metrics.current_drawdown);
        m
    }
}

/// Get current timestamp in milliseconds
pub fn current_timestamp_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nexus_creation() {
        let config = NexusConfig::default();
        let nexus = TraderNexus::new(config);
        
        assert_eq!(nexus.get_equity(), 100.0);
        assert_eq!(nexus.positions.len(), 0);
        assert!(!nexus.is_running);
    }

    #[test]
    fn test_position_pnl() {
        let mut position = Position::new("BTC/USDT", "BUY", 0.1, 50000.0, None, None);
        
        // Price goes up
        assert!((position.unrealized_pnl(55000.0) - 500.0).abs() < 0.01);
        assert!((position.unrealized_pnl_pct(55000.0) - 0.1).abs() < 0.01);
        
        // Price goes down
        assert!((position.unrealized_pnl(45000.0) + 500.0).abs() < 0.01);
    }

    #[test]
    fn test_metrics_update() {
        let config = NexusConfig::default();
        let mut nexus = TraderNexus::new(config);
        
        // Simulate some trades
        nexus.metrics.total_trades = 10;
        nexus.metrics.winning_trades = 6;
        nexus.metrics.losing_trades = 4;
        nexus.metrics.gross_profit = 1000.0;
        nexus.metrics.gross_loss = 600.0;
        
        nexus.update_metrics();
        
        assert!((nexus.metrics.win_rate - 0.6).abs() < 0.01);
        assert!((nexus.metrics.profit_factor - 1.667).abs() < 0.01);
    }
}
