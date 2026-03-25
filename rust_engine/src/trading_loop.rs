use crate::agents::{governor, oracle, executor};

/// Market Tick Data
#[derive(Clone, Debug)]
pub struct Tick {
    pub symbol: String,
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Trading Loop State
pub struct TradingLoop {
    pub governor: governor::GovernorState,
    pub executor: executor::ExecutorState,
    // Oracle is stateless
}

impl TradingLoop {
    pub fn new(initial_capital: f64) -> Self {
        let gov_config = governor::GovernorConfig::default();
        TradingLoop {
            governor: governor::GovernorState::new(gov_config),
            executor: executor::ExecutorState::new(initial_capital),
        }
    }

    /// Process a single tick
    /// Returns: (action, size, reason)
    pub fn process_tick(
        &mut self,
        tick: &Tick,
        closes: &[f64],
        rsi: &[f64],
        bb_lower: &[f64],
        bb_upper: &[f64],
        obv_slope: f64,
        entropy_score: f64,
        metabolism_state: &str,
    ) -> (String, f64, String) {
        
        // 1. Oracle: Generate Signal
        let signal = oracle::OracleEngine::analyze_for_entry(
            closes, rsi, bb_lower, bb_upper, obv_slope, metabolism_state
        );

        if signal.direction == oracle::SignalDirection::Hold {
            return ("HOLD".to_string(), 0.0, signal.reason);
        }

        // 2. Governor: Risk Check
        let position_size = self.governor.calculate_position_size(
            tick.close,
            0.0, // Current ATR (simplified)
            signal.confidence
        );

        if position_size <= 0.0 {
            return ("REJECT".to_string(), 0.0, "Governor vetoed: Insufficient size".to_string());
        }

        // Cluster Risk Check
        let held: Vec<String> = self.executor.held_assets.keys().cloned().collect();
        let mut temp_state = self.governor.clone();
        for s in &held {
            temp_state.positions.insert(s.clone(), governor::Position {
                symbol: s.clone(),
                direction: 1,
                entry_price: 0.0,
                quantity: 0.0,
                stop_loss: None,
            });
        }
        if !temp_state.check_cluster_risk(&tick.symbol) {
            return ("REJECT".to_string(), 0.0, "Governor vetoed: Cluster risk".to_string());
        }

        // 3. Executor: Decide Action
        let regime = if entropy_score < 1.0 {
            executor::Regime::Ordered
        } else if entropy_score > 2.0 {
            executor::Regime::Chaotic
        } else {
            executor::Regime::Transition
        };

        let trade_signal = executor::TradeSignal {
            symbol: tick.symbol.clone(),
            direction: executor::Direction::Buy,
            size: position_size,
            price: tick.close,
            conviction: signal.confidence,
            stop_loss_price: None,
        };

        let decision = self.executor.decide_trade(&trade_signal, entropy_score, &regime);

        let action_str = match decision.action {
            executor::TradeAction::Execute => "EXECUTE",
            executor::TradeAction::Halt => "HALT",
            executor::TradeAction::Reduce => "REDUCE",
        };

        (action_str.to_string(), decision.adjusted_size, signal.reason)
    }

    /// Run a batch backtest (Vectorized)
    /// Returns: Final balance
    pub fn run_backtest(
        &mut self,
        timestamps: &[i64],
        opens: &[f64],
        highs: &[f64],
        lows: &[f64],
        closes: &[f64],
        rsi: &[f64],
        bb_lower: &[f64],
        bb_upper: &[f64],
        obv_slopes: &[f64],
        entropy_scores: &[f64],
        metabolism_state: &str,
    ) -> f64 {
        let len = timestamps.len();
        
        for i in 20..len {  // Need 20 candles for indicators
            let tick = Tick {
                symbol: "TEST".to_string(),
                timestamp: timestamps[i],
                open: opens[i],
                high: highs[i],
                low: lows[i],
                close: closes[i],
                volume: 0.0,
            };

            let (action, size, _reason) = self.process_tick(
                &tick,
                &closes[..=i],
                &rsi[..=i],
                &bb_lower[..=i],
                &bb_upper[..=i],
                obv_slopes[i],
                entropy_scores[i],
                metabolism_state,
            );

            // Simple execution simulation
            if action == "EXECUTE" && size > 0.0 {
                // Buy
                let qty = size / tick.close;
                self.executor.balance_usd -= size;
                *self.executor.held_assets.entry(tick.symbol.clone()).or_insert(0.0) += qty;
                self.executor.entry_prices.insert(tick.symbol.clone(), tick.close);
            }
        }

        // Final equity
        let mut equity = self.executor.balance_usd;
        for (_symbol, qty) in &self.executor.held_assets {
            if let Some(&price) = Some(&closes[len-1]) {
                equity += qty * price;
            }
        }

        equity
    }
}
