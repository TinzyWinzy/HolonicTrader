use pyo3::prelude::*;
use std::collections::HashMap;

mod math;
mod agents;
mod trading_loop;
mod parallel_wfo;
#[cfg(feature = "onnx")]
mod onnx_inference;  // RESTORED: Phase 2 Intelligence & Adaptation
mod core;
mod execution_algorithms;  // PHASE 3: TWAP/VWAP Execution
mod arbitrage;  // PHASE 3: Spatial & Triangular Arbitrage
mod trader_nexus;  // PHASE 4: Rust Core Orchestration

// --- DOMAIN MODELS ---

#[derive(Clone, Debug)]
struct Candle {
    timestamp: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
}

#[derive(Clone, Debug)]
struct Position {
    entry_price: f64,
    entry_time: i64,
    quantity: f64,
    direction: i32, // 1 for Long, -1 for Short
}

#[derive(Clone, Debug)]
#[pyclass] // Export to Python for analysis
struct Trade {
    #[pyo3(get)]
    entry_time: i64,
    #[pyo3(get)]
    exit_time: i64,
    #[pyo3(get)]
    entry_price: f64,
    #[pyo3(get)]
    exit_price: f64,
    #[pyo3(get)]
    pnl: f64,
    #[pyo3(get)]
    roi: f64,
    #[pyo3(get)]
    direction: i32,
}

// --- ENGINE LOGIC ---

struct Backtester {
    balance: f64,
    position: Option<Position>,
    trades: Vec<Trade>,
    fee_rate: f64,
    
    // Risk Params
    leverage: f64,
    stop_loss_pct: f64,
    take_profit_pct: f64,
    trail_active_pct: f64,
    trail_dist_pct: f64,
    
    // State
    trailing_stop: Option<f64>,
    highest_price: f64, // For Trailing
}

impl Backtester {
    fn new(initial_capital: f64, fee_rate: f64, leverage: f64, sl: f64, tp: f64, trail_act: f64, trail_dist: f64) -> Self {
        Backtester {
            balance: initial_capital,
            position: None,
            trades: Vec::new(),
            fee_rate,
            leverage,
            stop_loss_pct: sl,
            take_profit_pct: tp,
            trail_active_pct: trail_act,
            trail_dist_pct: trail_dist,
            trailing_stop: None,
            highest_price: 0.0,
        }
    }

    fn process_candle(&mut self, candle: &Candle, signal: i32) {
        // ---------------------------------------------------
        // 1. LIQUIDATION CHECK (The Reaper)
        // ---------------------------------------------------
        if let Some(pos) = &self.position {
            let current_val = pos.quantity * candle.open; // Check at Open first (Gap risk)
            let _margin_used = (pos.quantity * pos.entry_price) / self.leverage;
            let pnl = (current_val - (pos.quantity * pos.entry_price)) * pos.direction as f64;
            let _equity = self.balance + pnl; // Balance is cash + unrealized pnl ?? No, balance is cash-margin. 
            // Simplified Model: Balance tracks CASH. Margin is checking constraint.
            // Correct Model: 
            // Equity = Balance + Unrealized PnL.
            // Balance was decremented by cost? In my open_position logic: yes.
            // So Equity = Balance + MarketValue. 
            // Wait, previous `open_position` did `balance -= cost`. 
            // So Balance = Remaining Cash. 
            // Equity = Remaining Cash + (Quantity * CurrentPrice).
            
            // Maintenance Margin = Notional * 0.5% (Binance/Kraken standard roughly for low lev, but let's say 50% of Initial Margin)
            // Initial Margin = Notional / Leverage.
            // MM = IM * 0.5.
            
            let mark_price = candle.low; // Worst case for Long
            let _notional = pos.quantity * mark_price;
            let _unrealized_pnl = (mark_price - pos.entry_price) * pos.quantity * pos.direction as f64;
            
            // Total Account Equity (Cash + MarginLocked + PnL)
            // My struct stores 'balance' as Free Cash? 
            // In open_position: balance -= (cost + fee). So Balance is Free Cash.
            // So Equity = Balance + (Cost + UnrealizedPnL).
            // Cost is actually Margin Locked in a futures model? 
            // Previous code: `self.balance -= (cost + fee)`. Cost was `quantity * price`. That's Spot logic.
            // For LEVERAGE: Cost should be `(quantity * price) / leverage`.
            // I need to fix open_position to use leverage first.
            
            // Let's assume open_position uses leverage correctly now.
        }

        // ---------------------------------------------------
        // 2. STOP LOSS & TRAILING STOP
        // ---------------------------------------------------
        if let Some(pos) = &mut self.position {
            let mut _exit_price = 0.0;
            let mut close_type = 0; // 0=None, 1=SL, 2=TP, 3=Trail, 4=Liq

            // A. Update High Water Mark for Longs
            if candle.high > self.highest_price {
                self.highest_price = candle.high;
            }

            // B. Calculate Trailing Stop
            if self.trail_active_pct > 0.0 {
                let activation_price = pos.entry_price * (1.0 + self.trail_active_pct);
                if self.highest_price >= activation_price {
                    let new_trail = self.highest_price * (1.0 - self.trail_dist_pct);
                    if self.trailing_stop.is_none() || new_trail > self.trailing_stop.unwrap() {
                        self.trailing_stop = Some(new_trail);
                    }
                }
            }

            // C. Check Stops (Low of candle triggers Long stop)
            // 1. Hard Stop
            let sl_price = pos.entry_price * (1.0 - self.stop_loss_pct);
            if candle.low <= sl_price {
                _exit_price = sl_price; // Assume fill at SL (Slippage ignored for speed, or add penalty)
                close_type = 1;
            }
            // 2. Trailing Stop
            else if let Some(trail) = self.trailing_stop {
                if candle.low <= trail {
                    _exit_price = trail;
                    close_type = 3;
                }
            }
            // 3. Take Profit (High triggers TP)
            else if self.take_profit_pct > 0.0 {
                let tp_price = pos.entry_price * (1.0 + self.take_profit_pct);
                if candle.high >= tp_price {
                    _exit_price = tp_price;
                    close_type = 2;
                }
            }
            
            // D. Signal Exit
            if close_type == 0 && signal == -1 {
                _exit_price = candle.close;
                close_type = 5; // Signal
            }

            if close_type > 0 {
                // Apply Slippage based on Volatility/Stop type?
                // For simplified engine: 0.1% slip on stops
                if close_type == 1 || close_type == 3 {
                    _exit_price *= 0.999;
                }
                
                // Need to call close_position, but we are inside a mutable borrow of position.
                // We must break out or clone needed data.
                // Rust ownership trick: set flag, handle outside.
            }
        }
        
        // REFACTORING TO AVOID BORROW HELL:
        // Move check logic to a helper or just do the calculation and return "Action"
    }
    
    // ... (Refined logic will be in the actual replacement block) ...
}

// REDEFINING THE WHOLE LOGIC BLOCK IN ONE GO FOR CORRECTNESS

#[pyfunction]
fn run_backtest_fast(
    timestamps: Vec<i64>,
    opens: Vec<f64>,
    highs: Vec<f64>,
    lows: Vec<f64>,
    closes: Vec<f64>,
    signals: Vec<i32>,
    initial_capital: f64,
    fee_rate: f64,
    leverage: f64,
    stop_loss: f64,
    take_profit: f64,
    trail_active: f64,
    trail_dist: f64,
) -> PyResult<(f64, Vec<Trade>)> {
    
    let len = timestamps.len();
    if opens.len() != len {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Len mismatch"));
    }

    let mut balance = initial_capital;
    let mut position: Option<Position> = None;
    let mut trades: Vec<Trade> = Vec::new();
    
    let mut trailing_stop: Option<f64> = None;
    let mut highest_price: f64 = 0.0;
    let mut pending_signal: i32 = 0;

    for i in 0..len {
        let ts = timestamps[i];
        let o = opens[i];
        let h = highs[i];
        let l = lows[i];
        let c = closes[i];
        
        // --- 1. EXECUTE PENDING SIGNAL (Next-Open Rule) ---
        if pending_signal != 0 && position.is_none() {
             if pending_signal == 1 {
                 // Open Long on o
                 let price = o;
                 let cost = balance / (1.0 + (leverage * fee_rate));
                 let notional = cost * leverage;
                 let fee = notional * fee_rate;
                 let quantity = notional / price;
                 
                 if quantity > 0.0 {
                     balance -= cost + fee;
                     position = Some(Position {
                         entry_price: price,
                         entry_time: ts,
                         quantity,
                         direction: 1,
                     });
                     highest_price = price;
                     trailing_stop = None;
                 }
             }
             pending_signal = 0;
        }

        // --- 2. MANAGE POSITION (Intra-Candle) ---
        if let Some(pos) = &position {
            // A. Liquidation Check (Against Low)
            let margin_locked = (pos.entry_price * pos.quantity) / leverage;
            let current_val_low = l * pos.quantity;
            let entry_val = pos.entry_price * pos.quantity;
            let pnl_low = current_val_low - entry_val; 
            let total_equity_low = balance + margin_locked + pnl_low;
            
            if total_equity_low <= margin_locked * 0.5 {
                trades.push(Trade {
                    entry_time: pos.entry_time,
                    exit_time: ts,
                    entry_price: pos.entry_price,
                    exit_price: l,
                    pnl: -margin_locked,
                    roi: -100.0,
                    direction: pos.direction,
                });
                balance = 0.0; // Account zeroed (simplified)
                position = None;
                continue;
            }

            // B. Trailing Update (Against High)
            if h > highest_price { highest_price = h; }
            if trail_active > 0.0 {
                let act_price = pos.entry_price * (1.0 + trail_active);
                if highest_price >= act_price {
                    let new_trail = highest_price * (1.0 - trail_dist);
                    if trailing_stop.is_none() || new_trail > trailing_stop.unwrap() {
                        trailing_stop = Some(new_trail);
                    }
                }
            }

            // C. Forced Exits (SL, Trail, TP)
            let mut exit_price = 0.0;
            let mut close_type = 0; // 1:SL, 2:Trail, 3:TP, 4:Signal

            let sl_price = pos.entry_price * (1.0 - stop_loss);
            let tp_price = if take_profit > 0.0 { pos.entry_price * (1.0 + take_profit) } else { 0.0 };

            if l <= sl_price {
                exit_price = sl_price;
                close_type = 1;
            } else if let Some(trail) = trailing_stop {
                if l <= trail {
                    exit_price = trail;
                    close_type = 2;
                }
            } else if tp_price > 0.0 && h >= tp_price {
                exit_price = tp_price;
                close_type = 3;
            } else if signals[i] == -1 {
                exit_price = c;
                close_type = 4;
            }

            if close_type > 0 {
                // Apply Slippage (0.1% on stops)
                if close_type == 1 || close_type == 2 {
                    exit_price *= 0.999;
                }
                
                let qty = pos.quantity;
                let gross_val = qty * exit_price;
                let fee = gross_val * fee_rate;
                let net_pnl = (exit_price - pos.entry_price) * qty - fee;
                
                balance += margin_locked + net_pnl;
                trades.push(Trade {
                    entry_time: pos.entry_time,
                    exit_time: ts,
                    entry_price: pos.entry_price,
                    exit_price,
                    pnl: net_pnl,
                    roi: (net_pnl / margin_locked) * 100.0,
                    direction: pos.direction,
                });
                position = None;
                continue;
            }
        }

        // --- 3. QUEUE NEXT SIGNAL ---
        if signals[i] == 1 && position.is_none() {
            pending_signal = 1;
        }
    }

    Ok((balance, trades))
}


#[pymodule]
fn holonic_speed(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Trade>()?;
    
    // === PHASE 4: TRADER NEXUS CLASSES ===
    m.add_class::<trader_nexus::NexusConfig>()?;
    m.add_class::<trader_nexus::TraderNexus>()?;
    
    m.add_function(wrap_pyfunction!(run_backtest_fast, m)?)?;
    
    // Entropy Functions
    #[pyfunction]
    fn calculate_shannon_entropy(data: Vec<f64>) -> f64 {
        math::entropy::calculate_shannon_entropy(&data)
    }
    
    #[pyfunction]
    fn calculate_renyi_entropy(data: Vec<f64>, alpha: f64) -> f64 {
        math::entropy::calculate_renyi_entropy(&data, alpha)
    }

    m.add_function(wrap_pyfunction!(calculate_shannon_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_renyi_entropy, m)?)?;

    // AEHML 2.0 Functions
    #[pyfunction]
    fn calculate_permutation_entropy(data: Vec<f64>, m: usize, delay: usize) -> f64 {
        math::entropy::calculate_permutation_entropy(&data, m, delay)
    }

    #[pyfunction]
    fn calculate_multiscale_entropy(data: Vec<f64>, max_scale: usize, m: usize) -> Vec<f64> {
        math::entropy::calculate_multiscale_entropy(&data, max_scale, m)
    }

    #[pyfunction]
    fn calculate_persistent_entropy(data: Vec<f64>, window_size: usize, dimension: usize, delay: usize) -> f64 {
        math::topology::calculate_persistent_entropy(&data, window_size, dimension, delay)
    }

    m.add_function(wrap_pyfunction!(calculate_permutation_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_multiscale_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_persistent_entropy, m)?)?;

    // Indicator Functions
    #[pyfunction]
    fn calculate_rsi(data: Vec<f64>, period: usize) -> Vec<f64> {
        math::indicators::calculate_rsi(&data, period)
    }

    #[pyfunction]
    fn calculate_bollinger_bands(data: Vec<f64>, period: usize, std_dev: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        math::indicators::calculate_bollinger_bands(&data, period, std_dev)
    }

    #[pyfunction]
    fn calculate_atr(high: Vec<f64>, low: Vec<f64>, close: Vec<f64>, period: usize) -> Vec<f64> {
        math::indicators::calculate_atr(&high, &low, &close, period)
    }

    m.add_function(wrap_pyfunction!(calculate_rsi, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_bollinger_bands, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_atr, m)?)?;

    // Governor Functions
    #[pyfunction]
    fn governor_calculate_position_size(
        balance: f64,
        equity: f64,
        high_water_mark: f64,
        reference_atr: f64,
        asset_price: f64,
        current_atr: f64,
        conviction: f64,
    ) -> f64 {
        let config = agents::governor::GovernorConfig::default();
        let mut state = agents::governor::GovernorState::new(config);
        state.balance = balance;
        state.equity = equity;
        state.high_water_mark = high_water_mark;
        state.reference_atr = reference_atr;
        
        state.calculate_position_size(asset_price, current_atr, conviction)
    }

    #[pyfunction]
    fn governor_check_cluster_risk(
        held_symbols: Vec<String>,
        new_symbol: String,
    ) -> bool {
        let config = agents::governor::GovernorConfig::default();
        let mut state = agents::governor::GovernorState::new(config);
        
        // Populate positions (simplified: just need symbols, not full Position)
        for sym in held_symbols {
            state.positions.insert(sym.clone(), agents::governor::Position {
                symbol: sym,
                direction: 1,
                entry_price: 0.0,
                quantity: 0.0,
                stop_loss: None,
            });
        }
        
        state.check_cluster_risk(&new_symbol)
    }

    #[pyfunction]
    fn governor_calculate_max_risk(balance: f64, principal: f64, equity: f64) -> f64 {
        let mut config = agents::governor::GovernorConfig::default();
        config.principal = principal;
        let mut state = agents::governor::GovernorState::new(config);
        state.balance = balance;
        state.equity = equity;
        
        state.calculate_max_risk()
    }

    m.add_function(wrap_pyfunction!(governor_calculate_position_size, m)?)?;
    m.add_function(wrap_pyfunction!(governor_check_cluster_risk, m)?)?;
    m.add_function(wrap_pyfunction!(governor_calculate_max_risk, m)?)?;

    // Executor Functions
    #[pyfunction]
    fn executor_decide_trade(
        signal_size: f64,
        entropy_score: f64,
        regime: String,  // "ORDERED", "CHAOTIC", "TRANSITION"
    ) -> (String, f64) {
        use agents::executor::{TradeSignal, Direction, Regime, ExecutorState};
        
        let signal = TradeSignal {
            symbol: String::new(),
            direction: Direction::Buy,
            size: signal_size,
            price: 0.0,
            conviction: 0.5,
            stop_loss_price: None,
        };
        
        let regime_enum = match regime.as_str() {
            "CHAOTIC" => Regime::Chaotic,
            "TRANSITION" => Regime::Transition,
            _ => Regime::Ordered,
        };
        
        let state = ExecutorState::new(10.0);
        let decision = state.decide_trade(&signal, entropy_score, &regime_enum);
        
        let action_str = match decision.action {
            agents::executor::TradeAction::Execute => "EXECUTE",
            agents::executor::TradeAction::Halt => "HALT",
            agents::executor::TradeAction::Reduce => "REDUCE",
        };
        
        (action_str.to_string(), decision.adjusted_size)
    }

    m.add_function(wrap_pyfunction!(executor_decide_trade, m)?)?;

    // Oracle Functions
    #[pyfunction]
    fn oracle_analyze_for_entry(
        closes: Vec<f64>,
        rsi: Vec<f64>,
        bb_lower: Vec<f64>,
        bb_upper: Vec<f64>,
        obv_slope: f64,
        metabolism_state: String,
    ) -> (String, f64, String) {
        let signal = agents::oracle::OracleEngine::analyze_for_entry(
            &closes, &rsi, &bb_lower, &bb_upper, obv_slope, &metabolism_state
        );
        
        let direction = match signal.direction {
            agents::oracle::SignalDirection::Buy => "BUY",
            agents::oracle::SignalDirection::Sell => "SELL",
            agents::oracle::SignalDirection::Hold => "HOLD",
        };
        
        (direction.to_string(), signal.confidence, signal.reason)
    }

    #[pyfunction]
    fn oracle_calculate_market_bias(btc_returns: Vec<f64>, sentiment_score: f64) -> f64 {
        agents::oracle::OracleEngine::calculate_market_bias(&btc_returns, sentiment_score)
    }

    #[pyfunction]
    fn oracle_detect_accumulation(volumes: Vec<f64>, atr_current: f64, atr_average: f64) -> bool {
        agents::oracle::OracleEngine::detect_accumulation(&volumes, atr_current, atr_average)
    }

    m.add_function(wrap_pyfunction!(oracle_analyze_for_entry, m)?)?;
    m.add_function(wrap_pyfunction!(oracle_calculate_market_bias, m)?)?;
    m.add_function(wrap_pyfunction!(oracle_detect_accumulation, m)?)?;

    // Parallel WFO
    #[pyfunction]
    fn run_parallel_wfo(
        window_configs: Vec<(usize, usize, usize)>,  // (start, train_end, test_end)
        timestamps: Vec<i64>,
        opens: Vec<f64>,
        highs: Vec<f64>,
        lows: Vec<f64>,
        closes: Vec<f64>,
        rsi: Vec<f64>,
        bb_lower: Vec<f64>,
        bb_upper: Vec<f64>,
        obv_slopes: Vec<f64>,
        entropy_scores: Vec<f64>,
        initial_capital: f64,
    ) -> Vec<(usize, f64, f64, f64)> {
        let windows: Vec<parallel_wfo::WFOWindow> = window_configs
            .iter()
            .map(|(s, te, e)| parallel_wfo::WFOWindow {
                start_idx: *s,
                train_end_idx: *te,
                test_end_idx: *e,
            })
            .collect();

        let results = parallel_wfo::run_parallel_wfo(
            windows,
            &timestamps,
            &opens,
            &highs,
            &lows,
            &closes,
            &rsi,
            &bb_lower,
            &bb_upper,
            &obv_slopes,
            &entropy_scores,
            initial_capital,
        );

        results
            .iter()
            .map(|r| (r.window_id, r.train_pnl, r.test_pnl, r.final_balance))
            .collect()
    }

    m.add_function(wrap_pyfunction!(run_parallel_wfo, m)?)?;

    // ONNX Inference - RESTORED (Phase 2 Intelligence & Adaptation)
    #[cfg(feature = "onnx")]
    {
        #[pyfunction]
        fn onnx_predict_trend(model_path: &str, prices: Vec<f32>) -> PyResult<f32> {
            use crate::onnx_inference::OnnxPredictor;
            let mut predictor = OnnxPredictor::new(model_path)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            let prob = predictor.predict(&prices)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            Ok(prob)
        }
        m.add_function(wrap_pyfunction!(onnx_predict_trend, m)?)?;
    }

    // Kalman Filter
    #[pyfunction]
    fn kalman_filter_batch(
        measurements: Vec<f64>,
        process_noise: f64,
        measurement_noise: f64,
    ) -> Vec<f64> {
        let mut filter = core::KalmanFilter1D::new(process_noise, measurement_noise, 1.0);
        filter.update_batch(&measurements)
    }

    #[pyfunction]
    fn kalman_filter_single(
        measurement: f64,
        state: (f64, f64, bool),  // (x, p, initialized)
        process_noise: f64,
        measurement_noise: f64,
    ) -> (f64, f64, f64, bool) {  // (estimate, x, p, initialized)
        let mut filter = core::KalmanFilter1D::new(process_noise, measurement_noise, 1.0);
        filter.x = state.0;
        filter.p = state.1;
        // Initialize if state says so
        if state.2 {
            filter.update(state.0);  // Warm up
        }
        let estimate = filter.update(measurement);
        (estimate, filter.x, filter.p, true)
    }

    m.add_function(wrap_pyfunction!(kalman_filter_batch, m)?)?;
    m.add_function(wrap_pyfunction!(kalman_filter_single, m)?)?;

    // SHA-256 Hashing (for Executor Ledger)
    #[pyfunction]
    fn compute_block_hash(
        timestamp: String,
        entropy_score: f64,
        regime: String,
        action: String,
        prev_hash: String,
    ) -> String {
        core::compute_block_hash(&timestamp, entropy_score, &regime, &action, &prev_hash)
    }

    m.add_function(wrap_pyfunction!(compute_block_hash, m)?)?;

    // SDE Physics Functions
    #[pyfunction]
    fn sde_estimate_ou(prices: Vec<f64>, dt: f64) -> HashMap<String, f64> {
        math::sde::estimate_ou_parameters(&prices, dt)
    }

    #[pyfunction]
    fn sde_estimate_gbm(prices: Vec<f64>, dt: f64) -> HashMap<String, f64> {
        math::sde::estimate_gbm_parameters(&prices, dt)
    }

    #[pyfunction]
    fn sde_simulate_paths(
        model: String,
        params: HashMap<String, f64>,
        start_price: f64,
        horizon: usize,
        num_paths: usize,
        dt: f64,
    ) -> Vec<Vec<f64>> {
        math::sde::simulate_paths(&model, params, start_price, horizon, num_paths, dt)
    }

    #[pyfunction]
    fn sde_calculate_ruin_probability(
        model: String,
        params: HashMap<String, f64>,
        start_price: f64,
        sl_price: f64,
        tp_price: f64,
        horizon: usize,
        num_paths: usize,
        dt: f64,
    ) -> f64 {
        math::sde::calculate_ruin_probability(&model, params, start_price, sl_price, tp_price, horizon, num_paths, dt)
    }

    m.add_function(wrap_pyfunction!(sde_calculate_ruin_probability, m)?)?;

    // Batch Signal Analysis
    #[pyfunction]
    fn calculate_signals_matrix(
        symbols: Vec<String>,
        prices: HashMap<String, Vec<f64>>,
        highs: HashMap<String, Vec<f64>>,
        lows: HashMap<String, Vec<f64>>,
    ) -> HashMap<String, HashMap<String, f64>> {
        math::batch_analysis::calculate_signals_matrix(symbols, prices, highs, lows)
    }

    m.add_function(wrap_pyfunction!(calculate_signals_matrix, m)?)?;

    // === PHASE 3: EXECUTION ALGORITHMS ===
    
    /// Create TWAP executor state
    #[pyfunction]
    fn twap_create(
        symbol: &str,
        side: &str,
        total_qty: f64,
        duration_minutes: u64,
        num_slices: usize,
    ) -> HashMap<String, f64> {
        use execution_algorithms::TwapExecutor;
        
        let executor = TwapExecutor::new(symbol, side, total_qty, duration_minutes, num_slices);
        
        let mut state = HashMap::new();
        state.insert("total_quantity".to_string(), executor.total_quantity);
        state.insert("num_slices".to_string(), executor.num_slices as f64);
        state.insert("interval_seconds".to_string(), executor.interval_seconds as f64);
        state.insert("executed_quantity".to_string(), 0.0);
        state
    }

    /// Get next TWAP slice quantity
    #[pyfunction]
    fn twap_next_slice(
        total_qty: f64,
        num_slices: f64,
        executed: f64,
    ) -> Option<f64> {
        let remaining = total_qty - executed;
        let executed_slices = executed / (total_qty / num_slices);
        let remaining_slices = num_slices - executed_slices;
        
        if remaining_slices > 0.0 && remaining > 0.0 {
            Some((remaining / remaining_slices).max(0.0001))
        } else {
            None
        }
    }

    /// Generate VWAP volume profile
    #[pyfunction]
    fn vwap_generate_volume_profile(num_slices: usize) -> Vec<f64> {
        execution_algorithms::generate_volume_profile(num_slices)
    }

    /// Calculate order book imbalance
    #[pyfunction]
    fn calculate_order_imbalance(
        bids: Vec<(f64, f64)>,
        asks: Vec<(f64, f64)>,
    ) -> f64 {
        execution_algorithms::calculate_order_imbalance(&bids, &asks)
    }

    // === PHASE 3: ARBITRAGE DETECTION ===

    /// Calculate optimal arbitrage position size
    #[pyfunction]
    fn arb_calculate_position_size(
        profit_pct: f64,
        total_capital: f64,
        risk_per_trade: f64,
        confidence: f64,
    ) -> f64 {
        arbitrage::calculate_arb_position_size(profit_pct, total_capital, risk_per_trade, confidence)
    }

    m.add_function(wrap_pyfunction!(twap_create, m)?)?;
    m.add_function(wrap_pyfunction!(twap_next_slice, m)?)?;
    m.add_function(wrap_pyfunction!(vwap_generate_volume_profile, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_order_imbalance, m)?)?;
    m.add_function(wrap_pyfunction!(arb_calculate_position_size, m)?)?;

    // === PHASE 4: TRADER NEXUS (RUST CORE) ===
    // Note: Global instance managed from Python for simplicity

    use trader_nexus::{TraderNexus, NexusConfig};

    /// Create and initialize TraderNexus
    #[pyfunction]
    fn nexus_create(
        initial_capital: f64,
        fee_rate: f64,
        leverage: f64,
        stop_loss_pct: f64,
        take_profit_pct: f64,
        max_positions: usize,
    ) -> PyResult<Py<PyAny>> {
        use pyo3::types::PyDict;

        let config = NexusConfig {
            initial_capital,
            fee_rate,
            leverage,
            stop_loss_pct,
            take_profit_pct,
            max_positions,
            ..Default::default()
        };

        let nexus = TraderNexus::new(config);
        
        // Return as Python object (managed by Python)
        Python::with_gil(|py| {
            // Create a simple handle
            let dict = PyDict::new(py);
            dict.set_item("status", "initialized")?;
            dict.set_item("initial_capital", initial_capital)?;
            dict.set_item("equity", nexus.get_equity())?;
            
            Ok(dict.into())
        })
    }

    /// Run single trading cycle (stateless version - Python manages state)
    #[pyfunction]
    fn nexus_run_cycle(
        symbols: Vec<String>,
        closes: Vec<Vec<f64>>,
        rsi: Vec<Vec<f64>>,
        bb_lower: Vec<Vec<f64>>,
        bb_upper: Vec<Vec<f64>>,
        obv: Vec<f64>,
        entropy: Vec<f64>,
        initial_capital: f64,
    ) -> PyResult<Vec<String>> {
        // Create temporary nexus for this cycle
        let config = NexusConfig {
            initial_capital,
            ..Default::default()
        };
        let mut nexus = TraderNexus::new(config);

        // Build market data structure
        let mut market_data: HashMap<String, Vec<f64>> = HashMap::new();
        let mut indicators: HashMap<String, Vec<f64>> = HashMap::new();

        for (i, symbol) in symbols.iter().enumerate() {
            if i < closes.len() {
                market_data.insert(symbol.clone(), closes[i].clone());
            }
            if i < rsi.len() {
                indicators.insert(format!("{}_rsi", symbol), rsi[i].clone());
            }
            if i < bb_lower.len() {
                indicators.insert(format!("{}_bb_lower", symbol), bb_lower[i].clone());
            }
            if i < bb_upper.len() {
                indicators.insert(format!("{}_bb_upper", symbol), bb_upper[i].clone());
            }
            if i < obv.len() {
                indicators.insert(format!("{}_obv", symbol), vec![obv[i]]);
            }
            if i < entropy.len() {
                indicators.insert(format!("{}_entropy", symbol), vec![entropy[i]]);
            }
        }

        // Run cycle
        let signals = nexus.run_cycle(&market_data, &indicators);

        // Convert to simple string format
        let results: Vec<String> = signals.iter()
            .map(|s| format!("{}:{}:{:.2}", s.symbol, s.action, s.confidence))
            .collect();

        Ok(results)
    }

    /// Calculate position PnL (utility function)
    #[pyfunction]
    fn nexus_calculate_pnl(
        entry_price: f64,
        current_price: f64,
        quantity: f64,
        side: &str,
    ) -> PyResult<f64> {
        let pnl = if side == "BUY" {
            (current_price - entry_price) * quantity
        } else {
            (entry_price - current_price) * quantity
        };
        Ok(pnl)
    }

    /// Calculate position PnL percentage
    #[pyfunction]
    fn nexus_calculate_pnl_pct(
        entry_price: f64,
        current_price: f64,
        side: &str,
    ) -> PyResult<f64> {
        let pnl_pct = if side == "BUY" {
            (current_price - entry_price) / entry_price
        } else {
            (entry_price - current_price) / entry_price
        };
        Ok(pnl_pct)
    }

    /// Scan for arbitrage (stateless)
    #[pyfunction]
    fn nexus_scan_arbitrage(
        _min_profit_pct: f64,
    ) -> PyResult<Vec<String>> {
        // Simplified version - returns empty array
        // Full implementation would require proper price feed integration
        Ok(vec![])
    }

    m.add_function(wrap_pyfunction!(nexus_create, m)?)?;
    m.add_function(wrap_pyfunction!(nexus_run_cycle, m)?)?;
    m.add_function(wrap_pyfunction!(nexus_calculate_pnl, m)?)?;
    m.add_function(wrap_pyfunction!(nexus_calculate_pnl_pct, m)?)?;
    m.add_function(wrap_pyfunction!(nexus_scan_arbitrage, m)?)?;

    Ok(())
}
