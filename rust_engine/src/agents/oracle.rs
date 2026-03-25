/// Signal Direction
#[derive(Clone, Debug, PartialEq)]
pub enum SignalDirection {
    Buy,
    Sell,
    Hold,
}

/// Entry Signal Result
#[derive(Clone, Debug)]
pub struct EntrySignal {
    pub direction: SignalDirection,
    pub confidence: f64,  // 0.0 - 1.0
    pub reason: String,
}

/// Oracle State (Stateless for now - pure functions)
pub struct OracleEngine;

impl OracleEngine {
    /// Analyze for Entry Signal
    /// Implements Scavenger Logic (Mean Reversion)
    /// 
    /// Entry Conditions:
    /// 1. Price < Lower Bollinger Band (Oversold)
    /// 2. RSI < 30 (Confirmation)
    /// 3. OBV Slope > 0 (Hidden Accumulation)
    pub fn analyze_for_entry(
        closes: &[f64],
        rsi: &[f64],
        bb_lower: &[f64],
        bb_upper: &[f64],
        obv_slope: f64,
        metabolism_state: &str,  // "SCAVENGER" or "PREDATOR"
    ) -> EntrySignal {
        if closes.is_empty() || rsi.is_empty() || bb_lower.is_empty() {
            return EntrySignal {
                direction: SignalDirection::Hold,
                confidence: 0.0,
                reason: "Insufficient data".to_string(),
            };
        }

        let last_idx = closes.len() - 1;
        let current_price = closes[last_idx];
        let current_rsi = rsi[last_idx];
        let lower_bb = bb_lower[last_idx];
        let upper_bb = bb_upper[last_idx];

        // Check for NaN
        if current_price.is_nan() || current_rsi.is_nan() || lower_bb.is_nan() {
            return EntrySignal {
                direction: SignalDirection::Hold,
                confidence: 0.0,
                reason: "NaN in data".to_string(),
            };
        }

        if metabolism_state == "SCAVENGER" {
            // Scavenger Logic: Mean Reversion
            // BUY when price is below Lower BB and RSI is oversold
            
            let below_bb = current_price < lower_bb;
            let rsi_oversold = current_rsi < 30.0;
            let accumulation = obv_slope > 0.0;

            if below_bb && rsi_oversold && accumulation {
                return EntrySignal {
                    direction: SignalDirection::Buy,
                    confidence: 0.8,
                    reason: "Scavenger: BB Oversold + RSI < 30 + OBV Accumulation".to_string(),
                };
            }

            // Weaker condition: Just BB
            if below_bb && rsi_oversold {
                return EntrySignal {
                    direction: SignalDirection::Buy,
                    confidence: 0.6,
                    reason: "Scavenger: BB Oversold + RSI < 30".to_string(),
                };
            }

        } else {
            // PREDATOR Logic: Trend Following
            // BUY when price breaks above Upper BB with momentum
            
            let above_bb = current_price > upper_bb;
            let rsi_strong = current_rsi > 50.0 && current_rsi < 70.0;
            let momentum = obv_slope > 0.0;

            if above_bb && rsi_strong && momentum {
                return EntrySignal {
                    direction: SignalDirection::Buy,
                    confidence: 0.75,
                    reason: "Predator: BB Breakout + Strong RSI + OBV Momentum".to_string(),
                };
            }
        }

        // No Signal
        EntrySignal {
            direction: SignalDirection::Hold,
            confidence: 0.0,
            reason: "No entry conditions met".to_string(),
        }
    }

    /// Calculate Global Market Bias
    /// Combines BTC trend and sentiment
    pub fn calculate_market_bias(
        btc_returns: &[f64],
        sentiment_score: f64,  // -1.0 to 1.0
    ) -> f64 {
        if btc_returns.is_empty() {
            return 0.0;
        }

        // BTC Momentum (last 20 returns average)
        let window = 20.min(btc_returns.len());
        let btc_momentum: f64 = btc_returns[btc_returns.len()-window..].iter().sum::<f64>() / window as f64;

        // Combine: 60% BTC momentum, 40% sentiment
        let bias = (btc_momentum * 100.0 * 0.6) + (sentiment_score * 0.4);
        
        // Clamp to -1.0 to 1.0
        bias.clamp(-1.0, 1.0)
    }

    /// Detect Accumulation Pattern
    /// High volume + Low volatility = Whale absorption
    pub fn detect_accumulation(
        volumes: &[f64],
        atr_current: f64,
        atr_average: f64,
    ) -> bool {
        if volumes.is_empty() || atr_average <= 0.0 {
            return false;
        }

        // Volume spike: Current > 1.5x average
        let avg_volume: f64 = volumes.iter().sum::<f64>() / volumes.len() as f64;
        let current_volume = *volumes.last().unwrap_or(&0.0);
        let volume_spike = current_volume > (avg_volume * 1.5);

        // Low volatility: ATR below average
        let low_vol = atr_current < atr_average;

        volume_spike && low_vol
    }
}
