/// Phase 3: Advanced Execution Algorithms
/// TWAP (Time-Weighted Average Price) and VWAP (Volume-Weighted Average Price) execution

use std::time::{Duration, Instant};

/// TWAP Execution State
#[derive(Clone, Debug)]
pub struct TwapExecutor {
    pub total_quantity: f64,
    pub executed_quantity: f64,
    pub num_slices: usize,
    pub executed_slices: usize,
    pub interval_seconds: u64,
    pub start_time: Option<Instant>,
    pub symbol: String,
    pub side: String, // "BUY" or "SELL"
}

/// VWAP Execution State
#[derive(Clone, Debug)]
pub struct VwapExecutor {
    pub total_quantity: f64,
    pub executed_quantity: f64,
    pub target_vwap: f64,
    pub executed_vwap: f64,
    pub volume_profile: Vec<f64>, // Expected volume distribution per slice
    pub current_slice: usize,
    pub symbol: String,
    pub side: String,
}

/// Execution slice result
#[derive(Clone, Debug)]
pub struct ExecutionSlice {
    pub quantity: f64,
    pub price: f64,
    pub timestamp: u64,
    pub filled: bool,
}

/// TWAP Executor Implementation
impl TwapExecutor {
    /// Create new TWAP executor
    pub fn new(symbol: &str, side: &str, total_qty: f64, duration_minutes: u64, num_slices: usize) -> Self {
        let interval_seconds = (duration_minutes * 60) / num_slices as u64;
        
        TwapExecutor {
            total_quantity: total_qty,
            executed_quantity: 0.0,
            num_slices,
            executed_slices: 0,
            interval_seconds,
            start_time: None,
            symbol: symbol.to_string(),
            side: side.to_string(),
        }
    }

    /// Get next slice quantity to execute
    pub fn next_slice(&mut self) -> Option<f64> {
        if self.executed_slices >= self.num_slices {
            return None;
        }

        if self.start_time.is_none() {
            self.start_time = Some(Instant::now());
        }

        // Calculate remaining quantity to ensure we complete the order
        let remaining_qty = self.total_quantity - self.executed_quantity;
        let remaining_slices = self.num_slices - self.executed_slices;
        
        // Equal distribution with remainder handling
        let slice_qty = remaining_qty / remaining_slices as f64;
        
        Some(slice_qty.max(0.0001)) // Minimum quantity threshold
    }

    /// Record a filled slice
    pub fn record_fill(&mut self, quantity: f64, _price: f64) {
        self.executed_quantity += quantity;
        self.executed_slices += 1;
    }

    /// Check if execution is complete
    pub fn is_complete(&self) -> bool {
        self.executed_slices >= self.num_slices || 
        self.executed_quantity >= self.total_quantity * 0.99 // 99% filled threshold
    }

    /// Get execution progress (0.0 to 1.0)
    pub fn progress(&self) -> f64 {
        self.executed_quantity / self.total_quantity
    }

    /// Get time until next slice
    pub fn time_to_next_slice(&self) -> Option<Duration> {
        self.start_time.map(|start| {
            let elapsed = start.elapsed().as_secs();
            let next_slice_time = (self.executed_slices as u64) * self.interval_seconds;
            
            if elapsed >= next_slice_time + self.interval_seconds {
                Duration::from_secs(0) // Ready now
            } else {
                Duration::from_secs(next_slice_time + self.interval_seconds - elapsed)
            }
        })
    }

    /// Reset executor for reuse
    pub fn reset(&mut self) {
        self.executed_quantity = 0.0;
        self.executed_slices = 0;
        self.start_time = None;
    }
}

/// VWAP Executor Implementation
impl VwapExecutor {
    /// Create new VWAP executor with volume profile
    pub fn new(
        symbol: &str, 
        side: &str, 
        total_qty: f64,
        volume_profile: Vec<f64>,
    ) -> Self {
        let total_volume: f64 = volume_profile.iter().sum();
        let normalized_profile: Vec<f64> = if total_volume > 0.0 {
            volume_profile.iter().map(|&v| v / total_volume).collect()
        } else {
            // Equal distribution if no profile provided
            let n = volume_profile.len().max(1);
            vec![1.0 / n as f64; n]
        };

        VwapExecutor {
            total_quantity: total_qty,
            executed_quantity: 0.0,
            target_vwap: 0.0,
            executed_vwap: 0.0,
            volume_profile: normalized_profile,
            current_slice: 0,
            symbol: symbol.to_string(),
            side: side.to_string(),
        }
    }

    /// Get next slice quantity based on volume profile
    pub fn next_slice(&mut self, current_vwap: f64) -> Option<f64> {
        if self.current_slice >= self.volume_profile.len() {
            return None;
        }

        // Update target VWAP
        self.target_vwap = current_vwap;

        // Calculate slice quantity based on volume profile
        let profile_weight = self.volume_profile[self.current_slice];
        let slice_qty = self.total_quantity * profile_weight;

        Some(slice_qty.max(0.0001))
    }

    /// Record a filled slice with execution price
    pub fn record_fill(&mut self, quantity: f64, price: f64) {
        // Update running VWAP
        let total_value = self.executed_vwap * self.executed_quantity + price * quantity;
        self.executed_quantity += quantity;
        self.executed_vwap = if self.executed_quantity > 0.0 {
            total_value / self.executed_quantity
        } else {
            0.0
        };

        self.current_slice += 1;
    }

    /// Check if execution is complete
    pub fn is_complete(&self) -> bool {
        self.current_slice >= self.volume_profile.len() ||
        self.executed_quantity >= self.total_quantity * 0.99
    }

    /// Get execution progress (0.0 to 1.0)
    pub fn progress(&self) -> f64 {
        self.executed_quantity / self.total_quantity
    }

    /// Get VWAP slippage (positive = worse than target)
    pub fn vwap_slippage(&self) -> f64 {
        if self.target_vwap > 0.0 && self.executed_vwap > 0.0 {
            (self.executed_vwap - self.target_vwap) / self.target_vwap
        } else {
            0.0
        }
    }

    /// Performance score (1.0 = perfect, <1.0 = slippage)
    pub fn performance_score(&self) -> f64 {
        let slippage = self.vwap_slippage().abs();
        (1.0 - slippage * 10.0).max(0.0) // 10% slippage = 0 score
    }

    /// Reset executor for reuse
    pub fn reset(&mut self) {
        self.executed_quantity = 0.0;
        self.executed_vwap = 0.0;
        self.current_slice = 0;
    }
}

/// Order Book Imbalance for VWAP prediction
pub fn calculate_order_imbalance(bids: &[(f64, f64)], asks: &[(f64, f64)]) -> f64 {
    // bids: [(price, size)], asks: [(price, size)]
    let bid_volume: f64 = bids.iter().map(|(_, size)| size).sum();
    let ask_volume: f64 = asks.iter().map(|(_, size)| size).sum();
    
    let total = bid_volume + ask_volume;
    if total > 0.0 {
        (bid_volume - ask_volume) / total
    } else {
        0.0
    }
}

/// Generate typical volume profile for a trading day (intraday pattern)
pub fn generate_volume_profile(num_slices: usize) -> Vec<f64> {
    // U-shaped pattern: high volume at open/close, low at midday
    let mut profile = Vec::with_capacity(num_slices);
    
    for i in 0..num_slices {
        let t = i as f64 / num_slices as f64;
        // U-curve: 4t^2 - 4t + 1.5 (normalized)
        let weight = 4.0 * t * t - 4.0 * t + 1.5;
        profile.push(weight.max(0.5));
    }
    
    // Normalize
    let total: f64 = profile.iter().sum();
    profile.iter().map(|&w| w / total).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twap_execution() {
        let mut twap = TwapExecutor::new("BTC/USDT", "BUY", 1.0, 60, 6);
        
        // Should have 6 slices
        assert_eq!(twap.num_slices, 6);
        
        // Execute all slices
        for i in 0..6 {
            let slice = twap.next_slice();
            assert!(slice.is_some());
            twap.record_fill(slice.unwrap(), 50000.0);
        }
        
        assert!(twap.is_complete());
        assert!(twap.progress() >= 0.99);
    }

    #[test]
    fn test_vwap_execution() {
        let profile = generate_volume_profile(10);
        let mut vwap = VwapExecutor::new("ETH/USDT", "SELL", 10.0, profile);
        
        // Execute with varying prices
        let prices = vec![2000.0, 2001.0, 1999.0, 2002.0, 2000.0, 1998.0, 2001.0, 2003.0, 2000.0, 1999.0];
        
        for price in prices {
            let slice = vwap.next_slice(2000.0);
            if let Some(qty) = slice {
                vwap.record_fill(qty, price);
            }
        }
        
        assert!(vwap.is_complete());
        assert!(vwap.performance_score() > 0.0);
    }

    #[test]
    fn test_volume_profile() {
        let profile = generate_volume_profile(12);
        assert_eq!(profile.len(), 12);
        
        // Sum should be 1.0
        let total: f64 = profile.iter().sum();
        assert!((total - 1.0).abs() < 0.001);
        
        // First and last should be higher than middle
        assert!(profile[0] > profile[5]);
        assert!(profile[11] > profile[5]);
    }
}
