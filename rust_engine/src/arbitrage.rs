/// Phase 3: Arbitrage Detection
/// Spatial (cross-exchange) and Triangular arbitrage opportunities

use std::collections::HashMap;

/// Spatial Arbitrage Opportunity (cross-exchange)
#[derive(Clone, Debug)]
pub struct SpatialArbOpportunity {
    pub base_asset: String,      // e.g., "BTC"
    pub quote_asset: String,     // e.g., "USDT"
    pub buy_exchange: String,    // Exchange to buy on
    pub sell_exchange: String,   // Exchange to sell on
    pub buy_price: f64,          // Price on buy exchange
    pub sell_price: f64,         // Price on sell exchange
    pub spread_pct: f64,         // Spread percentage
    pub max_quantity: f64,       // Maximum executable quantity
    pub expected_profit_pct: f64, // Expected profit after fees
    pub timestamp: u64,
}

/// Triangular Arbitrage Opportunity (single exchange, 3 legs)
#[derive(Clone, Debug)]
pub struct TriangularArbOpportunity {
    pub exchange: String,
    pub leg1: TriangularLeg,     // e.g., BTC -> USDT
    pub leg2: TriangularLeg,     // e.g., USDT -> ETH
    pub leg3: TriangularLeg,     // e.g., ETH -> BTC
    pub expected_profit_pct: f64,
    pub max_quantity: f64,
    pub timestamp: u64,
}

/// Single leg of triangular arbitrage
#[derive(Clone, Debug)]
pub struct TriangularLeg {
    pub from_asset: String,
    pub to_asset: String,
    pub price: f64,
    pub quantity: f64,
}

/// Exchange price data
#[derive(Clone, Debug)]
pub struct ExchangePrice {
    pub exchange: String,
    pub symbol: String,
    pub bid: f64,
    pub ask: f64,
    pub bid_size: f64,
    pub ask_size: f64,
    pub timestamp: u64,
}

/// Fee structure for an exchange
#[derive(Clone, Debug)]
pub struct FeeStructure {
    pub maker_fee: f64,  // Taker fee percentage
    pub taker_fee: f64,  // Maker fee percentage
    pub withdrawal_fee: f64, // Fixed withdrawal fee
}

impl FeeStructure {
    pub fn new(maker: f64, taker: f64, withdrawal: f64) -> Self {
        FeeStructure {
            maker_fee: maker,
            taker_fee: taker,
            withdrawal_fee: withdrawal,
        }
    }

    /// Total fee for a round-trip trade
    pub fn round_trip_fee(&self) -> f64 {
        self.taker_fee * 2.0 // Assume taker for both legs
    }
}

/// Arbitrage Scanner
pub struct ArbScanner {
    pub prices: HashMap<String, ExchangePrice>, // symbol_exchange -> price
    pub fees: HashMap<String, FeeStructure>,    // exchange -> fees
    pub min_profit_threshold: f64,              // Minimum profit % to consider
    pub max_execution_time_ms: u64,             // Max age of price data
}

impl ArbScanner {
    pub fn new(min_profit_pct: f64, max_age_ms: u64) -> Self {
        ArbScanner {
            prices: HashMap::new(),
            fees: HashMap::new(),
            min_profit_threshold: min_profit_pct,
            max_execution_time_ms: max_age_ms,
        }
    }

    /// Update price for a symbol on an exchange
    pub fn update_price(&mut self, price: ExchangePrice) {
        let key = format!("{}_{}", price.symbol, price.exchange);
        self.prices.insert(key, price);
    }

    /// Set fee structure for an exchange
    pub fn set_fees(&mut self, exchange: &str, fees: FeeStructure) {
        self.fees.insert(exchange.to_string(), fees);
    }

    /// Scan for spatial arbitrage opportunities
    pub fn scan_spatial_arb(&self) -> Vec<SpatialArbOpportunity> {
        let mut opportunities = Vec::new();

        // Group prices by symbol
        let mut symbol_prices: HashMap<String, Vec<&ExchangePrice>> = HashMap::new();
        
        for (_key, price) in &self.prices {
            // Skip stale prices
            if self.is_price_stale(price) {
                continue;
            }

            let symbol = price.symbol.clone();
            symbol_prices.entry(symbol).or_insert_with(Vec::new).push(price);
        }

        // Check each symbol for cross-exchange opportunities
        for (symbol, prices) in symbol_prices {
            if prices.len() < 2 {
                continue;
            }

            // Find best bid and ask across exchanges
            let mut best_bid_exchange = "";
            let mut best_bid_price = 0.0;
            let mut best_bid_size = 0.0;

            let mut best_ask_exchange = "";
            let mut best_ask_price = f64::MAX;
            let mut best_ask_size = 0.0;

            for price in &prices {
                if price.bid > best_bid_price {
                    best_bid_price = price.bid;
                    best_bid_exchange = &price.exchange;
                    best_bid_size = price.bid_size;
                }
                if price.ask < best_ask_price {
                    best_ask_price = price.ask;
                    best_ask_exchange = &price.exchange;
                    best_ask_size = price.ask_size;
                }
            }

            // Check if arbitrage opportunity exists
            if best_bid_exchange != best_ask_exchange && best_bid_price > best_ask_price {
                let spread_pct = (best_bid_price - best_ask_price) / best_ask_price * 100.0;
                
                // Calculate fees
                let buy_fee = self.get_exchange_fee(best_ask_exchange);
                let sell_fee = self.get_exchange_fee(best_bid_exchange);
                let total_fees = buy_fee.round_trip_fee() + sell_fee.round_trip_fee();

                let expected_profit = spread_pct - total_fees;

                if expected_profit >= self.min_profit_threshold {
                    let max_qty = best_bid_size.min(best_ask_size);
                    
                    opportunities.push(SpatialArbOpportunity {
                        base_asset: symbol.split('/').next().unwrap_or("UNKNOWN").to_string(),
                        quote_asset: symbol.split('/').last().unwrap_or("USD").to_string(),
                        buy_exchange: best_ask_exchange.to_string(),
                        sell_exchange: best_bid_exchange.to_string(),
                        buy_price: best_ask_price,
                        sell_price: best_bid_price,
                        spread_pct,
                        max_quantity: max_qty,
                        expected_profit_pct: expected_profit,
                        timestamp: current_timestamp_ms(),
                    });
                }
            }
        }

        // Sort by profit descending
        opportunities.sort_by(|a, b| {
            b.expected_profit_pct.partial_cmp(&a.expected_profit_pct).unwrap()
        });

        opportunities
    }

    /// Scan for triangular arbitrage opportunities
    pub fn scan_triangular_arb(&self, exchange: &str) -> Vec<TriangularArbOpportunity> {
        let mut opportunities = Vec::new();

        // Common triangular paths
        let triangles = vec![
            // BTC triangles
            ("BTC", "USDT", "ETH"),
            ("BTC", "USDT", "SOL"),
            ("ETH", "BTC", "SOL"),
            // USDT triangles
            ("USDT", "BTC", "ETH"),
        ];

        for (asset1, asset2, asset3) in triangles {
            if let Some(opportunity) = self.check_triangle(exchange, asset1, asset2, asset3) {
                if opportunity.expected_profit_pct >= self.min_profit_threshold {
                    opportunities.push(opportunity);
                }
            }
        }

        opportunities
    }

    /// Check a specific triangular path
    fn check_triangle(
        &self,
        exchange: &str,
        asset1: &str,
        asset2: &str,
        asset3: &str,
    ) -> Option<TriangularArbOpportunity> {
        // Build symbols
        let symbol1 = format!("{}/{}", asset1, asset2);
        let symbol2 = format!("{}/{}", asset2, asset3);
        let symbol3 = format!("{}/{}", asset3, asset1);

        // Get prices for all three legs
        let price1 = self.get_price(exchange, &symbol1)?;
        let price2 = self.get_price(exchange, &symbol2)?;
        let price3 = self.get_price(exchange, &symbol3)?;

        // Check if prices are stale
        if self.is_price_stale(price1) || self.is_price_stale(price2) || self.is_price_stale(price3) {
            return None;
        }

        // Calculate triangular arbitrage
        // Start with 1 unit of asset1
        let start_amount = 1.0;
        
        // Leg 1: asset1 -> asset2 (sell asset1 for asset2)
        let amount2 = start_amount * price1.bid;
        
        // Leg 2: asset2 -> asset3 (sell asset2 for asset3)
        let amount3 = amount2 * price2.bid;
        
        // Leg 3: asset3 -> asset1 (sell asset3 for asset1)
        let final_amount = amount3 * price3.bid;

        // Calculate profit
        let profit_pct = (final_amount - start_amount) / start_amount * 100.0;

        // Get fees for this exchange
        let fees = self.get_exchange_fee(exchange);
        let total_fees = fees.round_trip_fee() * 3.0; // 3 legs

        let expected_profit = profit_pct - total_fees;

        if expected_profit > 0.0 {
            let max_qty = price1.bid_size.min(price2.bid_size).min(price3.bid_size);

            Some(TriangularArbOpportunity {
                exchange: exchange.to_string(),
                leg1: TriangularLeg {
                    from_asset: asset1.to_string(),
                    to_asset: asset2.to_string(),
                    price: price1.bid,
                    quantity: start_amount,
                },
                leg2: TriangularLeg {
                    from_asset: asset2.to_string(),
                    to_asset: asset3.to_string(),
                    price: price2.bid,
                    quantity: amount2,
                },
                leg3: TriangularLeg {
                    from_asset: asset3.to_string(),
                    to_asset: asset1.to_string(),
                    price: price3.bid,
                    quantity: amount3,
                },
                expected_profit_pct: expected_profit,
                max_quantity: max_qty,
                timestamp: current_timestamp_ms(),
            })
        } else {
            None
        }
    }

    /// Get price for symbol on exchange
    fn get_price(&self, exchange: &str, symbol: &str) -> Option<&ExchangePrice> {
        let key = format!("{}_{}", symbol, exchange);
        self.prices.get(&key)
    }

    /// Get fee structure for exchange
    fn get_exchange_fee(&self, exchange: &str) -> FeeStructure {
        self.fees.get(exchange).cloned().unwrap_or_else(|| {
            // Default fees if not specified
            FeeStructure::new(0.001, 0.002, 0.0) // 0.1% maker, 0.2% taker
        })
    }

    /// Check if price data is stale
    fn is_price_stale(&self, price: &ExchangePrice) -> bool {
        let now = current_timestamp_ms();
        now - price.timestamp > self.max_execution_time_ms
    }

    /// Get all opportunities (spatial + triangular)
    pub fn scan_all(&self) -> (Vec<SpatialArbOpportunity>, Vec<TriangularArbOpportunity>) {
        let spatial = self.scan_spatial_arb();
        
        let mut triangular = Vec::new();
        for exchange in self.fees.keys() {
            triangular.extend(self.scan_triangular_arb(exchange));
        }

        (spatial, triangular)
    }
}

/// Get current timestamp in milliseconds
fn current_timestamp_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// Calculate optimal position size for arbitrage
pub fn calculate_arb_position_size(
    opportunity_profit_pct: f64,
    total_capital: f64,
    risk_per_trade: f64,
    confidence: f64,
) -> f64 {
    // Kelly-inspired sizing for arbitrage
    let win_probability = 0.95; // Arbitrage is high probability
    let win_loss_ratio = opportunity_profit_pct / (risk_per_trade * 100.0);
    
    let kelly_fraction = if win_loss_ratio > 0.0 {
        (win_probability * win_loss_ratio - (1.0 - win_probability)) / win_loss_ratio
    } else {
        0.0
    };

    // Apply confidence and risk limits
    let sized_position = total_capital * kelly_fraction * confidence;
    sized_position.min(total_capital * risk_per_trade)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_arb_detection() {
        let mut scanner = ArbScanner::new(0.1, 5000); // 0.1% min profit, 5s max age

        // Set fees
        scanner.set_fees("exchange_a", FeeStructure::new(0.001, 0.002, 0.0));
        scanner.set_fees("exchange_b", FeeStructure::new(0.001, 0.002, 0.0));

        // Add prices: exchange_a has lower ask, exchange_b has higher bid
        scanner.update_price(ExchangePrice {
            exchange: "exchange_a".to_string(),
            symbol: "BTC/USDT".to_string(),
            bid: 49900.0,
            ask: 49950.0,
            bid_size: 1.0,
            ask_size: 1.0,
            timestamp: current_timestamp_ms(),
        });

        scanner.update_price(ExchangePrice {
            exchange: "exchange_b".to_string(),
            symbol: "BTC/USDT".to_string(),
            bid: 50050.0,
            ask: 50100.0,
            bid_size: 1.0,
            ask_size: 1.0,
            timestamp: current_timestamp_ms(),
        });

        let opportunities = scanner.scan_spatial_arb();
        
        assert!(!opportunities.is_empty());
        assert_eq!(opportunities[0].buy_exchange, "exchange_a");
        assert_eq!(opportunities[0].sell_exchange, "exchange_b");
        assert!(opportunities[0].expected_profit_pct > 0.0);
    }

    #[test]
    fn test_fee_calculation() {
        let fees = FeeStructure::new(0.001, 0.002, 0.0);
        assert_eq!(fees.round_trip_fee(), 0.004); // 0.4% round trip
    }
}
