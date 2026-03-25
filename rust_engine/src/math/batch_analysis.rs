use std::collections::HashMap;
use crate::math::indicators;
use crate::math::entropy;
use crate::math::sde;

pub struct SignalSet {
    pub rsi: f64,
    pub atr: f64,
    pub bb_upper: f64,
    pub bb_lower: f64,
    pub entropy: f64,
    pub perm_entropy: f64,
    pub lambda: f64,
    pub mu: f64,
    pub sigma: f64,
}

pub fn calculate_signals_matrix(
    symbols: Vec<String>,
    prices: HashMap<String, Vec<f64>>,
    highs: HashMap<String, Vec<f64>>,
    lows: HashMap<String, Vec<f64>>,
) -> HashMap<String, HashMap<String, f64>> {
    use rayon::prelude::*;

    symbols.into_par_iter().map(|sym| {
        let mut results = HashMap::new();
        
        if let Some(p) = prices.get(&sym) {
            if p.len() >= 30 {
                // 1. RSI
                let rsi_vec = indicators::calculate_rsi(p, 14);
                let rsi = *rsi_vec.last().unwrap_or(&50.0);
                results.insert("rsi".to_string(), rsi);
                
                // 2. Bollinger Bands
                let (upper, _mid, lower) = indicators::calculate_bollinger_bands(p, 20, 2.0);
                results.insert("bb_upper".to_string(), *upper.last().unwrap_or(&p[p.len()-1]));
                results.insert("bb_lower".to_string(), *lower.last().unwrap_or(&p[p.len()-1]));
                
                // 3. ATR
                if let (Some(h), Some(l)) = (highs.get(&sym), lows.get(&sym)) {
                    let atr_vec = indicators::calculate_atr(h, l, p, 14);
                    results.insert("atr".to_string(), *atr_vec.last().unwrap_or(&0.0));
                }
                
                // 4. Entropy (compute returns from raw prices first for Shannon)
                let returns: Vec<f64> = p.windows(2).map(|w| {
                    if w[0].abs() > 1e-12 { (w[1] - w[0]) / w[0] } else { 0.0 }
                }).collect();
                if returns.len() >= 10 {
                    results.insert("shannon_entropy".to_string(), entropy::calculate_shannon_entropy(&returns));
                } else {
                    results.insert("shannon_entropy".to_string(), entropy::calculate_shannon_entropy(p));
                }
                results.insert("perm_entropy".to_string(), entropy::calculate_permutation_entropy(p, 3, 1));
                
                // 5. SDE (OU Parameters)
                let ou = sde::estimate_ou_parameters(p, 1.0 / 35040.0);
                for (k, v) in ou {
                    results.insert(format!("ou_{}", k), v);
                }
            }
        }
        
        (sym, results)
    }).collect()
}
