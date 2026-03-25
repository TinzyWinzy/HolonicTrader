use rand::prelude::*;
use rand_distr::{Normal, Distribution};
use std::collections::HashMap;

pub fn estimate_ou_parameters(prices: &[f64], dt: f64) -> HashMap<String, f64> {
    let n = prices.len();
    if n < 2 {
        return HashMap::new();
    }

    // AR(1) logic for OU: X_{i+1} = a*X_i + b + epsilon
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_xy = 0.0;

    for i in 0..n - 1 {
        let x = prices[i];
        let y = prices[i + 1];
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }

    let n_f = (n - 1) as f64;
    let denominator = n_f * sum_xx - sum_x * sum_x;
    if denominator.abs() < 1e-12 {
        return HashMap::new();
    }

    let a = (n_f * sum_xy - sum_x * sum_y) / denominator;
    let b = (sum_y - a * sum_x) / n_f;

    // lambda = -ln(a) / dt
    let lambda = if a > 0.0 { -a.ln() / dt } else { 0.0001 }; // Safety fallback
    
    // mu = b / (1 - a)
    let mu = if (1.0 - a).abs() > 1e-12 { b / (1.0 - a) } else { sum_x / n_f };

    // sigma
    let mut sum_sq_resid = 0.0;
    for i in 0..n - 1 {
        let resid = prices[i + 1] - (a * prices[i] + b);
        sum_sq_resid += resid * resid;
    }
    let variance = sum_sq_resid / n_f;
    
    // sigma_ou = sqrt(var * 2 * lambda / (1 - a^2))
    let sigma = if a.abs() < 1.0 {
        (variance * 2.0 * lambda / (1.0 - a * a)).sqrt()
    } else {
        variance.sqrt() // Fallback to simple std
    };

    let mut map = HashMap::new();
    map.insert("lambda".to_string(), lambda);
    map.insert("mu".to_string(), mu);
    map.insert("sigma".to_string(), sigma);
    map.insert("drift".to_string(), lambda * (mu - prices[n - 1])); // Instantaneous drift
    map.insert("a".to_string(), a); // Store exact AR1 coefficient
    map.insert("b".to_string(), b); // Store exact AR1 intercept
    map.insert("residual_sigma".to_string(), variance.sqrt()); // Store exact residual std
    
    map
}

pub fn estimate_gbm_parameters(prices: &[f64], dt: f64) -> HashMap<String, f64> {
    let n = prices.len();
    if n < 2 {
        return HashMap::new();
    }

    let mut returns = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        if prices[i] > 0.0 {
            returns.push((prices[i + 1] / prices[i]).ln());
        }
    }

    if returns.is_empty() {
        return HashMap::new();
    }

    let sum: f64 = returns.iter().sum();
    let mean = sum / returns.len() as f64;
    
    let sq_sum: f64 = returns.iter().map(|x| (x - mean).powi(2)).sum();
    let variance = (sq_sum / returns.len() as f64) / dt;
    let sigma = variance.sqrt();
    
    // mu = mean/dt + 0.5 * sigma^2
    let mu = (mean / dt) + 0.5 * variance;

    let mut map = HashMap::new();
    map.insert("mu".to_string(), mu);
    map.insert("sigma".to_string(), sigma);
    map.insert("drift".to_string(), mu); // Constant drift in GBM
    
    map
}

pub fn simulate_paths(
    model: &str,
    params: HashMap<String, f64>,
    start_price: f64,
    horizon: usize,
    num_paths: usize,
    dt: f64,
) -> Vec<Vec<f64>> {
    let mu = *params.get("mu").unwrap_or(&0.0);
    let sigma = *params.get("sigma").unwrap_or(&0.1);
    
    // For OU Exact Simulation
    let a = *params.get("a").unwrap_or(&0.999);
    let b = *params.get("b").unwrap_or(&0.0);
    let residual_sigma = *params.get("residual_sigma").unwrap_or(&0.01);

    use rayon::prelude::*;

    let all_paths = (0..num_paths).into_par_iter().map(|_| {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut path = Vec::with_capacity(horizon);
        let mut current_price = start_price;
        path.push(current_price);

        for _ in 1..horizon {
            let z = normal.sample(&mut rng);
            
            if model == "OU" {
                // Exact discrete AR(1) update avoids Euler explosion
                current_price = a * current_price + b + residual_sigma * z;
            } else {
                let drift = mu - 0.5 * sigma.powi(2);
                current_price *= (drift * dt + sigma * dt.sqrt() * z).exp();
            }
            path.push(current_price);
        }
        path
    }).collect();

    all_paths
}

pub fn calculate_ruin_probability(
    model: &str,
    params: HashMap<String, f64>,
    start_price: f64,
    sl_price: f64,
    tp_price: f64,
    horizon: usize,
    num_paths: usize,
    dt: f64,
) -> f64 {
    let mu = *params.get("mu").unwrap_or(&0.0);
    let sigma = *params.get("sigma").unwrap_or(&0.1);
    
    // For OU Exact Simulation
    let a = *params.get("a").unwrap_or(&0.999);
    let b = *params.get("b").unwrap_or(&0.0);
    let residual_sigma = *params.get("residual_sigma").unwrap_or(&0.01);
    
    let is_buy = start_price < tp_price; // Simple heuristic for direction

    use rayon::prelude::*;

    let failures = (0..num_paths).into_par_iter().map(|_| {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut current_price = start_price;
        let mut hit_sl = false;

        for _ in 1..horizon {
            let z = normal.sample(&mut rng);
            
            if model == "OU" {
                current_price = a * current_price + b + residual_sigma * z;
            } else {
                let drift = mu - 0.5 * sigma.powi(2);
                current_price *= (drift * dt + sigma * dt.sqrt() * z).exp();
            }

            if is_buy {
                if current_price <= sl_price {
                    hit_sl = true;
                    break;
                }
                if current_price >= tp_price {
                    break;
                }
            } else {
                if current_price >= sl_price {
                    hit_sl = true;
                    break;
                }
                if current_price <= tp_price {
                    break;
                }
            }
        }
        if hit_sl { 1 } else { 0 }
    }).sum::<i32>();

    failures as f64 / num_paths as f64
}
