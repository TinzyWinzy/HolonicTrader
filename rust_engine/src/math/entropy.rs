use std::f64;

/// Calculate Shannon Entropy
/// H(X) = -sum(p_i * ln(p_i))
/// Matches scipy.stats.entropy logic with 10 bins
pub fn calculate_shannon_entropy(data: &[f64]) -> f64 {
    let probs = compute_probabilities(data, 10);
    
    let mut entropy = 0.0;
    for &p in &probs {
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }
    
    entropy // Returns in nats (natural log base)
}

/// Calculate Rényi Entropy
/// H_alpha(X) = (1 / (1 - alpha)) * ln(sum(p_i ^ alpha))
pub fn calculate_renyi_entropy(data: &[f64], alpha: f64) -> f64 {
    let probs = compute_probabilities(data, 10);
    
    // Check for Shannon limit
    if (alpha - 1.0).abs() < 1e-9 {
        return calculate_shannon_entropy(data);
    }
    
    let mut sum_p_alpha = 0.0;
    for &p in &probs {
        sum_p_alpha += p.powf(alpha);
    }
    
    if sum_p_alpha == 0.0 {
        return 0.0;
    }
    
    (1.0 / (1.0 - alpha)) * sum_p_alpha.ln()
}

/// Helper: Discretize data into bins and return probabilities
fn compute_probabilities(data: &[f64], bins: usize) -> Vec<f64> {
    if data.is_empty() {
        return vec![0.0; bins];
    }

    let mut min_val = f64::MAX;
    let mut max_val = f64::MIN;
    
    for &x in data {
        if x < min_val { min_val = x; }
        if x > max_val { max_val = x; }
    }
    
    // Avoid division by zero if all values are same
    if (max_val - min_val).abs() < 1e-9 {
        // All in one bin
        let mut probs = vec![0.0; bins];
        probs[0] = 1.0;
        return probs;
    }

    let mut counts = vec![0; bins];
    let bin_width = (max_val - min_val) / bins as f64;
    
    // Add small epsilon to max validation logic to include the max value in last bin
    let _epsilon = 1e-9; 

    for &x in data {
        // Map x to bin index 0..bins-1
        // bin = floor((x - min) / width)
        let mut idx = ((x - min_val) / bin_width).floor() as usize;
        if idx >= bins {
            idx = bins - 1;
        }
        counts[idx] += 1;
    }
    
    let total_count = data.len() as f64;
    counts.iter().map(|&c| c as f64 / total_count).collect()
}

// === NEW AEHML 2.0 FUNCTIONS ===

/// Calculate Permutation Entropy (Bandt & Pompe, 2002)
/// Measures the complexity of the causality structure of the time series.
/// Embedding Dimension (m): Typically 3 to 7.
/// Delay (tau): Typically 1.
pub fn calculate_permutation_entropy(data: &[f64], m: usize, delay: usize) -> f64 {
    let n = data.len();
    if n < m { return 0.0; } // Insufficient data

    // 1. Create partitions (ordinal patterns)
    // Map each m-dim vector to a permutation index
    let num_patterns = factorial(m);
    let mut counts = vec![0; num_patterns];
    let mut total_patterns = 0;

    for i in 0..=(n - m * delay) {
        let mut pattern = Vec::with_capacity(m);
        for j in 0..m {
            pattern.push((data[i + j * delay], j));
        }
        // Sort by value to get permutation
        // If values equal, maintain original index order (stable sort)
        pattern.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        
        // Extract indices to form the permutation signature
        let perm: Vec<usize> = pattern.iter().map(|&(_, idx)| idx).collect();
        
        // Convert permutation to unique integer index (Lehmer code or similar)
        // For small m (<=5), straightforward mapping is fine.
        // For efficiency/simplicity here, we map simple permutations to index.
        let pattern_idx = permutation_to_index(&perm, m);
        
        if pattern_idx < num_patterns {
            counts[pattern_idx] += 1;
            total_patterns += 1;
        }
    }

    if total_patterns == 0 { return 0.0; }

    // 2. Calculate Shannon Entropy of the distribution of patterns
    let mut entropy = 0.0;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / total_patterns as f64;
            entropy -= p * p.ln();
        }
    }

    // Normalized Permutation Entropy (0 to 1)
    // H_norm = H / ln(m!)
    let max_entropy = (num_patterns as f64).ln();
    if max_entropy == 0.0 { return 0.0; }
    
    entropy / max_entropy
}

/// Calculate Multiscale Permutation Entropy (MPE / RCMWPE Lite)
/// Returns a vector of entropy values for scales 1 to max_scale.
/// Coarse-graining: Average of non-overlapping windows.
pub fn calculate_multiscale_entropy(data: &[f64], max_scale: usize, m: usize) -> Vec<f64> {
    let mut entropy_profile = Vec::with_capacity(max_scale);
    
    for scale in 1..=max_scale {
        // Coarse Graining
        let coarse_grained_len = data.len() / scale;
        if coarse_grained_len < m { 
            entropy_profile.push(0.0); // Not enough data for this scale
            continue;
        }

        let mut coarse_data = Vec::with_capacity(coarse_grained_len);
        for i in 0..coarse_grained_len {
            let start = i * scale;
            let end = start + scale;
            let sum: f64 = data[start..end].iter().sum();
            coarse_data.push(sum / scale as f64);
        }

        // Calculate PE for this scale
        let pe = calculate_permutation_entropy(&coarse_data, m, 1);
        entropy_profile.push(pe);
    }
    
    entropy_profile
}

// Helper: Factorial for m!
fn factorial(n: usize) -> usize {
    (1..=n).product()
}

// Helper: Map permutation vector to unique index [0, m!-1]
// Uses Lehmer code logic manually for small m
fn permutation_to_index(perm: &[usize], n: usize) -> usize {
    let mut index = 0;
    // O(n^2) approach is fine for small n (3-7)
    for i in 0..n {
        let mut smaller = 0;
        for j in (i + 1)..n {
            if perm[j] < perm[i] {
                smaller += 1;
            }
        }
        index += smaller * factorial(n - 1 - i);
    }
    index
}
