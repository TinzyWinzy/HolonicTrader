import os
import yaml
import pandas as pd
import logging
from typing import List, Dict, Any
from core.mathematics.entropy import compute_entropy_profile

# Configure logging
logger = logging.getLogger('EntropyScouter')

class EntropyScouter:
    """
    The Scouting Holon.
    Scans the market universe using Physics-based metrics (Entropy) to classify regimes.
    """
    
    def __init__(self, context_path: str = "context.yaml"):
        self.context = self._load_context(context_path)
        # Handle New Config Structure
        self.scout_config = self.context.get('entropy_scouter', {})
        self.metrics_config = self.scout_config.get('metrics', {}).get('sample_entropy', {})
        self.regime_config = self.scout_config.get('regimes', {})
        
        # Fallback if empty (or old config)
        if not self.metrics_config:
             self.thresholds = self.context.get('entropy', {}).get('sample_en', {})
        else:
             self.thresholds = {'r_sigma': self.metrics_config.get('r_sigma', 0.2)} # Map r

        
    def _load_context(self, path: str) -> dict:
        """Load the YAML context configuration."""
        # Try finding file in CWD or relative to this file
        target_path = path
        if not os.path.exists(target_path):
            # Try 2 levels up (if running from core/scouts)
            target_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), path)
        
        if os.path.exists(target_path):
            try:
                with open(target_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Failed to load context.yaml: {e}")
                return {}
        else:
            logger.warning(f"context.yaml not found at {path} or {target_path}. Using defaults.")
            return {}

    def scout_regimes(self, data_feed: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Analyze a batch of market data and return regime classifications.
        
        FIX 2026-03-03: Adjusted thresholds to reduce TRANSITION dominance.
        Live data shows most assets cluster in 0.8-1.2 range, so we narrow TRANSITION.
        """
        results = {}

        # Load Thresholds from New Config
        # FIX 2026-03-03: Narrowed TRANSITION zone to reduce dominance
        # Low Entropy (Ordered) < 0.85 (was 0.6) - more assets qualify as ORDERED
        # High Entropy (Chaos) > 1.3 (was 1.51) - easier to trigger CHAOTIC
        # TRANSITION: 0.85 - 1.3 (narrower band)

        low_ent_cfg = self.regime_config.get('low_entropy', {})
        med_ent_cfg = self.regime_config.get('medium_entropy', {})
        high_ent_cfg = self.regime_config.get('high_entropy', {})

        # 2026-03-19: Empirically calibrated from 15-asset live snapshot.
        # SampleEntropy distribution: P50=0.72, P90=1.10, max=1.16
        # Targets ~40% ORDERED / ~40% TRANSITION / ~20% CHAOTIC split.
        ordered_max = float(low_ent_cfg.get('threshold_max', 0.70))   # ≤ P50 = structured/trending
        chaos_min   = float(high_ent_cfg.get('threshold_min', 1.10))  # ≥ P90 = high complexity

        for symbol, df in data_feed.items():
            if df is None or df.empty or len(df) < 50:
                results[symbol] = {'regime': 'UNKNOWN', 'entropy': 0.0}
                continue

            # Extract Close prices
            closes = df['close'].tolist()

            # Compute Physics (Entropy)
            # Use 'm' and 'r' from config if possible
            r_sigma = float(self.metrics_config.get('r_sigma', 0.2))

            # Recalculate if needed, or rely on helper defaults?
            # Helper uses 0.2*std.
            profile = compute_entropy_profile(closes) # Helper uses defaults
            samp_en = profile['sample_entropy']

            # Classify Regime (Legacy Map)
            # low_entropy -> ORDERED
            # medium_entropy -> TRANSITION
            # high_entropy -> CHAOTIC

            regime = 'TRANSITION'
            if samp_en <= ordered_max:
                regime = 'ORDERED'
            elif samp_en >= chaos_min:
                regime = 'CHAOTIC'

            results[symbol] = {
                'regime': regime,
                'entropy': samp_en,
                'perm_entropy': profile['perm_entropy'],
                'meta': {
                    'action': med_ent_cfg.get('action', 'standard') if regime == 'TRANSITION' else
                             (low_ent_cfg.get('action', 'trend') if regime == 'ORDERED' else high_ent_cfg.get('action', 'halt'))
                }
            }

        return results

    def filter_whitelist(self, candidates: List[str], scout_results: Dict[str, Any]) -> List[str]:
        """
        Filter a list of candidates, removing those in explicitly dangerous regimes (e.g. CHAOTIC).
        
        Args:
            candidates: List of symbols
            scout_results: Output from scout_regimes
            
        Returns:
            Filtered list of symbols
        """
        approved = []
        for sym in candidates:
            res = scout_results.get(sym)
            if not res:
                # If no data, keep it? Or strict safety?
                # Keep it, maybe we just didn't fetch data yet.
                approved.append(sym)
                continue
                
            regime = res['regime']
            entropy = res['entropy']
            
            # LOGIC: 
            # - ORDERED: Always Approve
            # - TRANSITION: Approve (Cautious)
            # - CHAOTIC: REJECT (Unless explicitly configured to trade chaos)
            
            if regime == 'CHAOTIC':
                # OPTIONAL: Check if we have a Chaos Strategy
                # For now, strict filter for safety.
                # logger.info(f"🛑 VETO {sym}: High Entropy ({entropy:.2f}) -> CHAOTIC")
                continue
                
            approved.append(sym)
            
        return approved
