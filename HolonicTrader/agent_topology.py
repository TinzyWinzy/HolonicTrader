"""
TopologyHolon - The "Structure" Brain (AEHML 2.0)

Specialized in Topological Data Analysis (TDA) to detect market crashes
before they happen by monitoring the collapse of high-dimensional structure.

Logic:
    Healthy Market = Complex Topology (High Persistent Entropy)
    Crash Precursor = Topology Collapse (Low Persistent Entropy)
"""

import pandas as pd
from typing import Any, Dict
try:
    import holonic_speed
except ImportError:
    holonic_speed = None
    print(">> [Topology] Warning: holonic_speed not found. Topology analysis disabled.")

from HolonicTrader.holon_core import Holon, Disposition
import config

class TopologyHolon(Holon):
    def __init__(self, name: str = "TopologyAgent"):
        super().__init__(name=name, disposition=Disposition(autonomy=0.8, integration=0.4))
        self.last_entropy = 0.0
        self.crash_warning = False
        self.embedding_dim = 3
        self.delay = 1
        self.window_size = 50

        # FIX 2026-03-15: Adaptive threshold with rolling baseline
        # Prevents constant "Structure Collapse" alerts when market is naturally low-entropy
        self.entropy_history = []  # Rolling history of entropy scores
        self.entropy_history_maxlen = 100  # Keep last 100 readings
        self.baseline_entropy = 0.5  # Dynamic baseline (updated from history)

        # COMPUTE WASTE FIX: Alert cooldown to prevent spam
        self._alert_cooldown = {}  # {alert_type: last_ts}
        self._alert_cooldown_ttl = 30  # 30 seconds between identical alerts
        self._suppress_count = 0  # Count suppressed alerts for periodic summary

        # FIX 2026-03-15: Hard floor to prevent spurious alerts at extreme low values
        # If entropy is exactly 0.0000, it's likely a calculation artifact, not real collapse
        self.entropy_floor = 0.05  # Below this = data quality issue, not market signal

    def analyze_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze topological structure of the market.
        Returns: {'status': 'STABLE'|'CRITICAL', 'score': float}
        """
        if holonic_speed is None:
             return {'status': 'STABLE', 'score': 0.5, 'crash_warning': False}

        if len(df) < self.window_size:
            return {'status': 'WAITING', 'score': 0.0}

        # Extract close prices
        # We need a list of floats
        prices = df['close'].values.tolist()

        # Calculate Persistent Entropy (TDA Score)
        # Low Score (< 0.1) = Structure Collapse = CRASH RISK
        tda_score = holonic_speed.calculate_persistent_entropy(
            prices,
            self.window_size,
            self.embedding_dim,
            self.delay
        )

        self.last_entropy = tda_score

        # FIX 2026-03-15: Update rolling entropy history and baseline
        self.entropy_history.append(tda_score)
        if len(self.entropy_history) > self.entropy_history_maxlen:
            self.entropy_history.pop(0)

        # Update dynamic baseline (median of recent history, more robust than mean)
        if len(self.entropy_history) >= 10:
            sorted_history = sorted(self.entropy_history)
            mid = len(sorted_history) // 2
            self.baseline_entropy = sorted_history[mid]

        # FIX 2026-03-15: Apply entropy floor - below this is data quality issue
        if tda_score < self.entropy_floor:
            # Data quality issue - likely calculation artifact
            # Don't trigger crash warning, just note the anomaly
            return {
                'status': 'DATA_QUALITY_LOW',
                'score': tda_score,
                'crash_warning': False,
                'data_quality_issue': True
            }

        # FIX 2026-03-15: Adaptive threshold based on rolling baseline
        # Crash = entropy drops significantly below recent normal (50% of baseline)
        # This adapts to different market regimes instead of fixed threshold
        adaptive_threshold = self.baseline_entropy * 0.50

        # Ensure minimum threshold floor (don't go below 0.15)
        adaptive_threshold = max(adaptive_threshold, 0.15)

        warning_threshold = getattr(config, 'TOPOLOGY_WARNING_THRESHOLD', adaptive_threshold)

        status = 'STABLE'
        if tda_score < warning_threshold:
            status = 'CRITICAL'
            self.crash_warning = True

            # COMPUTE WASTE FIX: Alert cooldown to prevent spam
            import time
            now = time.time()
            alert_key = f"topology_collapse_{warning_threshold}"
            last_alert = self._alert_cooldown.get(alert_key, 0)

            if (now - last_alert) >= self._alert_cooldown_ttl:
                # Log the alert and update cooldown
                self._alert_cooldown[alert_key] = now
                print(f"[{self.name}] 🚨 TOPOLOGY ALERT: Structure Collapse (Score {tda_score:.4f} < {warning_threshold:.4f}, Baseline: {self.baseline_entropy:.4f})")
                self._suppress_count = 0  # Reset suppress counter
            else:
                self._suppress_count += 1
                # Periodic summary every 20 suppressed alerts
                if self._suppress_count % 20 == 0:
                    print(f"[{self.name}] ⚠️ Topology alert suppressed ({self._suppress_count}x in cooldown)")
        else:
            self.crash_warning = False

        return {
            'status': status,
            'score': tda_score,
            'crash_warning': self.crash_warning
        }

    def get_health(self) -> dict:
        return {
            'status': 'ACTIVE',
            'last_score': f"{self.last_entropy:.4f}",
            'warning': self.crash_warning
        }
        
    def receive_message(self, sender: Any, content: Any) -> None:
        pass
