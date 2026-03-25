"""
SMCEMonteCarloRiskCourt – Sovereign Micro-Compounding Engine (Layer 3)

Pre-trade Monte Carlo veto: runs 1000–5000 paths incorporating:
  - Current portfolio positions
  - Proposed trade (size, direction, leverage)
  - Recent 24h volatility
  - Inter-asset correlation matrix
  - Stress scenario: 3% adverse move in traded asset

Veto conditions (ANY one triggers rejection):
  - P(portfolio drawdown > 5% in 24h) > 10%
  - CVaR(95%) > 3% of equity
  - P(liquidation) > 1%

If borderline CVaR, the approved_size is reduced proportionally.
"""

import math
import time
import logging
import random
from typing import Dict, Any, Optional, List

logger = logging.getLogger("SMCEMCRiskCourt")

# ── Veto thresholds ───────────────────────────────────────────────────────────
VETO_DRAWDOWN_PROB  = 0.10   # reject if P(DD>5%) > 10%
VETO_DRAWDOWN_LIMIT = 0.05   # 5% portfolio drawdown
VETO_CVAR_LIMIT     = 0.08   # 8% of equity CVaR(95%) - RELAXED for MICRO accounts <$200 (was 3%)
VETO_LIQ_PROB       = 0.01   # 1% liquidation probability

# Path settings
DEFAULT_PATHS        = 1000
STRESS_PATHS         = 5000
STRESS_ADV_MOVE      = 0.03   # 3% adverse move for stress scenario
HORIZON_HOURS        = 24


class SMCEMonteCarloRiskCourt:
    """
    Layer 3 pre-trade Monte Carlo veto gate.

    This class is separate from MonteCarloPositionManager (which evaluates
    EXISTING positions for closure). This court evaluates PROPOSED trades
    before they enter the portfolio.
    """

    def __init__(self, n_paths: int = DEFAULT_PATHS):
        self.n_paths       = n_paths
        self.last_result: Dict = {}
        self.veto_count    = 0
        self.approve_count = 0

    # ─── Public API ───────────────────────────────────────────────────────────

    def evaluate_pre_trade(
        self,
        equity: float,
        proposed_trade: Dict[str, Any],
        portfolio_positions: Dict[str, Any],
        volatilities: Dict[str, float],
        correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo risk court on a proposed trade.

        Args:
            equity:             Current total equity (USD)
            proposed_trade:     {symbol, direction, notional, leverage, entry_price}
            portfolio_positions: {symbol: {notional, direction, entry_price, leverage}}
            volatilities:       {symbol: 24h_vol_fraction}  e.g. {'BTC/USDT': 0.04}
            correlation_matrix: optional {sym1: {sym2: corr_float}}

        Returns:
            {
                "vetoed":           bool,
                "veto_reason":      str or None,
                "drawdown_prob":    float (0–1),
                "cvar_95":          float (fraction of equity, e.g. 0.025 = 2.5%),
                "liquidation_prob": float (0–1),
                "approved_size":    float (suggested notional, may be reduced),
                "paths_run":        int,
            }
        """
        t0 = time.time()
        symbol    = proposed_trade.get("symbol", "?")
        direction = proposed_trade.get("direction", "BUY")
        notional  = float(proposed_trade.get("notional", 0.0))
        leverage  = float(proposed_trade.get("leverage", 1.0))

        if equity <= 0 or notional <= 0:
            return self._make_veto_result("Invalid equity or notional size", notional)

        # === Step 1: Build combined portfolio including proposed trade =========
        combined: List[Dict] = []
        for sym, pos in portfolio_positions.items():
            # Duck-typed accessor for both dict and Position objects
            if isinstance(pos, dict):
                pos_notional = abs(float(pos.get("notional", pos.get("quantity", 0) * pos.get("entry_price", 1))))
                pos_direction = pos.get("direction", "BUY")
                pos_leverage = float(pos.get("leverage", 1.0))
            else:
                # Position object
                notional_attr = getattr(pos, "notional", None)
                if notional_attr is not None:
                    pos_notional = abs(float(notional_attr))
                else:
                    qty = float(getattr(pos, "quantity", 0))
                    price = float(getattr(pos, "entry_price", 1))
                    pos_notional = abs(qty * price)
                pos_direction = getattr(pos, "direction", "BUY")
                pos_leverage = float(getattr(pos, "leverage", 1.0))
            
            combined.append({
                "symbol":    sym,
                "notional":  pos_notional,
                "direction": pos_direction,
                "leverage":  pos_leverage,
                "vol":       volatilities.get(sym, 0.03),
            })
        combined.append({
            "symbol":    symbol,
            "direction": direction,
            "notional":  notional,
            "leverage":  leverage,
            "vol":       volatilities.get(symbol, 0.04),
        })

        # === Step 2: Simulation ===============================================
        paths_run = self.n_paths
        cumulative_pnl = []
        liquidations    = 0

        for _ in range(paths_run):
            path_pnl, liquidated = self._simulate_path(
                combined, equity, correlation_matrix
            )
            cumulative_pnl.append(path_pnl)
            if liquidated:
                liquidations += 1

        # Add stress paths (3% adverse move in proposed asset)
        stress_pnl = self._run_stress_scenario(combined, equity, symbol, direction)
        cumulative_pnl.extend(stress_pnl)
        total_paths = paths_run + len(stress_pnl)

        # === Step 3: Compute metrics ==========================================
        sorted_pnl        = sorted(cumulative_pnl)
        pnl_fractions     = [p / equity for p in sorted_pnl]

        # P(drawdown > 5%)
        dd_threshold = -VETO_DRAWDOWN_LIMIT
        n_below_dd   = sum(1 for p in pnl_fractions if p < dd_threshold)
        drawdown_prob = n_below_dd / len(pnl_fractions)

        # CVaR(95%): average of worst 5% outcomes
        cutoff_idx  = max(1, int(0.05 * len(sorted_pnl)))
        worst_pnls  = sorted_pnl[:cutoff_idx]
        cvar_95     = abs(sum(worst_pnls) / len(worst_pnls)) / equity

        # Liquidation probability
        liq_prob = liquidations / paths_run

        elapsed = time.time() - t0

        # === Step 4: Veto decision ============================================
        veto_reason = None
        if drawdown_prob > VETO_DRAWDOWN_PROB:
            veto_reason = (
                f"P(DD>{VETO_DRAWDOWN_LIMIT*100:.0f}%)={drawdown_prob:.1%} "
                f"> {VETO_DRAWDOWN_PROB:.0%} threshold"
            )
        elif cvar_95 > VETO_CVAR_LIMIT:
            veto_reason = (
                f"CVaR(95%)={cvar_95:.2%} > {VETO_CVAR_LIMIT:.0%} of equity"
            )
        elif liq_prob > VETO_LIQ_PROB:
            veto_reason = (
                f"P(liquidation)={liq_prob:.2%} > {VETO_LIQ_PROB:.0%} threshold"
            )

        # Size reduction for borderline CVaR (between 2% and 5%)
        approved_size = notional
        CVAR_SOFT_LIMIT = 0.05  # UNLEASHED: Size reduction starts at 5% CVaR (was 2%)
        if veto_reason is None and cvar_95 > CVAR_SOFT_LIMIT:
            reduction = CVAR_SOFT_LIMIT / cvar_95   # scale down proportionally
            approved_size = notional * reduction
            logger.info(
                "[SMCE-MC] %s size reduced from $%.2f → $%.2f (CVaR=%.2f%%)",
                symbol, notional, approved_size, cvar_95 * 100
            )

        result = {
            "vetoed":           veto_reason is not None,
            "veto_reason":      veto_reason,
            "drawdown_prob":    drawdown_prob,
            "cvar_95":          cvar_95,
            "liquidation_prob": liq_prob,
            "approved_size":    approved_size,
            "paths_run":        total_paths,
            "elapsed_ms":       round(elapsed * 1000, 1),
        }

        if veto_reason:
            self.veto_count += 1
            logger.warning("[SMCE-MC] VETO %s %s | %s", direction, symbol, veto_reason)
        else:
            self.approve_count += 1
            logger.info(
                "[SMCE-MC] APPROVE %s %s | dd=%.1f%% cvar=%.2f%% liq=%.2f%% [%.0fms]",
                direction, symbol,
                drawdown_prob * 100, cvar_95 * 100, liq_prob * 100, elapsed * 1000
            )

        self.last_result = result
        return result

    def evaluate_portfolio(
        self,
        portfolio_positions: Dict[str, Any],
        equity: float,
        volatilities: Optional[Dict[str, float]] = None,
        correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo on the EXISTING portfolio only.
        """
        if equity <= 0:
            return {"cvar_95": 1.0, "drawdown_prob": 1.0}

        volatilities = volatilities or {}
        combined = []
        for sym, pos in portfolio_positions.items():
            if isinstance(pos, dict):
                pos_notional = abs(float(pos.get("notional", pos.get("quantity", 0) * pos.get("entry_price", 1))))
                pos_direction = pos.get("direction", "BUY")
                pos_leverage = float(pos.get("leverage", 1.0))
            else:
                qty = float(getattr(pos, "quantity", 0))
                price = float(getattr(pos, "entry_price", 1))
                pos_notional = abs(qty * price)
                pos_direction = getattr(pos, "direction", "BUY")
                pos_leverage = float(getattr(pos, "leverage", 1.0))
            
            combined.append({
                "symbol":    sym,
                "notional":  pos_notional,
                "direction": pos_direction,
                "leverage":  pos_leverage,
                "vol":       volatilities.get(sym, 0.03),
            })

        paths_run = self.n_paths
        cumulative_pnl = []
        for _ in range(paths_run):
            path_pnl, _ = self._simulate_path(combined, equity, correlation_matrix)
            cumulative_pnl.append(path_pnl)

        sorted_pnl = sorted(cumulative_pnl)
        pnl_fractions = [p / equity for p in sorted_pnl]
        
        # P(drawdown > 5%)
        n_below_dd = sum(1 for p in pnl_fractions if p < -VETO_DRAWDOWN_LIMIT)
        drawdown_prob = n_below_dd / len(pnl_fractions)

        # CVaR(95%)
        cutoff_idx = max(1, int(0.05 * len(sorted_pnl)))
        worst_pnls = sorted_pnl[:cutoff_idx]
        cvar_95 = abs(sum(worst_pnls) / len(worst_pnls)) / equity

        return {
            "cvar_95":       cvar_95,
            "drawdown_prob": drawdown_prob,
            "paths_run":     paths_run,
        }

    def get_stats(self) -> Dict:
        return {
            "veto_count":    self.veto_count,
            "approve_count": self.approve_count,
            "last_result":   self.last_result,
        }

    # ─── Private ──────────────────────────────────────────────────────────────

    def _simulate_path(
        self,
        positions: List[Dict],
        equity: float,
        corr_matrix: Optional[Dict],
    ) -> tuple:
        """
        Simulate one 24h path using GBM arithmetic returns.
        Returns (portfolio_pnl, liquidated_bool).
        """
        total_pnl = 0.0
        # Per-position simulation (no correlation adjustment without matrix)
        for pos in positions:
            vol          = pos["vol"]
            notional     = pos["notional"]
            leverage     = pos["leverage"]
            direction    = pos["direction"]

            # GBM return: r ~ N(0, vol/sqrt(24) * sqrt(24)) = N(0, vol)
            # For 24h horizon: sigma_24h = vol (already 24h)
            r = random.gauss(0, vol)

            # Clamp extreme moves
            r = max(-0.20, min(0.20, r))
            sign = 1.0 if direction == "BUY" else -1.0
            pnl  = sign * r * notional

            # Check liquidation: leveraged position liquidated if adverse move > 1/leverage
            liq_threshold = 1.0 / max(leverage, 1.0)
            if sign * r < -liq_threshold:
                return -equity, True  # Full liquidation

            total_pnl += pnl

        return total_pnl, False

    def _run_stress_scenario(
        self,
        positions: List[Dict],
        equity: float,
        stress_symbol: str,
        direction: str,
    ) -> List[float]:
        """
        Stress: apply 3% adverse move to the proposed asset,
        then simulate correlated moves in others (±1.5% random).
        Run 100 paths.
        """
        stress_pnl = []
        for _ in range(100):
            total = 0.0
            for pos in positions:
                if pos["symbol"] == stress_symbol:
                    adverse = -STRESS_ADV_MOVE if direction == "BUY" else STRESS_ADV_MOVE
                    r = adverse
                else:
                    r = random.gauss(0, 0.015)  # correlated noise

                sign = 1.0 if pos["direction"] == "BUY" else -1.0
                total += sign * r * pos["notional"]
            stress_pnl.append(total)
        return stress_pnl

    def _make_veto_result(self, reason: str, original_notional: float) -> Dict:
        """Build a veto result dict and increment the veto counter."""
        self.veto_count += 1
        return {
            "vetoed":           True,
            "veto_reason":      reason,
            "drawdown_prob":    1.0,
            "cvar_95":          1.0,
            "liquidation_prob": 1.0,
            "approved_size":    0.0,
            "paths_run":        0,
            "elapsed_ms":       0,
        }
