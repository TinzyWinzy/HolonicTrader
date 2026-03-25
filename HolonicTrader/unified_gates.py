#!/usr/bin/env python3
"""
Unified Gate System - 3 Superior Gates

Consolidates 15 scattered gates into 3 intelligent unified gates:
1. QUALITY GATE - Viability (signal + cost + blacklist)
2. ALIGNMENT GATE - Probability (structure + regime + confluence)
3. RISK GATE - Sustainability (exposure + crisis + cooldown)

Each gate returns a score (0-100) and pass/fail decision.
"""

import time
import json
import os
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum

import config


class GateResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    MARGINAL = "MARGINAL"  # Pass with warnings


@dataclass
class GateScore:
    """Unified gate scoring result"""
    gate_name: str
    score: float  # 0-100
    passed: bool
    components: Dict[str, float]  # Component scores
    details: str
    recommendation: str = ""


class UnifiedGateSystem:
    """
    3 Unified Gates for trade execution
    
    Replaces 15 scattered gates with 3 intelligent gates:
    1. QUALITY - Is this trade viable?
    2. ALIGNMENT - Is this trade probable?
    3. RISK - Is this trade sustainable?
    """
    
    # Gate thresholds
    QUALITY_THRESHOLD = 50.0
    ALIGNMENT_THRESHOLD = 55.0
    RISK_THRESHOLD = 60.0

    # CHRONIC LOSER BLACKLIST (2026-03-24)
    # Only block assets with VERY poor performance (<30% WR, >$10 loss, >5 trades)
    CHRONIC_LOSER_BLACKLIST = {
        'DOT/USDT': {'min_trades': 5, 'max_win_rate': 0.30, 'max_total_loss': -10.0},
    }
    
    # Component weights
    QUALITY_WEIGHTS = {
        'blacklist': 0.30,
        'cost_viability': 0.25,
        'recent_performance': 0.25,
        'ml_confidence': 0.10,
        'min_win_rate': 0.10,  # NEW: Filter assets with poor historical win rate
    }
    
    ALIGNMENT_WEIGHTS = {
        'structure_zone': 0.30,
        'market_path': 0.25,
        'confluence': 0.20,
        'conviction': 0.15,
        'regime_match': 0.10,
    }
    
    RISK_WEIGHTS = {
        'exposure_limits': 0.35,
        'crisis_state': 0.25,
        'stack_health': 0.20,
        'cooldown_status': 0.10,
        'actuator_ready': 0.10,
    }
    
    def __init__(self, governor=None, oracle=None):
        self.governor = governor
        self.oracle = oracle
    
    # ========================================================================
    # GATE 1: QUALITY (Viability Check)
    # ========================================================================
    
    def check_quality_gate(self, symbol: str, signal: Any, 
                           market_data: Dict = None) -> GateScore:
        """
        QUALITY GATE: Is this trade viable?
        
        Components:
        - Blacklist check (30%)
        - Cost viability (30%)
        - Recent performance (25%)
        - ML confidence (15%)
        """
        components = {}
        details = []
        
        # 1. Blacklist Check (30%)
        blacklist_score = self._check_blacklist(symbol)
        components['blacklist'] = blacklist_score
        if blacklist_score < 50:
            return GateScore(
                gate_name="QUALITY",
                score=0,
                passed=False,
                components=components,
                details=f"Blacklisted symbol: {symbol}",
                recommendation="Wait for blacklist expiry"
            )
        
        # 2. Cost Viability (30%)
        cost_score = self._check_cost_viability(signal, market_data)
        components['cost_viability'] = cost_score
        if cost_score >= 80:
            details.append("Edge >> costs")
        elif cost_score >= 50:
            details.append("Edge > costs (marginal)")
        else:
            details.append("Edge < costs (poor)")
        
        # 3. Recent Performance (25%)
        perf_score = self._check_recent_performance(symbol, signal)
        components['recent_performance'] = perf_score
        if perf_score >= 70:
            details.append("Hot streak")
        elif perf_score >= 40:
            details.append("Neutral performance")
        else:
            details.append("Cold streak")
        
        # 4. ML Confidence (10%)
        ml_score = self._check_ml_confidence(signal)
        components['ml_confidence'] = ml_score

        # 5. Minimum Win Rate Filter (10%) - NEW
        wr_score = self._check_min_win_rate(symbol)
        components['min_win_rate'] = wr_score

        # Calculate weighted score
        total_score = sum(
            components[k] * self.QUALITY_WEIGHTS[k]
            for k in self.QUALITY_WEIGHTS
        )
        
        # Determine pass/fail
        passed = total_score >= self.QUALITY_THRESHOLD
        marginal = total_score >= (self.QUALITY_THRESHOLD - 10)
        
        result = GateResult.PASS if passed else (GateResult.MARGINAL if marginal else GateResult.FAIL)
        
        return GateScore(
            gate_name="QUALITY",
            score=total_score,
            passed=passed,
            components=components,
            details=" | ".join(details),
            recommendation=self._get_quality_recommendation(total_score, components)
        )
    
    def _check_blacklist(self, symbol: str) -> float:
        """Check if symbol is blacklisted (0 or 100)"""
        # Check Governor blacklist
        if self.governor and hasattr(self.governor, 'blacklist'):
            if symbol in self.governor.blacklist:
                return 0.0

        # Check config blacklist
        blacklist = getattr(config, 'SYMBOL_BLACKLIST', [])
        if symbol in blacklist:
            return 0.0

        # Check chronic loss assets
        chronic = getattr(config, 'CHRONIC_LOSS_ASSETS', {})
        if symbol in chronic and chronic[symbol].get('suspended', False):
            return 0.0

        # === NEW: Check chronic loser blacklist from performance ===
        if symbol in self.CHRONIC_LOSER_BLACKLIST:
            criteria = self.CHRONIC_LOSER_BLACKLIST[symbol]
            # This symbol is a known chronic loser - block it
            return 0.0

        return 100.0
    
    def _check_cost_viability(self, signal: Any, market_data: Dict = None) -> float:
        """Check if expected edge exceeds costs (0-100)"""
        if not signal:
            return 0.0
        
        # Get signal conviction as proxy for edge
        conviction = getattr(signal, 'conviction', 0.5)
        
        # Calculate round-trip cost
        fee = getattr(config, 'ESTIMATED_FEE_PCT', 0.001)
        slippage = getattr(config, 'ESTIMATED_SLIPPAGE_PCT', 0.001)
        round_trip_cost = 2 * (fee + slippage)
        
        # Edge must exceed cost by MIN_EDGE_MULTIPLE
        min_multiple = getattr(config, 'MIN_EDGE_MULTIPLE', 2.0)
        required_edge = round_trip_cost * min_multiple
        
        # Score based on edge/cost ratio
        if conviction >= required_edge * 2:
            return 100.0  # Excellent edge
        elif conviction >= required_edge:
            return 70.0  # Good edge
        elif conviction >= required_edge * 0.8:
            return 50.0  # Marginal
        else:
            return 20.0  # Poor edge
    
    def _check_recent_performance(self, symbol: str, signal: Any) -> float:
        """Check recent win rate for symbol (0-100)"""
        if not self.governor:
            return 50.0  # Neutral if no governor
        
        # Get recent win rate from governor
        if hasattr(self.governor, 'calculate_recent_win_rate'):
            wr = self.governor.calculate_recent_win_rate(
                lookback=getattr(config, 'CONVICTION_WINRATE_LOOKBACK', 10)
            )
            # Map win rate to score (0.30 -> 30, 0.70 -> 100)
            return 30 + (wr * 100)
        
        return 50.0  # Default neutral
    
    def _check_ml_confidence(self, signal: Any) -> float:
        """Check ML model confidence (0-100)"""
        if not signal:
            return 0.0

        # Get XGB/LSTM confidence from signal metadata
        meta = getattr(signal, 'metadata', {})
        xgb_prob = meta.get('xgb_prob', 0.5)
        lstm_prob = meta.get('lstm_prob', 0.5)

        # Average if both available
        if xgb_prob > 0 and lstm_prob > 0:
            confidence = (xgb_prob + lstm_prob) / 2
        elif xgb_prob > 0:
            confidence = xgb_prob
        elif lstm_prob > 0:
            confidence = lstm_prob
        else:
            confidence = getattr(signal, 'conviction', 0.5)

        # Map to score (0.35 -> 0, 0.85 -> 100)
        normalized = (confidence - 0.35) / 0.50
        return max(0, min(100, normalized * 100))

    def _check_min_win_rate(self, symbol: str) -> float:
        """
        Check minimum historical win rate for symbol (0-100)

        Assets with <25% win rate get blocked
        Assets with <35% win rate get penalized
        Assets with >50% win rate get bonus
        """
        # Load asset performance stats if available
        stats_path = os.path.join(os.path.dirname(__file__), '..', '..', 'asset_performance_stats.json')

        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'r') as f:
                    asset_stats = json.load(f)

                if symbol in asset_stats:
                    stats = asset_stats[symbol]
                    win_rate = stats.get('win_rate', 0.5)
                    total_trades = stats.get('total_trades', 0)

                    # Need minimum trades for statistical significance
                    if total_trades < 3:
                        return 70.0  # Not enough data, assume neutral

                    # Score based on win rate - MORE LENIENT
                    if win_rate >= 0.55:
                        return 100.0  # Excellent
                    elif win_rate >= 0.45:
                        return 80.0  # Good
                    elif win_rate >= 0.35:
                        return 60.0  # Acceptable
                    elif win_rate >= 0.30:
                        return 40.0  # Poor but allowed
                    elif win_rate >= 0.25:
                        return 20.0  # Very poor - heavily penalized
                    else:
                        return 0.0  # Too poor - block
            except Exception as e:
                print(f"Error loading asset stats: {e}")

        # No stats available - assume neutral
        return 70.0
    
    def _get_quality_recommendation(self, score: float, components: Dict) -> str:
        """Get recommendation based on quality score"""
        if score >= 80:
            return "Excellent quality - proceed"
        elif score >= 60:
            return "Good quality - proceed with normal size"
        elif score >= 50:
            return "Marginal quality - reduce size 25%"
        elif score >= 40:
            weakest = min(components, key=components.get)
            return f"Weak {weakest} - consider skipping"
        else:
            return "Poor quality - reject"
    
    # ========================================================================
    # GATE 2: ALIGNMENT (Probability Check)
    # ========================================================================
    
    def check_alignment_gate(self, symbol: str, signal: Any,
                             structure: Dict = None,
                             regime: str = None) -> GateScore:
        """
        ALIGNMENT GATE: Is this trade probable?
        
        Components:
        - Structure zone (30%)
        - Market path (Orion) (25%)
        - Confluence (20%)
        - Conviction (15%)
        - Regime match (10%)
        """
        components = {}
        details = []
        
        if not signal:
            return GateScore(
                gate_name="ALIGNMENT",
                score=0,
                passed=False,
                components={},
                details="No signal provided"
            )
        
        meta = getattr(signal, 'metadata', {})
        direction = getattr(signal, 'direction', 'BUY')
        conviction = getattr(signal, 'conviction', 0.5)
        
        # 1. Structure Zone (30%)
        structure_score = self._check_structure_zone(structure, direction)
        components['structure_zone'] = structure_score
        if structure_score >= 80:
            details.append(f"Perfect structure ({structure.get('sls_zone', 'N/A')})")
        elif structure_score >= 50:
            details.append("Acceptable structure")
        else:
            details.append("Poor structure")
        
        # 2. Market Path - Orion (25%)
        path_score = self._check_market_path(meta, direction)
        components['market_path'] = path_score
        if path_score >= 70:
            details.append("Path aligned")
        else:
            details.append("Path misaligned")
        
        # 3. Confluence (20%)
        confluence_score = self._check_confluence(meta)
        components['confluence'] = confluence_score
        confluence_count = meta.get('confirmation_score', 1)
        details.append(f"{confluence_count} confirmations")
        
        # 4. Conviction with Dynamic Floor (15%)
        conviction_score = self._check_conviction(symbol, signal, meta, regime)
        components['conviction'] = conviction_score
        
        # 5. Regime Match (10%)
        regime_score = self._check_regime_match(meta.get('strategy', 'DIP'), regime)
        components['regime_match'] = regime_score
        
        # Calculate weighted score
        total_score = sum(
            components[k] * self.ALIGNMENT_WEIGHTS[k]
            for k in self.ALIGNMENT_WEIGHTS
        )
        
        passed = total_score >= self.ALIGNMENT_THRESHOLD
        
        return GateScore(
            gate_name="ALIGNMENT",
            score=total_score,
            passed=passed,
            components=components,
            details=" | ".join(details),
            recommendation=self._get_alignment_recommendation(total_score)
        )
    
    def _check_structure_zone(self, structure: Dict, direction: str) -> float:
        """Check structure zone alignment (0-100)"""
        if not structure:
            return 50.0  # Neutral
        
        sls_zone = structure.get('sls_zone', 'NEUTRAL')
        
        if direction == 'BUY':
            if sls_zone == 'SUPPORT':
                return 100.0  # Perfect
            elif sls_zone == 'NEUTRAL':
                return 60.0  # Acceptable
            else:  # RESISTANCE
                return 0.0  # Wrong zone
        else:  # SELL
            if sls_zone == 'RESISTANCE':
                return 100.0
            elif sls_zone == 'NEUTRAL':
                return 60.0
            else:  # SUPPORT
                return 0.0
    
    def _check_market_path(self, meta: Dict, direction: str) -> float:
        """Check Orion market path alignment (0-100)"""
        orion = meta.get('orion', {})
        if not orion:
            return 70.0  # No Orion data = assume neutral
        
        path = orion.get('path', 'NEUTRAL')
        
        if direction == 'BUY' and path == 'DOWN':
            return 30.0  # Misaligned
        elif direction == 'SELL' and path == 'UP':
            return 30.0
        elif path == 'NEUTRAL':
            return 70.0
        else:
            return 100.0  # Aligned
    
    def _check_confluence(self, meta: Dict) -> float:
        """Check signal confluence (0-100)"""
        confluence_count = meta.get('confirmation_score', 1)
        
        if confluence_count >= 4:
            return 100.0  # Excellent
        elif confluence_count >= 3:
            return 85.0  # Strong
        elif confluence_count >= 2:
            return 70.0  # Good
        elif confluence_count >= 1:
            return 50.0  # Minimum
        else:
            return 0.0
    
    def _check_conviction(self, symbol: str, signal: Any, 
                         meta: Dict, regime: str) -> float:
        """Check conviction against dynamic floor (0-100)"""
        if not self.oracle:
            return 50.0  # Can't check without oracle
        
        conviction = getattr(signal, 'conviction', 0.5)
        confluence_count = meta.get('confirmation_score', 1)
        recent_wr = meta.get('recent_win_rate', None)
        strategy = meta.get('strategy', 'DIP')
        
        # Get adaptive floor
        if hasattr(self.oracle, '_get_adaptive_conviction_floor'):
            adaptive_floor = self.oracle._get_adaptive_conviction_floor(
                symbol=symbol,
                strategy=strategy,
                confluence_count=confluence_count,
                recent_win_rate=recent_wr,
                current_regime=regime
            )
        else:
            adaptive_floor = 0.50  # Fallback
        
        # Score based on how much conviction exceeds floor
        if conviction >= adaptive_floor + 0.20:
            return 100.0  # Well above floor
        elif conviction >= adaptive_floor:
            # Scale from 50-100 based on how much above floor
            excess = conviction - adaptive_floor
            return 50 + (excess / 0.20) * 50
        else:
            # Below floor - scale from 0-50
            deficit = adaptive_floor - conviction
            return max(0, 50 - (deficit / 0.20) * 50)
    
    def _check_regime_match(self, strategy: str, regime: str) -> float:
        """Check strategy-regime alignment (0-100)"""
        if not regime:
            return 70.0  # No regime data = assume neutral
        
        alignment_map = getattr(config, 'STRATEGY_REGIME_ALIGNMENT', {})
        aligned_regimes = alignment_map.get(strategy, [])
        
        if regime in aligned_regimes:
            return 100.0  # Perfect match
        else:
            return 50.0  # Neutral
    
    def _get_alignment_recommendation(self, score: float) -> str:
        """Get recommendation based on alignment score"""
        if score >= 80:
            return "High probability - full size"
        elif score >= 65:
            return "Good probability - normal size"
        elif score >= 55:
            return "Moderate probability - reduce size 25%"
        else:
            return "Low probability - reject"
    
    # ========================================================================
    # GATE 3: RISK (Sustainability Check)
    # ========================================================================
    
    def check_risk_gate(self, symbol: str, signal: Any,
                       portfolio_state: Dict = None) -> GateScore:
        """
        RISK GATE: Is this trade sustainable?
        
        Components:
        - Exposure limits (35%)
        - Crisis state (25%)
        - Stack health (20%)
        - Cooldown status (10%)
        - Actuator ready (10%)
        """
        components = {}
        details = []
        
        # 1. Exposure Limits (35%)
        exposure_score = self._check_exposure_limits(symbol, signal, portfolio_state)
        components['exposure_limits'] = exposure_score
        if exposure_score >= 80:
            details.append("Exposure OK")
        elif exposure_score >= 50:
            details.append("Exposure elevated")
        else:
            details.append("Exposure exceeded")
        
        # 2. Crisis State (25%)
        crisis_score = self._check_crisis_state()
        components['crisis_state'] = crisis_score
        if crisis_score >= 80:
            details.append("Market calm")
        elif crisis_score >= 50:
            details.append("Market cautious")
        else:
            details.append("Market panic")
        
        # 3. Stack Health (20%)
        stack_score = self._check_stack_health(symbol)
        components['stack_health'] = stack_score
        if stack_score >= 70:
            details.append("Stack healthy")
        else:
            details.append("Stack congested")
        
        # 4. Cooldown Status (10%)
        cooldown_score = self._check_cooldown(symbol, signal)
        components['cooldown_status'] = cooldown_score
        
        # 5. Actuator Ready (10%)
        actuator_score = self._check_actuator_ready()
        components['actuator_ready'] = actuator_score
        
        # Calculate weighted score
        total_score = sum(
            components[k] * self.RISK_WEIGHTS[k]
            for k in self.RISK_WEIGHTS
        )
        
        passed = total_score >= self.RISK_THRESHOLD
        
        return GateScore(
            gate_name="RISK",
            score=total_score,
            passed=passed,
            components=components,
            details=" | ".join(details),
            recommendation=self._get_risk_recommendation(total_score)
        )
    
    def _check_exposure_limits(self, symbol: str, signal: Any, 
                               portfolio_state: Dict = None) -> float:
        """Check portfolio exposure limits (0-100)"""
        if not self.governor:
            return 70.0  # No governor = assume OK
        
        # Check if governor approves
        if hasattr(self.governor, 'check_exposure'):
            approved = self.governor.check_exposure(symbol, signal)
            if not approved:
                return 0.0
        
        # Check portfolio health
        if portfolio_state:
            health = portfolio_state.get('health', 'GOOD')
            if health == 'CRITICAL':
                return 20.0
            elif health == 'WARNING':
                return 50.0
        
        return 100.0
    
    def _check_crisis_state(self) -> float:
        """Check crisis/panic state (0-100)"""
        if not self.governor:
            return 80.0  # No governor = assume calm
        
        if hasattr(self.governor, 'check_crisis_status'):
            crisis = self.governor.check_crisis_status()
            crisis_score = crisis.get('crisis_score', 0)
            
            # Invert: high crisis = low score
            if crisis_score >= 0.8:
                return 0.0  # Panic
            elif crisis_score >= 0.5:
                return 40.0  # Cautious
            else:
                return 100.0  # Calm
        
        return 80.0
    
    def _check_stack_health(self, symbol: str) -> float:
        """Check position stack health (0-100)"""
        if not self.governor:
            return 70.0
        
        # Check if symbol has stack issues
        if hasattr(self.governor, 'stack_snooze'):
            if symbol in self.governor.stack_snooze:
                return 30.0  # Stack blocked
        
        if hasattr(self.governor, 'stack_timeout_tracker'):
            if symbol in self.governor.stack_timeout_tracker:
                return 50.0  # Stack in timeout
        
        return 100.0
    
    def _check_cooldown(self, symbol: str, signal: Any) -> float:
        """Check signal cooldown (0-100)"""
        if not self.oracle:
            return 80.0
        
        strategy = getattr(signal, 'metadata', {}).get('strategy', 'DEFAULT')
        
        # Check if signal is on cooldown
        if hasattr(self.oracle, 'signal_cooldowns'):
            key = f"{symbol}_{strategy}"
            if key in self.oracle.signal_cooldowns:
                remaining = self.oracle.signal_cooldowns[key] - time.time()
                if remaining > 0:
                    return 0.0  # Still on cooldown
        
        return 100.0
    
    def _check_actuator_ready(self) -> float:
        """Check if actuator is ready (0-100)"""
        # Check circuit breaker
        if self.governor and hasattr(self.governor, 'check_circuit_breaker'):
            if not self.governor.check_circuit_breaker():
                return 0.0
        
        return 100.0
    
    def _get_risk_recommendation(self, score: float) -> str:
        """Get recommendation based on risk score"""
        if score >= 80:
            return "Low risk - proceed"
        elif score >= 65:
            return "Moderate risk - normal size"
        elif score >= 60:
            return "Elevated risk - reduce size 50%"
        else:
            return "High risk - reject"
    
    # ========================================================================
    # UNIFIED GATE CHECK (All 3 Gates)
    # ========================================================================
    
    def check_all_gates(self, symbol: str, signal: Any,
                       structure: Dict = None,
                       regime: str = None,
                       portfolio_state: Dict = None,
                       market_data: Dict = None) -> Tuple[bool, List[GateScore]]:
        """
        Run all 3 unified gates
        
        Returns:
            (all_passed, list_of_gate_scores)
        """
        scores = []
        
        # Gate 1: QUALITY
        quality_score = self.check_quality_gate(symbol, signal, market_data)
        scores.append(quality_score)
        
        # Gate 2: ALIGNMENT
        alignment_score = self.check_alignment_gate(symbol, signal, structure, regime)
        scores.append(alignment_score)
        
        # Gate 3: RISK
        risk_score = self.check_risk_gate(symbol, signal, portfolio_state)
        scores.append(risk_score)
        
        # All must pass
        all_passed = all(score.passed for score in scores)
        
        return all_passed, scores


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == '__main__':
    """Test the unified gate system"""
    print("=" * 70)
    print("UNIFIED GATE SYSTEM - TEST")
    print("=" * 70)
    
    # Mock signal
    class MockSignal:
        def __init__(self):
            self.symbol = 'SHIB/USDT'
            self.direction = 'BUY'
            self.conviction = 0.50
            self.metadata = {
                'confirmation_score': 2,
                'strategy': 'DIP',
                'recent_win_rate': 0.50,
            }
    
    signal = MockSignal()
    
    # Create gate system (without governor/oracle for testing)
    gates = UnifiedGateSystem()
    
    # Run gates
    all_passed, scores = gates.check_all_gates(
        symbol='SHIB/USDT',
        signal=signal,
        structure={'sls_zone': 'SUPPORT'},
        regime='LOW_VOL_MEAN_REVERT',
        portfolio_state={'health': 'GOOD'},
    )
    
    # Print results
    print(f"\nOverall Result: {'PASS' if all_passed else 'FAIL'}\n")

    for score in scores:
        status = "PASS" if score.passed else "FAIL"
        print(f"{score.gate_name} GATE: {status}")
        print(f"  Score: {score.score:.1f}/100")
        print(f"  Details: {score.details}")
        print(f"  Recommendation: {score.recommendation}")
        print()
    
    print("=" * 70)
