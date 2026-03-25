"""
CHRONOS MARKET FORENSICS v2 - Quantitative Trading Loss Auditor

Role: Forensic analyst for trading system failures
Experience: 30+ years across hedge funds, HFT desks, derivatives risk

Core Mission:
- Determine WHY losses occur
- Classify losses as structural or temporary
- Identify if strategy is being exploited or over-vetoed
- Detect execution infrastructure degradation
- Audit market assumption validity

Guiding Principle:
> "Profit is evidence. Loss is information."

v2 Changes:
- Fixed DB column names (pnl_percent, price instead of pnl_pct, entry_price)
- Fixed exit detection (pnl != 0 rather than cost_usd <= 1e-9)
- Added ChronosLogParser: live session log analysis
- Added veto attribution analysis
- Richer failure classification using real pnl_percent data
"""

import sqlite3
import re
import os
import glob
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import defaultdict
import json
import logging

logger = logging.getLogger("Chronos.Forensics")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TradeAutopsy:
    """Complete forensic analysis of a single trade."""
    trade_id: int
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    entry_timestamp: str
    exit_timestamp: str
    hold_duration_sec: float

    # Signal analysis
    entry_signal: str = ""
    entry_conviction: float = 0.0
    exit_reason: str = ""

    # Market context
    market_regime_at_entry: str = ""
    volatility_at_entry: float = 0.0
    spread_at_entry: float = 0.0

    # Execution quality
    slippage_bps: float = 0.0
    latency_ms: float = 0.0

    # Failure attribution
    primary_cause: str = ""
    secondary_cause: str = ""
    is_structural: bool = False
    is_exploitable: bool = False

    def to_dict(self) -> Dict:
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'entry_signal': self.entry_signal,
            'exit_reason': self.exit_reason,
            'primary_cause': self.primary_cause,
            'secondary_cause': self.secondary_cause,
            'is_structural': self.is_structural,
            'is_exploitable': self.is_exploitable
        }


@dataclass
class LossAttribution:
    """Breakdown of loss causes."""
    category: str  # EXECUTION, SIGNAL, RISK, REGIME, EXPLOITATION, VETO_OVERPROTECTION
    percentage: float  # % of total losses attributed
    total_loss_usd: float
    trade_count: int
    avg_loss_per_trade: float
    confidence: float  # Confidence in attribution (0.0-1.0)
    evidence: List[str] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> Dict:
        return {
            'category': self.category,
            'percentage': self.percentage,
            'total_loss_usd': self.total_loss_usd,
            'trade_count': self.trade_count,
            'avg_loss_per_trade': self.avg_loss_per_trade,
            'confidence': self.confidence,
            'evidence': self.evidence,
            'recommendation': self.recommendation
        }


@dataclass
class StrategyHealthScore:
    """Overall strategy health assessment."""
    timestamp: float
    overall_score: float        # 0.0-10.0
    signal_quality: float       # 0.0-10.0
    execution_quality: float    # 0.0-10.0
    risk_management: float      # 0.0-10.0
    market_compatibility: float # 0.0-10.0
    expectancy_status: str      # POSITIVE, NEGATIVE, NEUTRAL
    regime_alignment: str       # ALIGNED, MISALIGNED, UNKNOWN
    exploitation_risk: str      # LOW, MEDIUM, HIGH

    recommendations: List[str] = field(default_factory=list)
    critical_findings: List[str] = field(default_factory=list)

    # Expectancy raw values
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'overall_score': self.overall_score,
            'signal_quality': self.signal_quality,
            'execution_quality': self.execution_quality,
            'risk_management': self.risk_management,
            'market_compatibility': self.market_compatibility,
            'expectancy_status': self.expectancy_status,
            'regime_alignment': self.regime_alignment,
            'exploitation_risk': self.exploitation_risk,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'expectancy': self.expectancy,
            'recommendations': self.recommendations,
            'critical_findings': self.critical_findings
        }


@dataclass
class VetoEvent:
    """A single veto event from the log."""
    timestamp: str
    symbol: str
    veto_layer: str      # HOLONIC, STRUCTURE_BOSS, GOVERNOR, SMCE
    veto_reason: str
    signal_direction: str = ""
    conviction: float = 0.0


@dataclass
class SessionForensics:
    """Forensic analysis of a single trading session log."""
    log_path: str
    session_start: str
    session_end: str

    # Signal counts
    signals_generated: int = 0
    signals_blocked: int = 0
    signals_executed: int = 0

    # Veto breakdown
    veto_by_layer: Dict[str, int] = field(default_factory=dict)
    veto_by_reason: Dict[str, int] = field(default_factory=dict)
    veto_by_symbol: Dict[str, int] = field(default_factory=dict)

    # Management mode
    management_mode_events: int = 0
    management_mode_max_duration_sec: float = 0.0
    management_mode_reasons: Dict[str, int] = field(default_factory=dict)

    # Regime distribution
    regime_distribution: Dict[str, int] = field(default_factory=dict)

    # Crisis scores
    avg_crisis_score: float = 0.0
    max_crisis_score: float = 0.0

    # Scout promotions / demotions
    scout_promotions: int = 0
    scout_demotions: int = 0

    def pass_rate(self) -> float:
        """Fraction of generated signals that actually executed."""
        total = self.signals_generated
        return (self.signals_executed / total) if total > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            'log_path': self.log_path,
            'session_start': self.session_start,
            'session_end': self.session_end,
            'signals_generated': self.signals_generated,
            'signals_blocked': self.signals_blocked,
            'signals_executed': self.signals_executed,
            'pass_rate_pct': round(self.pass_rate() * 100, 1),
            'veto_by_layer': self.veto_by_layer,
            'veto_by_reason': self.veto_by_reason,
            'top_vetoed_symbols': sorted(self.veto_by_symbol.items(), key=lambda x: x[1], reverse=True)[:10],
            'management_mode_events': self.management_mode_events,
            'management_mode_max_duration_sec': self.management_mode_max_duration_sec,
            'management_mode_reasons': self.management_mode_reasons,
            'regime_distribution': self.regime_distribution,
            'avg_crisis_score': round(self.avg_crisis_score, 2),
            'max_crisis_score': round(self.max_crisis_score, 2),
            'scout_promotions': self.scout_promotions,
            'scout_demotions': self.scout_demotions,
        }


# =============================================================================
# LOG PARSER
# =============================================================================

class ChronosLogParser:
    """
    Parses HolonicTrader session logs to extract forensic signals.

    Reconstructs:
    - Signal generation events
    - Veto events (who blocked what and why)
    - Execution events
    - Management mode durations
    - Regime changes
    - Crisis score evolution
    """

    # Regex patterns for key log events
    _SIGNAL_PATTERNS = [
        re.compile(r'\[EntryOracle\] 🚀 (\S+) (BUY|SELL) SIGNAL \((\w+)\).*?XGB:([\d.]+)', re.IGNORECASE),
        re.compile(r'\[EntryOracle\] ⚡ VOLATILITY SQUEEZE: (\S+) (BUY|SELL).*?Conviction: ([\d.]+)', re.IGNORECASE),
        re.compile(r'\[EntryOracle\] 🐋 WHALE SHADOW: (\S+).*?Triggering (Long|Short)', re.IGNORECASE),
        re.compile(r'\[EntryOracle\] 🪤 SCAVENGER TRAP: (\S+).*?Triggering (Long|Short)', re.IGNORECASE),
    ]

    _VETO_PATTERNS = [
        (re.compile(r'\[EntryOracle\] 🌊 HOLONIC VETO: (\S+) (\S+) - Conviction ([\d.]+) < ([\d.]+)'), 'HOLONIC', lambda m: f"LOW_CONVICTION_{m.group(2)}"),
        (re.compile(r'\[TraderNexus\] 🚫 STRUCTURE BOSS VETO: (\S+) (\S+) blocked\. \((.+?)\)'), 'STRUCTURE_BOSS', lambda m: m.group(3).strip()),
        (re.compile(r'\[TraderNexus\] 🛡️ Governor Vetoed (\S+): (.+)'), 'GOVERNOR', lambda m: m.group(2).strip()),
        (re.compile(r'\[TraderNexus\] \[SMCE\] VETO (\S+) \| \[Layer\d+\] (.+)'), 'SMCE', lambda m: m.group(2).strip()),
        (re.compile(r'\[EntryOracle\] 🐋🚫 WHALE STRUCTURE GATE: (\S+) (\S+) rejected \((.+?)\)'), 'WHALE_GATE', lambda m: m.group(3).strip()),
        (re.compile(r'\[EntryOracle\] 🛡️ PIVOT VETO: (\S+) (.+)'), 'PIVOT', lambda m: m.group(2)[:60].strip()),
        (re.compile(r'\[EntryOracle\] ☢️ CRISIS CAUTION: (\S+) conviction reduced'), 'CRISIS', lambda m: 'CRISIS_SCORE_REDUCTION'),
    ]

    _EXEC_PATTERN = re.compile(
        r'\[ActuatorAgent\].*?(?:FILLED|ORDER|SEND).*?(\S+/USDT)', re.IGNORECASE
    )
    _EXEC_IMPORT = re.compile(r'\[ExecutorAgent\] 📥 Importing: (\S+) \((BUY|SELL)\)')

    _MGMT_MODE_PATTERN = re.compile(
        r'\[TraderNexus\] 🛠️ MANAGEMENT MODE ACTIVE: (\S+)\s+Duration: ([\d]+)s'
    )
    _MGMT_ACTIVATE = re.compile(
        r'\[GovernorAgent\] 🛠️ MANAGEMENT MODE: Activated due to (\S+)'
    )

    _REGIME_PATTERN = re.compile(
        r'\[HolonicAdaptor\] 🌊 REGIME CHANGE: (\S+) → (\S+) \(confidence: ([\d.]+)\)'
    )
    _CRISIS_PATTERN = re.compile(
        r'Crisis Score: ([\d.]+)'
    )
    _SCOUT_PROMO = re.compile(r'\[TraderNexus\] 🚀 SCOUT PROMOTION: (\S+)')
    _SCOUT_DEMO = re.compile(r'\[TraderNexus\] 📉 SCOUT DEMOTION: (\S+)')
    _TIMESTAMP_PATTERN = re.compile(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]')

    def parse_log(self, log_path: str) -> SessionForensics:
        """
        Parse a complete session log file.

        Args:
            log_path: Path to the .log file

        Returns:
            SessionForensics dataclass
        """
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"Log file not found: {log_path}")

        forensics = SessionForensics(
            log_path=log_path,
            session_start="",
            session_end="",
        )

        crisis_scores = []
        last_ts = ""

        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                # Track timestamp
                ts_match = self._TIMESTAMP_PATTERN.search(line)
                if ts_match:
                    ts = ts_match.group(1)
                    if not forensics.session_start:
                        forensics.session_start = ts
                    last_ts = ts

                # ------- SIGNAL DETECTION -------
                for pat in self._SIGNAL_PATTERNS:
                    m = pat.search(line)
                    if m:
                        forensics.signals_generated += 1
                        break

                # ------- VETO DETECTION -------
                for pat, layer, reason_fn in self._VETO_PATTERNS:
                    m = pat.search(line)
                    if m:
                        symbol = m.group(1)
                        reason = reason_fn(m)

                        forensics.signals_blocked += 1
                        forensics.veto_by_layer[layer] = forensics.veto_by_layer.get(layer, 0) + 1
                        forensics.veto_by_reason[reason[:60]] = forensics.veto_by_reason.get(reason[:60], 0) + 1
                        forensics.veto_by_symbol[symbol] = forensics.veto_by_symbol.get(symbol, 0) + 1
                        break

                # ------- MANAGEMENT MODE -------
                mgmt_m = self._MGMT_ACTIVATE.search(line)
                if mgmt_m:
                    reason = mgmt_m.group(1)
                    forensics.management_mode_events += 1
                    forensics.management_mode_reasons[reason] = forensics.management_mode_reasons.get(reason, 0) + 1

                mgmt_dur = self._MGMT_MODE_PATTERN.search(line)
                if mgmt_dur:
                    dur = float(mgmt_dur.group(2))
                    if dur > forensics.management_mode_max_duration_sec:
                        forensics.management_mode_max_duration_sec = dur

                # ------- REGIME -------
                regime_m = self._REGIME_PATTERN.search(line)
                if regime_m:
                    new_regime = regime_m.group(2)
                    forensics.regime_distribution[new_regime] = forensics.regime_distribution.get(new_regime, 0) + 1

                # ------- CRISIS SCORE -------
                crisis_m = self._CRISIS_PATTERN.search(line)
                if crisis_m:
                    score = float(crisis_m.group(1))
                    crisis_scores.append(score)

                # ------- SCOUT COUNTS -------
                if self._SCOUT_PROMO.search(line):
                    forensics.scout_promotions += 1
                if self._SCOUT_DEMO.search(line):
                    forensics.scout_demotions += 1

        forensics.session_end = last_ts

        if crisis_scores:
            forensics.avg_crisis_score = sum(crisis_scores) / len(crisis_scores)
            forensics.max_crisis_score = max(crisis_scores)

        # Approximate signals_executed = signals_generated - signals_blocked
        # (not perfect but log doesn't have a single "EXECUTED" marker cleanly)
        forensics.signals_executed = max(0, forensics.signals_generated - forensics.signals_blocked)

        return forensics

    def find_latest_log(self, directory: str = ".") -> Optional[str]:
        """Find the most recent live trading session log."""
        pattern = os.path.join(directory, "live_trading_session_*.log")
        logs = sorted(glob.glob(pattern), reverse=True)
        return logs[0] if logs else None


# =============================================================================
# CHRONOS FORENSICS ENGINE v2
# =============================================================================

class ChronosForensicsEngine:
    """
    Forensic analysis engine for trading system losses.

    Investigates five domains:
    1. Market Environment
    2. Strategy Logic
    3. Execution Efficiency
    4. Risk Management
    5. Structural Exploitation

    + NEW in v2:
    6. Veto Over-protection (log-based analysis)

    DB Schema (trades table):
        id, symbol, direction, quantity, price, cost_usd,
        timestamp, pnl, pnl_percent, unrealized_pnl,
        unrealized_pnl_percent, mfe, mae
    """

    def __init__(self, db_path: str = "holonic_trader.db", log_dir: str = "."):
        self.db_path = db_path
        self.log_dir = log_dir
        self.log_parser = ChronosLogParser()
        self._cache: Dict[str, Any] = {}
        self._session_forensics: Optional[SessionForensics] = None

        # Loss cause taxonomy
        self.LOSS_CAUSES = {
            'EXECUTION': {
                'SLIPPAGE': 'Trade executed at worse price than expected',
                'LATENCY': 'Order delayed, entered/exited at wrong time',
                'REJECTION': 'Order rejected by exchange',
                'PARTIAL_FILL': 'Order only partially filled',
                'SPREAD_COST': 'Bid-ask spread eroded the edge',
            },
            'SIGNAL': {
                'FALSE_POSITIVE': 'Signal triggered but no follow-through',
                'LAGGING': 'Signal followed price, led to late entry',
                'WHIPSAW': 'Signal triggered, price reversed immediately',
                'STALE_DATA': 'Signal based on outdated market data',
                'LOW_CONVICTION': 'Conviction score too low for market conditions',
            },
            'RISK': {
                'OVERSIZED': 'Position too large for account',
                'STOP_TOO_TIGHT': 'Stop loss hit by normal volatility',
                'STOP_TOO_LOOSE': 'Loss exceeded acceptable range',
                'LEVERAGE_DRAG': 'Funding/margin costs eroded profit',
                'COMPLIANCE_REDUCE': 'Forced compliance reduction ate into winners',
            },
            'REGIME': {
                'VOLATILITY_SHIFT': 'Market volatility changed',
                'LIQUIDITY_DROP': 'Market liquidity decreased',
                'TREND_CHANGE': 'Market regime shifted (mean-rev to trend)',
                'CORRELATION_BREAK': 'Historical correlations broke down',
                'CRISIS_DAMPENING': 'Crisis score dampened otherwise-valid signals',
            },
            'EXPLOITATION': {
                'STOP_HUNT': 'Stop loss targeted by other algorithms',
                'LIQUIDITY_TRAP': 'Liquidity appeared then vanished',
                'ADVERSE_SELECTION': 'Consistently on wrong side of spread',
                'PREDATORY_ALGO': 'Another algo exploiting strategy patterns',
            },
            'VETO_OVERPROTECTION': {
                'EXCESSIVE_VETO': 'Valid signals blocked by conservative veto stack',
                'MGMT_MODE_LOCKOUT': 'System locked in management mode too long',
                'HOLONIC_OVERCAUTION': 'Holonic adaptor conviction threshold too high',
            }
        }

        logger.info("Chronos Forensics Engine v2 initialized")

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _get_closed_trades(self, limit: int = 200) -> pd.DataFrame:
        """
        Fetch closed trades (trades with non-zero pnl) from the DB.

        NOTE: In the HolonicTrader schema, a 'closed' trade is recorded
        when pnl is non-zero. Entries have pnl=0 on open; exits have pnl filled.
        We use pnl != 0 as the closure indicator.
        """
        conn = self._get_connection()
        query = """
            SELECT id, symbol, direction, quantity, price, cost_usd,
                   timestamp, pnl, pnl_percent, mfe, mae
            FROM trades
            WHERE pnl IS NOT NULL AND pnl != 0.0
            ORDER BY id DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        return df

    def _get_all_trades(self, limit: int = 200) -> pd.DataFrame:
        """Fetch all trade rows including open entries."""
        conn = self._get_connection()
        query = "SELECT * FROM trades ORDER BY id DESC LIMIT ?"
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        return df

    def analyze_recent_losses(self, limit: int = 50) -> List[TradeAutopsy]:
        """
        Perform forensic analysis on recent losing trades.

        Args:
            limit: Number of trades to analyze

        Returns:
            List of TradeAutopsy objects
        """
        conn = self._get_connection()
        query = """
            SELECT id, symbol, direction, quantity, price, cost_usd,
                   timestamp, pnl, pnl_percent, mfe, mae
            FROM trades
            WHERE pnl < 0
            ORDER BY id DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()

        if df.empty:
            return []

        autopsies = []
        for _, row in df.iterrows():
            autopsy = self._perform_trade_autopsy(row)
            autopsies.append(autopsy)

        return autopsies

    def _perform_trade_autopsy(self, trade_row) -> TradeAutopsy:
        """
        Perform detailed forensic analysis on a single trade.

        Reconstructs entry/exit timeline and classifies root cause.
        """
        # Calculate hold duration (best effort — only one timestamp per trade)
        try:
            entry_time = datetime.fromisoformat(str(trade_row.get('timestamp', '')))
            hold_duration = (datetime.now(timezone.utc) - entry_time.replace(tzinfo=timezone.utc)).total_seconds()
        except Exception:
            hold_duration = 0.0

        pnl = float(trade_row.get('pnl', 0))
        pnl_pct = float(trade_row.get('pnl_percent', 0))  # Fixed: was 'pnl_pct'
        price = float(trade_row.get('price', 0))           # Fixed: was 'entry_price'

        exit_reason = self._classify_exit_reason(pnl_pct)
        primary_cause, secondary_cause = self._classify_failure_cause(pnl_pct, pnl)
        is_exploitable = self._detect_exploitation_pattern(pnl_pct)
        is_structural = self._is_structural_failure(secondary_cause)

        return TradeAutopsy(
            trade_id=int(trade_row.get('id', 0)),
            symbol=str(trade_row.get('symbol', 'UNKNOWN')),
            direction=str(trade_row.get('direction', 'UNKNOWN')),
            entry_price=price,
            exit_price=0.0,   # Not separately stored in schema
            quantity=float(trade_row.get('quantity', 0)),
            pnl=pnl,
            pnl_pct=pnl_pct,
            entry_timestamp=str(trade_row.get('timestamp', '')),
            exit_timestamp=datetime.now(timezone.utc).isoformat(),
            hold_duration_sec=hold_duration,
            exit_reason=exit_reason,
            primary_cause=primary_cause,
            secondary_cause=secondary_cause,
            is_structural=is_structural,
            is_exploitable=is_exploitable
        )

    def _classify_exit_reason(self, pnl_pct: float) -> str:
        """Classify why the trade exited based on pnl_percent magnitude."""
        pnl_pct_abs = abs(pnl_pct)

        if pnl_pct_abs >= 0.04:    # >= 4% loss → likely stop
            return "STOP_LOSS"
        if pnl_pct_abs >= 0.02:    # 2-4% → staged exit / large TP
            return "STAGED_EXIT"
        if pnl_pct_abs < 0.005:    # < 0.5% → friction / fee / timeout
            return "FRICTION_EXIT"
        return "SIGNAL_EXIT"

    def _classify_failure_cause(self, pnl_pct: float, pnl: float) -> Tuple[str, str]:
        """
        Classify primary and secondary failure causes using real pnl_percent.

        Heuristic tiers:
        - |pnl_pct| < 0.3%  → pure execution friction (spread, fee drag)
        - 0.3%–1.5%         → signal timing issue (false positive, whipsaw)
        - 1.5%–4%           → risk management (stop too tight, leverage drag)
        - >= 4%             → regime mismatch or risk model failure
        """
        pnl_pct_abs = abs(pnl_pct)

        if pnl_pct_abs == 0.0:
            # Zero pnl_percent: fee/spread only trade, pure execution friction
            return ('EXECUTION', 'SPREAD_COST')

        if pnl_pct_abs < 0.003:
            return ('EXECUTION', 'SLIPPAGE')

        if pnl_pct_abs < 0.015:
            return ('SIGNAL', 'FALSE_POSITIVE')

        if pnl_pct_abs < 0.04:
            return ('RISK', 'STOP_TOO_TIGHT')

        # Large loss: regime failure or bad risk controls
        if pnl_pct_abs < 0.08:
            return ('REGIME', 'VOLATILITY_SHIFT')

        return ('RISK', 'STOP_TOO_LOOSE')

    def _detect_exploitation_pattern(self, pnl_pct: float) -> bool:
        """
        Detect if trade shows signs of algorithmic exploitation.
        Stop hunt: loss of exactly 3-5% (stop-loss range hit then reversal)
        """
        pnl_pct_abs = abs(pnl_pct)
        return 0.03 <= pnl_pct_abs <= 0.055

    def _is_structural_failure(self, secondary_cause: str) -> bool:
        """Determine if failure is structural (built into strategy design)."""
        structural_causes = {
            'SLIPPAGE', 'SPREAD_COST', 'LATENCY',
            'LEVERAGE_DRAG', 'ADVERSE_SELECTION', 'COMPLIANCE_REDUCE'
        }
        return secondary_cause in structural_causes

    def get_loss_attribution(self, lookback_trades: int = 100) -> List[LossAttribution]:
        """
        Attribute losses to root causes across five domains.

        Returns breakdown like:
        - 40% SIGNAL
        - 30% EXECUTION
        - 20% RISK
        - 10% REGIME
        """
        conn = self._get_connection()
        query = """
            SELECT id, symbol, direction, pnl, pnl_percent
            FROM trades
            WHERE pnl < 0
            ORDER BY id DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(lookback_trades,))
        conn.close()

        if df.empty:
            return []

        categories = defaultdict(lambda: {'count': 0, 'total_loss': 0.0, 'evidence': []})

        for _, row in df.iterrows():
            pnl_pct = float(row.get('pnl_percent', 0))
            pnl = abs(float(row.get('pnl', 0)))
            symbol = str(row.get('symbol', ''))

            primary, secondary = self._classify_failure_cause(pnl_pct, pnl)
            categories[primary]['count'] += 1
            categories[primary]['total_loss'] += pnl
            evidence_str = f"{symbol}: {pnl_pct*100:.2f}% ({secondary})"
            if len(categories[primary]['evidence']) < 5:
                categories[primary]['evidence'].append(evidence_str)

        total_loss = sum(cat['total_loss'] for cat in categories.values())
        attributions = []

        for category, data in categories.items():
            percentage = (data['total_loss'] / total_loss * 100) if total_loss > 0 else 0
            avg_loss = data['total_loss'] / data['count'] if data['count'] > 0 else 0
            attribution = LossAttribution(
                category=category,
                percentage=percentage,
                total_loss_usd=data['total_loss'],
                trade_count=data['count'],
                avg_loss_per_trade=avg_loss,
                confidence=0.75,
                evidence=data['evidence'],
                recommendation=self._get_category_recommendation(category)
            )
            attributions.append(attribution)

        attributions.sort(key=lambda x: x.percentage, reverse=True)
        return attributions

    def _get_category_recommendation(self, category: str) -> str:
        """Get recommendation for loss category."""
        recommendations = {
            'EXECUTION':           'Review order types and spread thresholds; fee drag may be eliminating edge on small moves',
            'SIGNAL':              'Retrain model, raise conviction thresholds, or add confirmation filters before entry',
            'RISK':                'Reduce leverage, widen stops to ATR multiples, or implement volatility targeting',
            'REGIME':              'Add regime detection layer, reduce exposure during TRANSITION/CHAOS, or pause trading in crisis',
            'EXPLOITATION':        'Randomize entry timing, add decoy orders, reduce pattern predictability',
            'VETO_OVERPROTECTION': 'Re-evaluate veto stack thresholds — system may be over-filtering valid signals',
        }
        return recommendations.get(category, 'Investigate further with log analysis')

    def get_strategy_health_score(self) -> StrategyHealthScore:
        """
        Calculate comprehensive strategy health score (0.0–10.0).

        Uses closed trades only (pnl != 0).
        """
        closed_trades = self._get_closed_trades(limit=200)

        if closed_trades.empty or len(closed_trades) < 5:
            count = len(closed_trades) if not closed_trades.empty else 0
            return StrategyHealthScore(
                timestamp=datetime.now(timezone.utc).timestamp(),
                overall_score=5.0,
                signal_quality=5.0,
                execution_quality=5.0,
                risk_management=5.0,
                market_compatibility=5.0,
                expectancy_status="INSUFFICIENT_DATA",
                regime_alignment="UNKNOWN",
                exploitation_risk="UNKNOWN",
                critical_findings=[f"Only {count} closed trades found — need 5+ for analysis"]
            )

        df = closed_trades
        winning = df[df['pnl'] > 0]
        losing = df[df['pnl'] < 0]

        win_rate = len(winning) / len(df)
        avg_win = float(winning['pnl'].mean()) if not winning.empty else 0.0
        avg_loss = float(abs(losing['pnl'].mean())) if not losing.empty else 0.0

        # Expectancy: E = (W × Aw) − (L × Al)
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # --- 1. Signal Quality (win rate + expectancy contribution) ---
        signal_quality = min(10.0, max(0.0,
            (win_rate * 5.0) +
            (min(expectancy / max(avg_win, 0.01) * 5, 5) if expectancy > 0 else max(expectancy * 5, -5))
        ))

        # --- 2. Execution Quality (consistency of trade results) ---
        pnl_std = float(df['pnl'].std())
        avg_abs_pnl = float(df['pnl'].abs().mean())
        cv = pnl_std / max(avg_abs_pnl, 0.001)  # Coefficient of variation
        execution_quality = min(10.0, max(0.0, 10.0 - min(cv * 2, 10)))

        # --- 3. Risk Management (max loss discipline) ---
        max_single_loss = float(abs(df['pnl'].min()))
        avg_account = 90.0  # Approximate from log ($91.64)
        max_loss_pct_account = max_single_loss / avg_account
        risk_management = min(10.0, max(0.0, 10.0 - (max_loss_pct_account * 100)))  # Penalize > 1% per trade

        # --- 4. Market Compatibility (recent vs older trend) ---
        midpoint = len(df) // 2
        recent_pnl = float(df.head(midpoint)['pnl'].mean()) if midpoint > 0 else 0.0
        older_pnl = float(df.tail(midpoint)['pnl'].mean()) if midpoint > 0 else 0.0

        if recent_pnl > 0 and recent_pnl >= older_pnl:
            market_compatibility = 8.0
        elif recent_pnl > 0 and recent_pnl < older_pnl:
            market_compatibility = 6.0
        elif recent_pnl <= 0 and older_pnl > 0:
            market_compatibility = 3.0
        else:
            market_compatibility = 4.0

        overall_score = (
            signal_quality * 0.35 +
            execution_quality * 0.25 +
            risk_management * 0.25 +
            market_compatibility * 0.15
        )

        expectancy_status = "POSITIVE" if expectancy > 0 else ("NEUTRAL" if expectancy == 0 else "NEGATIVE")
        regime_alignment = "ALIGNED" if market_compatibility >= 6.0 else "MISALIGNED"
        exploitation_risk = "HIGH" if self._detect_systematic_exploitation(df) else ("MEDIUM" if win_rate < 0.4 else "LOW")

        recommendations = []
        critical_findings = []

        if win_rate < 0.4:
            critical_findings.append(f"Low win rate ({win_rate*100:.1f}%) — signal quality degraded")
            recommendations.append("Retrain entry model or add confirmation filters")

        if expectancy < 0:
            critical_findings.append(f"Negative expectancy (${expectancy:.4f}/trade) — system loses money over time")
            recommendations.append("Address loss/win asymmetry: either raise Avg_Win or reduce Avg_Loss")

        if max_single_loss > 1.5:
            critical_findings.append(f"Large single-trade loss (${max_single_loss:.2f}) — risk management breach")
            recommendations.append("Reduce position size or tighten stops for volatile assets")

        if market_compatibility < 5.0:
            critical_findings.append("Strategy performance degrading — possibly wrong market regime")
            recommendations.append("Review strategy assumptions against current SMCE regime")

        return StrategyHealthScore(
            timestamp=datetime.now(timezone.utc).timestamp(),
            overall_score=overall_score,
            signal_quality=signal_quality,
            execution_quality=execution_quality,
            risk_management=risk_management,
            market_compatibility=market_compatibility,
            expectancy_status=expectancy_status,
            regime_alignment=regime_alignment,
            exploitation_risk=exploitation_risk,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            expectancy=expectancy,
            recommendations=recommendations,
            critical_findings=critical_findings
        )

    def _detect_systematic_exploitation(self, df: pd.DataFrame) -> bool:
        """Detect if strategy shows signs of being systematically exploited."""
        if len(df) < 20:
            return False

        losing = df[df['pnl'] < 0]
        loss_rate = len(losing) / len(df)
        if loss_rate > 0.7:
            return True

        avg_loss = abs(losing['pnl'].mean()) if not losing.empty else 0
        winning = df[df['pnl'] > 0]
        avg_win = winning['pnl'].mean() if not winning.empty else 0

        if avg_win > 0 and avg_loss / avg_win > 3:
            return True

        return False

    def analyze_session_log(self, log_path: Optional[str] = None) -> Optional[SessionForensics]:
        """
        Analyze the most recent (or specified) session log.

        Args:
            log_path: Optional specific log path. If None, uses most recent.

        Returns:
            SessionForensics object or None
        """
        if log_path is None:
            log_path = self.log_parser.find_latest_log(self.log_dir)

        if log_path is None:
            logger.warning("No session log found")
            return None

        try:
            self._session_forensics = self.log_parser.parse_log(log_path)
            return self._session_forensics
        except Exception as e:
            logger.error(f"Log analysis failed: {e}")
            return None

    def get_veto_attribution(self, log_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze which veto layers are blocking the most signals.

        Returns:
            Dict with veto layer breakdown, top blocked symbols, and assessment
        """
        sf = self._session_forensics
        if sf is None:
            sf = self.analyze_session_log(log_path)

        if sf is None:
            return {'error': 'No session log available for veto analysis'}

        total_vetoes = sum(sf.veto_by_layer.values()) or 1

        layer_pct = {
            layer: round(count / total_vetoes * 100, 1)
            for layer, count in sorted(sf.veto_by_layer.items(), key=lambda x: x[1], reverse=True)
        }

        # Assess if system is over-vetoing
        pass_rate = sf.pass_rate()
        veto_assessment = "NORMAL"
        if pass_rate < 0.05:
            veto_assessment = "CRITICAL_OVERPROTECTION"
        elif pass_rate < 0.15:
            veto_assessment = "HIGH_OVERPROTECTION"
        elif pass_rate < 0.30:
            veto_assessment = "MODERATE_OVERPROTECTION"

        return {
            'veto_assessment': veto_assessment,
            'signals_generated': sf.signals_generated,
            'signals_blocked': sf.signals_blocked,
            'signals_executed': sf.signals_executed,
            'pass_rate_pct': round(pass_rate * 100, 1),
            'veto_by_layer_pct': layer_pct,
            'top_blocked_symbols': sorted(sf.veto_by_symbol.items(), key=lambda x: x[1], reverse=True)[:8],
            'management_mode_events': sf.management_mode_events,
            'management_mode_max_duration_sec': sf.management_mode_max_duration_sec,
            'management_mode_reasons': sf.management_mode_reasons,
            'avg_crisis_score': sf.avg_crisis_score,
        }

    def generate_forensic_report(self, log_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive forensic report.

        Combines:
        - DB trade analysis (health score, loss attribution, autopsies)
        - Log-based session analysis (veto breakdown, regime distribution)

        Returns:
            Complete diagnostic report dict
        """
        autopsies = self.analyze_recent_losses(limit=50)
        attributions = self.get_loss_attribution(lookback_trades=100)
        health = self.get_strategy_health_score()
        session = self.analyze_session_log(log_path)
        veto = self.get_veto_attribution()

        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'chronos_version': '2.0',
            'executive_summary': self._generate_executive_summary(health, attributions, veto),
            'strategy_health': health.to_dict(),
            'loss_attribution': [a.to_dict() for a in attributions],
            'trade_autopsies': [a.to_dict() for a in autopsies[:10]],
            'session_forensics': session.to_dict() if session else None,
            'veto_attribution': veto,
            'critical_findings': health.critical_findings,
            'recommendations': health.recommendations,
            'next_actions': self._generate_action_plan(health, attributions, veto)
        }

        return report

    def _generate_executive_summary(
        self,
        health: StrategyHealthScore,
        attributions: List[LossAttribution],
        veto: Dict[str, Any]
    ) -> str:
        """Generate executive summary text."""
        if health.overall_score >= 7:
            status = "HEALTHY"
        elif health.overall_score >= 5:
            status = "DEGRADED"
        else:
            status = "CRITICAL"

        summary = f"Strategy Status: {status}\n"
        summary += f"Overall Score: {health.overall_score:.1f}/10\n"
        summary += f"Expectancy: ${health.expectancy:.4f}/trade ({health.expectancy_status})\n"
        summary += f"Win Rate: {health.win_rate*100:.1f}% | Avg Win: ${health.avg_win:.2f} | Avg Loss: ${health.avg_loss:.2f}\n\n"

        if attributions:
            summary += "Primary Loss Drivers:\n"
            for attr in attributions[:3]:
                summary += f"  - {attr.category}: {attr.percentage:.1f}% (${attr.total_loss_usd:.2f})\n"

        if veto and veto.get('pass_rate_pct') is not None:
            summary += f"\nVeto Analysis: Only {veto.get('pass_rate_pct', 0):.1f}% of signals execute ({veto.get('veto_assessment', 'N/A')})\n"

        if health.critical_findings:
            summary += "\nCritical Issues:\n"
            for finding in health.critical_findings:
                summary += f"  ⚠️ {finding}\n"

        return summary

    def _generate_action_plan(
        self,
        health: StrategyHealthScore,
        attributions: List[LossAttribution],
        veto: Dict[str, Any]
    ) -> List[str]:
        """Generate prioritized action plan."""
        actions = []

        if health.critical_findings:
            actions.append("IMMEDIATE: Address critical findings")
            for finding in health.critical_findings:
                actions.append(f"  → {finding}")

        if attributions:
            top = attributions[0]
            actions.append(f"\nHIGH: Address {top.category} losses ({top.percentage:.1f}%)")
            actions.append(f"  → {top.recommendation}")

        veto_assess = veto.get('veto_assessment', 'NORMAL')
        if veto_assess in ('CRITICAL_OVERPROTECTION', 'HIGH_OVERPROTECTION'):
            actions.append(f"\nHIGH: Veto stack blocking too many signals ({veto_assess})")
            actions.append(f"  → Pass rate {veto.get('pass_rate_pct', 0):.1f}% — review HOLONIC conviction thresholds and management mode duration")

        if health.signal_quality < 5:
            actions.append("\nMEDIUM: Improve signal quality")
            actions.append("  → Review entry model, add confirmation filters")

        if health.execution_quality < 5:
            actions.append("\nMEDIUM: Improve execution quality")
            actions.append("  → Review order types, monitor spread costs")

        if health.risk_management < 5:
            actions.append("\nMEDIUM: Strengthen risk management")
            actions.append("  → Reduce position size, adjust stops to ATR multiples")

        return actions


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_forensics_engine: Optional[ChronosForensicsEngine] = None


def get_chronos_engine(
    db_path: str = "holonic_trader.db",
    log_dir: str = "."
) -> ChronosForensicsEngine:
    """Get or create global Chronos forensics engine."""
    global _forensics_engine
    if _forensics_engine is None:
        _forensics_engine = ChronosForensicsEngine(db_path=db_path, log_dir=log_dir)
    return _forensics_engine


def generate_chronos_report(
    db_path: str = "holonic_trader.db",
    log_path: Optional[str] = None,
    log_dir: str = "."
) -> Dict[str, Any]:
    """Generate comprehensive Chronos forensic report."""
    engine = get_chronos_engine(db_path=db_path, log_dir=log_dir)
    return engine.generate_forensic_report(log_path=log_path)
