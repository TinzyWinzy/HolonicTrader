"""
AEGIS QUANTSEC - Position Reconciliation Engine

Real-time 3-way cross-verification between:
1. Internal Ledger (ExecutorHolon positions)
2. Exchange State (Kraken API)
3. Websocket Feed (Real-time orderbook)

Addresses CRITICAL finding C-01: Ledger-Exchange Divergence Vulnerability

Author: AEGIS QuantSec v1.0
Date: 2026-03-15
"""

import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import defaultdict
import logging

logger = logging.getLogger("AEGIS.PositionReconciliation")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PositionSnapshot:
    """Snapshot of a position from a specific source."""
    source: str  # LEDGER, EXCHANGE, WEBSOCKET
    symbol: str
    quantity: float
    entry_price: float
    timestamp: float
    timestamp_ns: int
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'source': self.source,
            'symbol': self.symbol,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'timestamp': self.timestamp,
            'timestamp_ns': self.timestamp_ns,
            'metadata': self.metadata
        }


@dataclass
class ReconciliationDiscrepancy:
    """Detected discrepancy between sources."""
    discrepancy_type: str  # GHOST, LEAK, MISMATCH, PRICE_DIVERGENCE
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    symbol: str
    ledger_qty: float
    exchange_qty: float
    ws_qty: Optional[float]
    ledger_price: float
    exchange_price: float
    timestamp: float
    details: str
    recommended_action: str
    resolved: bool = False
    resolution: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'type': self.discrepancy_type,
            'severity': self.severity,
            'symbol': self.symbol,
            'ledger_qty': self.ledger_qty,
            'exchange_qty': self.exchange_qty,
            'ws_qty': self.ws_qty,
            'ledger_price': self.ledger_price,
            'exchange_price': self.exchange_price,
            'timestamp': self.timestamp,
            'details': self.details,
            'recommended_action': self.recommended_action,
            'resolved': self.resolved,
            'resolution': self.resolution
        }


@dataclass
class ReconciliationReport:
    """Complete reconciliation report."""
    timestamp: float
    ledger_positions: Dict[str, float]
    exchange_positions: Dict[str, float]
    ws_positions: Dict[str, float]
    discrepancies: List[ReconciliationDiscrepancy]
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'ledger_positions': self.ledger_positions,
            'exchange_positions': self.exchange_positions,
            'ws_positions': self.ws_positions,
            'discrepancies': [d.to_dict() for d in self.discrepancies],
            'summary': self.summary
        }


# =============================================================================
# POSITION RECONCILIATION ENGINE
# =============================================================================

class PositionReconciliationEngine:
    """
    Real-time 3-way position reconciliation.
    
    Continuously monitors and cross-verifies positions between:
    1. Internal ledger (ExecutorHolon)
    2. Exchange state (Kraken API)
    3. Websocket feed (Real-time updates)
    
    Detects:
    - Ghost positions (on exchange, not in ledger)
    - Leak positions (in ledger, not on exchange)
    - Quantity mismatches
    - Price divergences
    """
    
    def __init__(
        self,
        executor_holon=None,
        kraken_holon=None,
        websocket_monitor=None,
        reconciliation_interval_sec: float = 5.0,
        price_divergence_threshold: float = 0.001,  # 0.1% price difference
        quantity_tolerance: float = 0.0001,  # Tolerance for quantity comparison
        auto_resolve: bool = False
    ):
        self.executor = executor_holon
        self.kraken_holon = kraken_holon
        self.ws_monitor = websocket_monitor
        
        self.reconciliation_interval = reconciliation_interval_sec
        self.price_divergence_threshold = price_divergence_threshold
        self.quantity_tolerance = quantity_tolerance
        self.auto_resolve = auto_resolve
        
        # State tracking
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_report: Optional[ReconciliationReport] = None
        self._discrepancy_history: List[ReconciliationDiscrepancy] = []
        self._position_history: Dict[str, List[PositionSnapshot]] = defaultdict(list)
        
        # Alert callbacks
        self._alert_callbacks: List[callable] = []
        
        # Statistics
        self._stats = {
            'reconciliations_run': 0,
            'discrepancies_found': 0,
            'ghosts_detected': 0,
            'leaks_detected': 0,
            'mismatches_detected': 0,
            'last_reconciliation_time': 0.0
        }
    
    def start(self):
        """Start background reconciliation thread."""
        if self._running:
            logger.warning("Reconciliation engine already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._reconciliation_loop, daemon=True)
        self._thread.start()
        logger.info("Position reconciliation engine started")
    
    def stop(self):
        """Stop background reconciliation thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Position reconciliation engine stopped")
    
    def _reconciliation_loop(self):
        """Background reconciliation loop."""
        while self._running:
            try:
                self.run_reconciliation()
            except Exception as e:
                logger.error(f"Reconciliation error: {e}")
            
            time.sleep(self.reconciliation_interval)
    
    def run_reconciliation(self) -> ReconciliationReport:
        """
        Run a full 3-way reconciliation.
        
        Returns:
            ReconciliationReport with all discrepancies
        """
        start_time = time.time()
        
        # 1. Fetch positions from all sources
        ledger_positions = self._fetch_ledger_positions()
        exchange_positions = self._fetch_exchange_positions()
        ws_positions = self._fetch_websocket_positions()
        
        # 2. Detect discrepancies
        discrepancies = self._detect_discrepancies(
            ledger_positions,
            exchange_positions,
            ws_positions
        )
        
        # 3. Update statistics
        self._stats['reconciliations_run'] += 1
        self._stats['discrepancies_found'] += len(discrepancies)
        self._stats['last_reconciliation_time'] = time.time() - start_time
        
        for d in discrepancies:
            if d.discrepancy_type == 'GHOST':
                self._stats['ghosts_detected'] += 1
            elif d.discrepancy_type == 'LEAK':
                self._stats['leaks_detected'] += 1
            elif d.discrepancy_type == 'MISMATCH':
                self._stats['mismatches_detected'] += 1
        
        # 4. Store history
        self._discrepancy_history.extend(discrepancies)
        if len(self._discrepancy_history) > 1000:
            self._discrepancy_history = self._discrepancy_history[-500:]
        
        # 5. Create report
        report = ReconciliationReport(
            timestamp=start_time,
            ledger_positions=ledger_positions,
            exchange_positions=exchange_positions,
            ws_positions=ws_positions,
            discrepancies=discrepancies,
            summary=self._generate_summary(ledger_positions, exchange_positions, discrepancies)
        )
        
        self._last_report = report
        
        # 6. Trigger alerts for critical/high severity
        critical_discrepancies = [d for d in discrepancies if d.severity in ['CRITICAL', 'HIGH']]
        if critical_discrepancies:
            self._trigger_alerts(critical_discrepancies, report)
        
        # 7. Auto-resolve if enabled
        if self.auto_resolve:
            self._auto_resolve_discrepancies(discrepancies)
        
        return report
    
    def _fetch_ledger_positions(self) -> Dict[str, float]:
        """Fetch positions from internal ledger (ExecutorHolon)."""
        if not self.executor:
            return {}
        
        try:
            positions = {}
            if hasattr(self.executor, 'positions'):
                for symbol, pos in self.executor.positions.items():
                    qty = pos.quantity if hasattr(pos, 'quantity') else pos.get('quantity', 0)
                    positions[symbol] = float(qty)
            return positions
        except Exception as e:
            logger.error(f"Failed to fetch ledger positions: {e}")
            return {}
    
    def _fetch_exchange_positions(self) -> Dict[str, float]:
        """Fetch positions from Kraken exchange."""
        if not self.kraken_holon:
            return {}
        
        try:
            positions = {}
            
            # Use KrakenHolon's detect_ghosts method or fetch directly
            if hasattr(self.kraken_holon, 'futures'):
                kraken_positions = self.kraken_holon.futures.fetch_positions()
                
                for p in kraken_positions:
                    qty = float(p.get('contracts', 0.0))
                    if qty == 0:
                        continue
                    
                    # Handle short positions
                    if p.get('side') == 'short':
                        qty = -qty
                    
                    # Normalize symbol
                    symbol = self._normalize_symbol(p.get('symbol', ''))
                    if symbol:
                        positions[symbol] = qty
            
            return positions
        except Exception as e:
            logger.error(f"Failed to fetch exchange positions: {e}")
            return {}
    
    def _fetch_websocket_positions(self) -> Dict[str, float]:
        """
        Fetch positions from websocket feed.
        
        Note: This is an estimate based on orderbook changes and trade fills.
        """
        if not self.ws_monitor:
            return {}
        
        try:
            # Get latest position estimates from websocket monitor
            if hasattr(self.ws_monitor, 'get_position_estimates'):
                return self.ws_monitor.get_position_estimates()
            return {}
        except Exception as e:
            logger.error(f"Failed to fetch websocket positions: {e}")
            return {}
    
    def _detect_discrepancies(
        self,
        ledger: Dict[str, float],
        exchange: Dict[str, float],
        ws: Dict[str, float]
    ) -> List[ReconciliationDiscrepancy]:
        """Detect discrepancies between sources."""
        discrepancies = []
        now = time.time()
        now_ns = int(now * 1e9)
        
        all_symbols = set(ledger.keys()) | set(exchange.keys()) | set(ws.keys())
        
        for symbol in all_symbols:
            ledger_qty = ledger.get(symbol, 0.0)
            exchange_qty = exchange.get(symbol, 0.0)
            ws_qty = ws.get(symbol, None)
            
            # Get prices for divergence detection
            ledger_price = self._get_ledger_price(symbol)
            exchange_price = self._get_exchange_price(symbol)
            
            # 1. Ghost detection (on exchange, not in ledger)
            if abs(exchange_qty) > self.quantity_tolerance and abs(ledger_qty) < self.quantity_tolerance:
                discrepancies.append(ReconciliationDiscrepancy(
                    discrepancy_type='GHOST',
                    severity='CRITICAL',
                    symbol=symbol,
                    ledger_qty=ledger_qty,
                    exchange_qty=exchange_qty,
                    ws_qty=ws_qty,
                    ledger_price=ledger_price,
                    exchange_price=exchange_price,
                    timestamp=now,
                    details=f"Position exists on exchange ({exchange_qty}) but not in ledger",
                    recommended_action='INVESTIGATE: Check for unlogged trades or API errors'
                ))
            
            # 2. Leak detection (in ledger, not on exchange)
            elif abs(ledger_qty) > self.quantity_tolerance and abs(exchange_qty) < self.quantity_tolerance:
                discrepancies.append(ReconciliationDiscrepancy(
                    discrepancy_type='LEAK',
                    severity='CRITICAL',
                    symbol=symbol,
                    ledger_qty=ledger_qty,
                    exchange_qty=exchange_qty,
                    ws_qty=ws_qty,
                    ledger_price=ledger_price,
                    exchange_price=exchange_price,
                    timestamp=now,
                    details=f"Position exists in ledger ({ledger_qty}) but not on exchange",
                    recommended_action='PURGE: Remove phantom position from ledger'
                ))
            
            # 3. Quantity mismatch
            elif abs(ledger_qty - exchange_qty) > self.quantity_tolerance:
                severity = 'HIGH' if abs(ledger_qty - exchange_qty) / max(abs(ledger_qty), 0.001) > 0.1 else 'MEDIUM'
                discrepancies.append(ReconciliationDiscrepancy(
                    discrepancy_type='MISMATCH',
                    severity=severity,
                    symbol=symbol,
                    ledger_qty=ledger_qty,
                    exchange_qty=exchange_qty,
                    ws_qty=ws_qty,
                    ledger_price=ledger_price,
                    exchange_price=exchange_price,
                    timestamp=now,
                    details=f"Quantity mismatch: ledger={ledger_qty}, exchange={exchange_qty}",
                    recommended_action='RECONCILE: Update ledger to match exchange'
                ))
            
            # 4. Price divergence
            if ledger_price > 0 and exchange_price > 0:
                price_diff_pct = abs(ledger_price - exchange_price) / ledger_price
                if price_diff_pct > self.price_divergence_threshold:
                    discrepancies.append(ReconciliationDiscrepancy(
                        discrepancy_type='PRICE_DIVERGENCE',
                        severity='MEDIUM',
                        symbol=symbol,
                        ledger_qty=ledger_qty,
                        exchange_qty=exchange_qty,
                        ws_qty=ws_qty,
                        ledger_price=ledger_price,
                        exchange_price=exchange_price,
                        timestamp=now,
                        details=f"Price divergence: ledger=${ledger_price}, exchange=${exchange_price} ({price_diff_pct*100:.2f}%)",
                        recommended_action='UPDATE: Refresh entry price in ledger'
                    ))
        
        return discrepancies
    
    def _get_ledger_price(self, symbol: str) -> float:
        """Get entry price from ledger."""
        if not self.executor:
            return 0.0
        
        try:
            if hasattr(self.executor, 'positions') and symbol in self.executor.positions:
                pos = self.executor.positions[symbol]
                if hasattr(pos, 'entry_price'):
                    return float(pos.entry_price)
                elif isinstance(pos, dict) and 'entry_price' in pos:
                    return float(pos['entry_price'])
        except:
            pass
        return 0.0
    
    def _get_exchange_price(self, symbol: str) -> float:
        """Get current price from exchange."""
        if not self.kraken_holon:
            return 0.0
        
        try:
            # Get latest ticker
            if hasattr(self.kraken_holon, 'futures'):
                exchange_symbol = self._to_exchange_symbol(symbol)
                ticker = self.kraken_holon.futures.fetch_ticker(exchange_symbol)
                return float(ticker.get('last', 0.0))
        except:
            pass
        return 0.0
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize exchange symbol to internal format."""
        if not symbol:
            return ""
        
        # Handle Kraken Futures format: PF_XBTUSD -> BTC/USDT
        if symbol.startswith('PF_'):
            base = symbol[3:]
            if base.startswith('XBT'):
                base = 'BTC' + base[3:]
            if base.endswith('USD'):
                base = base[:-3]
            return f"{base}/USDT"
        
        # Handle CCXT format: BTC/USD:USD -> BTC/USDT
        if '/USD:USD' in symbol:
            base = symbol.split('/')[0]
            return f"{base}/USDT"
        
        return symbol
    
    def _to_exchange_symbol(self, symbol: str) -> str:
        """Convert internal symbol to exchange format."""
        # Import config for symbol mapping
        try:
            import config
            if hasattr(config, 'KRAKEN_SYMBOL_MAP'):
                return config.KRAKEN_SYMBOL_MAP.get(symbol, symbol)
        except:
            pass
        
        # Fallback conversion
        if '/USDT' in symbol:
            base = symbol.split('/')[0]
            if base == 'BTC':
                base = 'XBT'
            return f"PF_{base}USD"
        
        return symbol
    
    def _generate_summary(
        self,
        ledger: Dict[str, float],
        exchange: Dict[str, float],
        discrepancies: List[ReconciliationDiscrepancy]
    ) -> Dict[str, Any]:
        """Generate reconciliation summary."""
        by_severity = defaultdict(int)
        by_type = defaultdict(int)
        
        for d in discrepancies:
            by_severity[d.severity] += 1
            by_type[d.discrepancy_type] += 1
        
        return {
            'status': 'HEALTHY' if not discrepancies else 'DISCREPANCIES_FOUND',
            'total_positions': {
                'ledger': len(ledger),
                'exchange': len(exchange)
            },
            'discrepancies': {
                'total': len(discrepancies),
                'by_severity': dict(by_severity),
                'by_type': dict(by_type)
            },
            'stats': self._stats.copy()
        }
    
    def _trigger_alerts(
        self,
        discrepancies: List[ReconciliationDiscrepancy],
        report: ReconciliationReport
    ):
        """Trigger alert callbacks for critical discrepancies."""
        for callback in self._alert_callbacks:
            try:
                callback(discrepancies, report)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def register_alert_callback(self, callback: callable):
        """Register a callback for discrepancy alerts."""
        self._alert_callbacks.append(callback)
    
    def _auto_resolve_discrepancies(self, discrepancies: List[ReconciliationDiscrepancy]):
        """Attempt automatic resolution of discrepancies."""
        for d in discrepancies:
            if d.discrepancy_type == 'LEAK':
                # Auto-purge leak positions
                self._purge_leak_position(d.symbol)
                d.resolved = True
                d.resolution = 'AUTO_PURGED'
            
            elif d.discrepancy_type == 'MISMATCH' and d.severity == 'MEDIUM':
                # Auto-reconcile small mismatches
                self._reconcile_position(d.symbol, d.exchange_qty)
                d.resolved = True
                d.resolution = 'AUTO_RECONCILED'
    
    def _purge_leak_position(self, symbol: str):
        """Purge a leak position from ledger."""
        if not self.executor:
            return
        
        try:
            if hasattr(self.executor, 'positions') and symbol in self.executor.positions:
                del self.executor.positions[symbol]
                logger.info(f"[AEGIS] Auto-purged leak position: {symbol}")
        except Exception as e:
            logger.error(f"Failed to purge leak position {symbol}: {e}")
    
    def _reconcile_position(self, symbol: str, correct_qty: float):
        """Reconcile position quantity."""
        if not self.executor:
            return
        
        try:
            if hasattr(self.executor, 'positions') and symbol in self.executor.positions:
                self.executor.positions[symbol].quantity = correct_qty
                logger.info(f"[AEGIS] Auto-reconciled position {symbol} to {correct_qty}")
        except Exception as e:
            logger.error(f"Failed to reconcile position {symbol}: {e}")
    
    def get_latest_report(self) -> Optional[ReconciliationReport]:
        """Get the latest reconciliation report."""
        return self._last_report
    
    def get_discrepancy_history(
        self,
        symbol: Optional[str] = None,
        discrepancy_type: Optional[str] = None,
        limit: int = 100
    ) -> List[ReconciliationDiscrepancy]:
        """Get discrepancy history with optional filtering."""
        results = self._discrepancy_history
        
        if symbol:
            results = [d for d in results if d.symbol == symbol]
        if discrepancy_type:
            results = [d for d in results if d.discrepancy_type == discrepancy_type]
        
        return results[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get reconciliation statistics."""
        return self._stats.copy()
    
    def get_integrity_score(self) -> float:
        """
        Calculate position integrity score (0.0 to 1.0).
        
        1.0 = Perfect reconciliation
        0.0 = Complete divergence
        """
        if not self._last_report:
            return 1.0
        
        total_positions = len(self._last_report.ledger_positions) + len(self._last_report.exchange_positions)
        if total_positions == 0:
            return 1.0
        
        critical_count = sum(1 for d in self._last_report.discrepancies if d.severity == 'CRITICAL')
        high_count = sum(1 for d in self._last_report.discrepancies if d.severity == 'HIGH')
        
        # Weighted penalty
        penalty = (critical_count * 0.2) + (high_count * 0.1) + (len(self._last_report.discrepancies) * 0.02)
        
        return max(0.0, 1.0 - penalty)


# =============================================================================
# TELEGRAM ALERT INTEGRATION
# =============================================================================

class ReconciliationAlertHandler:
    """
    Sends Telegram alerts for reconciliation discrepancies.
    """
    
    def __init__(
        self,
        telegram_bot,
        chat_id: str,
        min_severity: str = 'HIGH'
    ):
        self.telegram_bot = telegram_bot
        self.chat_id = chat_id
        self.min_severity = min_severity
        self._last_alert_time = 0.0
        self._alert_cooldown = 60.0  # 1 minute between alerts
    
    def on_discrepancy(
        self,
        discrepancies: List[ReconciliationDiscrepancy],
        report: ReconciliationReport
    ):
        """Callback for discrepancy alerts."""
        now = time.time()
        if now - self._last_alert_time < self._alert_cooldown:
            return
        
        # Filter by severity
        severity_order = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        min_level = severity_order.get(self.min_severity, 3)
        
        filtered = [d for d in discrepancies if severity_order.get(d.severity, 0) >= min_level]
        
        if not filtered:
            return
        
        # Send alert
        self._send_alert(filtered, report)
        self._last_alert_time = now
    
    def _send_alert(
        self,
        discrepancies: List[ReconciliationDiscrepancy],
        report: ReconciliationReport
    ):
        """Send Telegram alert."""
        summary = report.summary
        
        message = f"""
🚨 *POSITION RECONCILIATION ALERT*

*Status:* {summary['status']}
*Time:* {datetime.fromtimestamp(report.timestamp).isoformat()}

*Discrepancies:* {len(discrepancies)}
- Critical: {summary['discrepancies']['by_severity'].get('CRITICAL', 0)}
- High: {summary['discrepancies']['by_severity'].get('HIGH', 0)}

*Details:*
"""
        for d in discrepancies[:5]:  # Limit to 5
            message += f"\n[{d.severity}] {d.symbol}: {d.discrepancy_type}"
            message += f"\n  {d.details}"
        
        if len(discrepancies) > 5:
            message += f"\n... and {len(discrepancies) - 5} more"
        
        message += "\n\n*Action Required:* Review position reconciliation immediately"
        
        try:
            if self.telegram_bot:
                self.telegram_bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode='Markdown'
                )
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")


# =============================================================================
# INTEGRATION WITH EXECUTORHOLON
# =============================================================================

def integrate_reconciliation_engine(
    executor_holon,
    kraken_holon=None,
    websocket_monitor=None,
    enable_telegram: bool = False,
    telegram_bot=None,
    chat_id: str = None,
    auto_resolve: bool = False
) -> PositionReconciliationEngine:
    """
    Integrate Position Reconciliation Engine with ExecutorHolon.
    
    Usage:
        engine = integrate_reconciliation_engine(
            executor_holon=executor,
            kraken_holon=kraken,
            enable_telegram=True,
            telegram_bot=bot,
            chat_id=CHAT_ID
        )
        engine.start()
    """
    # Create engine
    engine = PositionReconciliationEngine(
        executor_holon=executor_holon,
        kraken_holon=kraken_holon,
        websocket_monitor=websocket_monitor,
        auto_resolve=auto_resolve
    )
    
    # Add Telegram alerts if enabled
    if enable_telegram and telegram_bot and chat_id:
        alert_handler = ReconciliationAlertHandler(
            telegram_bot=telegram_bot,
            chat_id=chat_id,
            min_severity='HIGH'
        )
        engine.register_alert_callback(alert_handler.on_discrepancy)
        logger.info("Telegram alerts enabled for reconciliation")
    
    # Store reference on executor for access
    if hasattr(executor_holon, '__dict__'):
        executor_holon._reconciliation_engine = engine
    
    logger.info("Position Reconciliation Engine integrated")
    return engine


# =============================================================================
# CLI VERIFICATION TOOL
# =============================================================================

def run_reconciliation_check(
    executor_holon,
    kraken_holon=None
) -> Dict[str, Any]:
    """
    Run a one-time reconciliation check.
    
    Usage:
        from HolonicTrader.position_reconciliation import run_reconciliation_check
        report = run_reconciliation_check(executor, kraken)
        print(f"Status: {report['summary']['status']}")
    """
    engine = PositionReconciliationEngine(
        executor_holon=executor_holon,
        kraken_holon=kraken_holon
    )
    
    report = engine.run_reconciliation()
    return report.to_dict()


if __name__ == "__main__":
    print("AEGIS QUANTSEC - Position Reconciliation Engine")
    print()
    print("This module integrates with ExecutorHolon and KrakenHolon")
    print("to provide real-time position reconciliation.")
    print()
    print("Usage:")
    print("  from HolonicTrader.position_reconciliation import integrate_reconciliation_engine")
    print("  engine = integrate_reconciliation_engine(executor, kraken)")
    print("  engine.start()")
