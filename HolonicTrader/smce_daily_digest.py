"""
SMCE Daily Digest – Audit-Only Report Generator

Generates a daily audit digest at 00:00 UTC containing:
  - Equity: start, end, change %
  - Trades executed: symbol, direction, size, entry/exit, P&L
  - Regime: classification and key inputs
  - Scorecards: per-trade probability scores
  - Monte Carlo summary: portfolio VaR, CVaR, stress results
  - Constitutional violations: flagged for audit (should be zero)
  - System health: warnings (data feeds, execution delays)

This digest is OBSERVATION ONLY – no pause or approval required.
Output: log file (and optionally Telegram).
"""

import time
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger("SMCEDailyDigest")


class DailyDigestGenerator:
    """
    Generates and schedules the 00:00 UTC daily audit digest.

    Usage:
        digest = DailyDigestGenerator(log_dir="logs/", telegram_fn=None)
        digest.record_trade({...})          # Call after each trade
        digest.record_regime({...})         # Call each cycle
        digest.schedule_daily()             # Start background scheduler
    """

    def __init__(
        self,
        log_dir: str = "logs/",
        telegram_fn=None,  # Optional callable(message: str) for Telegram
    ):
        self.log_dir       = Path(log_dir)
        self.telegram_fn   = telegram_fn
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Daily accumulators (reset after each digest)
        self._reset_accumulators()
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None

    # ─── Recording API ────────────────────────────────────────────────────────

    def set_start_equity(self, equity: float):
        """Call at system start or after digest generation."""
        with self._lock:
            if self._start_equity == 0.0:
                self._start_equity = equity

    def record_trade(self, trade: Dict[str, Any]):
        """
        Record an executed trade for today's digest.
        Expected keys: symbol, direction, notional, entry_price, exit_price,
                       pnl_usd, scorecard, mc_result
        """
        with self._lock:
            self._trades.append({
                "ts":          time.time(),
                "symbol":      trade.get("symbol", "?"),
                "direction":   trade.get("direction", "?"),
                "notional":    trade.get("notional", 0.0),
                "entry_price": trade.get("entry_price", 0.0),
                "exit_price":  trade.get("exit_price", 0.0),
                "pnl_usd":     trade.get("pnl_usd", 0.0),
                "scorecard":   trade.get("scorecard", {}),
                "mc_result":   trade.get("mc_result", {}),
            })

    def record_regime(self, regime_snapshot: Dict[str, Any]):
        """Record regime classification snapshot for digest."""
        with self._lock:
            self._regime_log.append({
                "ts":     time.time(),
                "regime": regime_snapshot.get("regime", "?"),
                "inputs": regime_snapshot.get("last_inputs", {}),
            })
            # Keep only last 1000 regime logs
            if len(self._regime_log) > 1000:
                self._regime_log = self._regime_log[-1000:]

    def record_violation(self, rule: str, detail: str):
        """Record a constitutional violation."""
        with self._lock:
            self._violations.append({
                "ts":     time.time(),
                "rule":   rule,
                "detail": detail,
            })

    def record_health_warning(self, warning: str):
        """Record a system health warning."""
        with self._lock:
            self._health_warnings.append({"ts": time.time(), "warning": warning})

    def record_mc_portfolio_state(self, mc_state: Dict[str, Any]):
        """Record latest portfolio MC metrics (called periodically)."""
        with self._lock:
            self._mc_state = mc_state

    def set_end_equity(self, equity: float):
        with self._lock:
            self._end_equity = equity

    # ─── Scheduling ───────────────────────────────────────────────────────────

    def schedule_daily(self):
        """
        Schedule digest generation at 00:00 UTC daily.
        Starts a recurring background timer.
        """
        self._schedule_next()
        logger.info("[Digest] Daily digest scheduler started")

    def _schedule_next(self):
        """Calculate seconds until next 00:00 UTC and set timer."""
        now_utc   = datetime.now(timezone.utc)
        tomorrow  = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        # Advance to NEXT midnight (if we're past midnight already)
        from datetime import timedelta
        tomorrow += timedelta(days=1)
        wait_sec  = (tomorrow - now_utc).total_seconds()

        self._timer = threading.Timer(wait_sec, self._generate_and_reschedule)
        self._timer.daemon = True
        self._timer.start()
        logger.info("[Digest] Next digest scheduled in %.1fh", wait_sec / 3600)

    def _generate_and_reschedule(self):
        """Called by timer: generate digest then schedule the next one."""
        try:
            self.generate_now()
        except Exception as e:
            logger.error("[Digest] Error generating digest: %s", e)
        finally:
            self._schedule_next()

    def stop(self):
        """Cancel pending timer."""
        if self._timer:
            self._timer.cancel()

    # ─── Digest Generation ────────────────────────────────────────────────────

    def generate_now(self, end_equity: Optional[float] = None) -> str:
        """
        Generate and persist today's digest. Returns the formatted text.
        Can be called manually (e.g., for testing).
        """
        with self._lock:
            if end_equity is not None:
                self._end_equity = end_equity
            text = self._format_digest()
            self._persist(text)
            if self.telegram_fn:
                try:
                    self.telegram_fn(text[:4096])   # Telegram 4096 char limit
                except Exception as e:
                    logger.error("[Digest] Telegram send error: %s", e)
            # FIX 2026-03-21: Carry over end equity as next day's start baseline
            # Without this, _reset_accumulators sets _start_equity=0.0, and the
            # digest shows $0.00 until the first trading cycle sets it again.
            carry_over_equity = self._end_equity
            self._reset_accumulators()
            if carry_over_equity > 0:
                self._start_equity = carry_over_equity

        return text

    def _format_digest(self) -> str:
        now     = datetime.now(timezone.utc)
        dstr    = now.strftime("%Y-%m-%d")
        start_e = self._start_equity
        end_e   = self._end_equity if self._end_equity > 0 else start_e
        delta_e = end_e - start_e
        delta_p = (delta_e / start_e * 100) if start_e > 0 else 0.0

        lines = [
            "=" * 62,
            f"  SMCE DAILY DIGEST – {dstr} UTC",
            "=" * 62,
            "",
            "── EQUITY ──────────────────────────────────────────────────",
            f"  Start:  ${start_e:.2f}",
            f"  End:    ${end_e:.2f}",
            f"  Change: {'+' if delta_p >= 0 else ''}{delta_p:.2f}%  (${delta_e:+.2f})",
            "",
        ]

        # ── Trades ────────────────────────────────────────────────────────────
        lines.append("── TRADES ──────────────────────────────────────────────────")
        if not self._trades:
            lines.append("  No trades executed today.")
        else:
            total_pnl = 0.0
            for t in self._trades:
                ts_str = datetime.fromtimestamp(t["ts"], tz=timezone.utc).strftime("%H:%M")
                score  = t.get("scorecard", {}).get("score", "?")
                pnl    = t["pnl_usd"]
                total_pnl += pnl if isinstance(pnl, (int, float)) else 0.0
                lines.append(
                    f"  [{ts_str}] {t['direction']:4s} {t['symbol']:12s} "
                    f"${t['notional']:.2f} | "
                    f"in={t['entry_price']:.4f} out={t['exit_price']:.4f} | "
                    f"P&L={pnl:+.2f} | score={score}"
                )
            lines.append(f"  Total P&L: ${total_pnl:+.2f}  ({len(self._trades)} trades)")
        lines.append("")

        # ── Regime ────────────────────────────────────────────────────────────
        lines.append("── REGIME ──────────────────────────────────────────────────")
        if self._regime_log:
            regime_counts: Dict[str, int] = {}
            for r in self._regime_log:
                regime_counts[r["regime"]] = regime_counts.get(r["regime"], 0) + 1
            dominant = max(regime_counts, key=regime_counts.get)
            lines.append(f"  Dominant: {dominant}")
            for k, v in sorted(regime_counts.items()):
                pct = v / len(self._regime_log) * 100
                lines.append(f"    {k:12s}: {pct:.0f}% of cycles")
            # Last inputs
            if self._regime_log:
                last = self._regime_log[-1]["inputs"]
                lines.append(f"  Last inputs: {json.dumps(last)}")
        else:
            lines.append("  No regime data recorded.")
        lines.append("")

        # ── Scorecards ────────────────────────────────────────────────────────
        lines.append("── PROBABILITY SCORECARDS ──────────────────────────────────")
        scored_trades = [t for t in self._trades if t.get("scorecard")]
        if scored_trades:
            for t in scored_trades[:10]:  # limit to 10 in digest
                sc  = t["scorecard"]
                bd  = sc.get("breakdown", {})
                lines.append(
                    f"  {t['direction']:4s} {t['symbol']:12s} "
                    f"score={sc.get('score','?'):4}  "
                    f"struct={bd.get('structure','-')} mom={bd.get('momentum','-')} "
                    f"liq={bd.get('liquidity','-')} ent={bd.get('entropy','-')} "
                    f"corr={bd.get('correlation','-')} cvar={bd.get('cvar','-')}"
                )
        else:
            lines.append("  No scorecard data.")
        lines.append("")

        # ── Monte Carlo ───────────────────────────────────────────────────────
        lines.append("── MONTE CARLO SUMMARY ─────────────────────────────────────")
        mc = self._mc_state
        if mc:
            lines.append(f"  Portfolio CVaR(95%): {mc.get('cvar_95', 0)*100:.2f}%")
            lines.append(f"  P(drawdown>5%/24h): {mc.get('drawdown_prob', 0)*100:.1f}%")
            lines.append(f"  Veto count today:   {mc.get('veto_count', 0)}")
            lines.append(f"  Approve count:      {mc.get('approve_count', 0)}")
        else:
            lines.append("  No MC data available.")
        lines.append("")

        # ── Violations ────────────────────────────────────────────────────────
        lines.append("── CONSTITUTIONAL VIOLATIONS ───────────────────────────────")
        if not self._violations:
            lines.append("  ✅ Zero violations – all rules respected.")
        else:
            for v in self._violations:
                ts_str = datetime.fromtimestamp(v["ts"], tz=timezone.utc).strftime("%H:%M")
                lines.append(f"  ⚠️ [{ts_str}] {v['rule']}: {v['detail']}")
        lines.append("")

        # ── System Health ─────────────────────────────────────────────────────
        lines.append("── SYSTEM HEALTH ───────────────────────────────────────────")
        if not self._health_warnings:
            lines.append("  ✅ No warnings.")
        else:
            for w in self._health_warnings[-10:]:
                ts_str = datetime.fromtimestamp(w["ts"], tz=timezone.utc).strftime("%H:%M")
                lines.append(f"  ⚠️ [{ts_str}] {w['warning']}")
        lines.append("")
        lines.append("=" * 62)

        return "\n".join(lines)

    def _persist(self, text: str):
        """Write digest to log file."""
        date_str  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        filename  = self.log_dir / f"smce_digest_{date_str}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info("[Digest] Written to %s", filename)

    def _reset_accumulators(self):
        self._start_equity:   float = 0.0
        self._end_equity:     float = 0.0
        self._trades:         list  = []
        self._regime_log:     list  = []
        self._violations:     list  = []
        self._health_warnings:list  = []
        self._mc_state:       Dict  = {}
