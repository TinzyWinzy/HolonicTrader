"""
FlexlineAgent - Kraken Flexline Credit Facility Manager (Production v1.0)

Responsibilities:
- Monitor available credit in real-time
- Track interest costs and accrual
- Optimize collateral allocation
- Emergency liquidity management
- LTV monitoring and liquidation protection
- Auto-repay functionality

Integration Points:
- GovernorHolon: Position sizing with credit boost
- ArbitrageHolon: Capital allocation enhancement
- KrakenHolon: Balance and account health
- ExecutorHolon: Trade execution with Flexline backing
"""

import time
import ccxt
import config
import logging
from typing import Dict, List, Any, Optional, Tuple
from HolonicTrader.holon_core import Holon, Disposition, Message

logger = logging.getLogger("HolonicTrader.FlexlineAgent")

class FlexlineAgent(Holon):
    """
    Manages Kraken Flexline credit facility for HolonicTrader.
    
    Features:
    - Real-time credit line monitoring
    - Collateral LTV calculation
    - Interest cost tracking
    - Borrow/repay automation
    - Liquidation risk alerts
    - Strategic capital allocation
    """

    def __init__(self, name: str = "FlexlineAgent"):
        super().__init__(name=name, disposition=Disposition(autonomy=0.6, integration=0.9))

        # Initialize Kraken Futures API
        self.kraken = ccxt.krakenfutures({
            'apiKey': config.KRAKEN_FUTURES_API_KEY or config.API_KEY,
            'secret': config.KRAKEN_FUTURES_PRIVATE_KEY or config.API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'
            }
        })

        # Credit Line State
        self.credit_limit: float = 0.0
        self.utilized: float = 0.0
        self.available_credit: float = 0.0
        self.interest_rate_hourly: float = 0.0001  # Default ~0.87% daily

        # Collateral Configuration
        self.collateral_ltv: Dict[str, float] = getattr(config, 'FLEXLINE_COLLATERAL_LTV', {
            'BTC': 0.70,
            'ETH': 0.65,
            'USDT': 0.90,
            'USDC': 0.90,
            'XBT': 0.70,  # Kraken uses XBT for Bitcoin
        })

        # Risk Parameters
        self.max_utilization: float = getattr(config, 'FLEXLINE_MAX_UTILIZATION', 0.50)
        self.emergency_reserve: float = getattr(config, 'FLEXLINE_EMERGENCY_RESERVE', 0.20)
        self.min_net_apy: float = getattr(config, 'FLEXLINE_MIN_NET_APY', 50.0)
        self.max_hourly_rate: float = getattr(config, 'FLEXLINE_MAX_HOURLY_RATE', 0.0002)

        # LTV Monitoring
        self.current_ltv: float = 0.0
        self.liquidation_ltv: float = 0.80  # Kraken typical liquidation threshold
        self.ltv_warning_threshold: float = 0.65  # Alert before liquidation

        # Collateral Portfolio
        self.collateral_portfolio: Dict[str, float] = {}
        self.collateral_values_usd: Dict[str, float] = {}

        # Sync State
        self.last_sync: float = 0
        self.sync_interval: int = 60  # Sync every 60 seconds
        self.last_interest_calc: float = 0
        self.interest_accrued: float = 0.0

        # Emergency Loan Tracking
        self.emergency_loans: List[Dict[str, Any]] = []
        self.repayment_reserve: float = 0.0

        # Enabled Flag
        self.enabled: bool = getattr(config, 'FLEXLINE_ENABLED', False)

        # Dashboard State
        self._dashboard_state: Dict[str, Any] = {}

        logger.info(f"[{self.name}] 🏦 FlexlineAgent initialized (Enabled: {self.enabled})")

    def receive_message(self, sender: Any, content: Any) -> None:
        """Handle incoming messages from other holons."""
        if not isinstance(content, dict):
            return

        msg_type = content.get('type', '')

        if msg_type == 'SYNC_FLEXLINE':
            self.sync_credit_line(force=True)
        elif msg_type == 'BORROW_REQUEST':
            amount = content.get('amount', 0)
            purpose = content.get('purpose', 'trading')
            self.borrow(amount, purpose)
        elif msg_type == 'REPAY_REQUEST':
            amount = content.get('amount', 0)
            self.repay(amount)
        elif msg_type == 'CHECK_AVAILABLE_CREDIT':
            self.sync_credit_line()

    def sync_credit_line(self, force: bool = False) -> bool:
        """
        Fetch current Flexline status from Kraken Futures API.
        
        Returns:
            bool: True if sync successful
        """
        now = time.time()
        if not force and now - self.last_sync < self.sync_interval:
            return True

        if not self.enabled:
            return False

        try:
            balance = self.kraken.fetch_balance()
            info = balance.get('info', {})
            accounts = info.get('accounts', {})

            # Flexline/Flex account
            flex = accounts.get('flex', {})
            flex_credit = accounts.get('flexCredit', {})  # Alternative name

            # Use flexCredit if available, otherwise flex
            if flex_credit and flex_credit.get('creditLimit'):
                self.credit_limit = self._parse_float(flex_credit.get('creditLimit', 0.0))
                self.utilized = self._parse_float(flex_credit.get('utilized', 0.0))
                interest_info = flex_credit.get('interestRate', {})
                if isinstance(interest_info, dict):
                    self.interest_rate_hourly = self._parse_float(interest_info.get('hourly', 0.0001))
            else:
                # Fallback: calculate from flex account
                margin_equity = self._parse_float(flex.get('marginEquity', 0.0))
                used_margin = self._parse_float(flex.get('usedMargin', 0.0))
                available_margin = self._parse_float(flex.get('availableMargin', 0.0))

                # Estimate credit limit based on collateral
                self._calculate_credit_from_collateral(flex)

            # Calculate available credit
            self.available_credit = self.credit_limit - self.utilized

            # Apply max utilization cap
            max_allowed = self.credit_limit * self.max_utilization
            self.available_credit = min(self.available_credit, max_allowed - self.utilized)

            # Update collateral portfolio
            self._update_collateral_portfolio(flex)

            # Calculate current LTV
            self._calculate_ltv()

            self.last_sync = now

            # Update dashboard state
            self._update_dashboard_state()

            logger.info(
                f"[{self.name}] 💳 Flexline Sync: "
                f"Limit=${self.credit_limit:.2f}, "
                f"Used=${self.utilized:.2f}, "
                f"Available=${self.available_credit:.2f}, "
                f"LTV={self.current_ltv:.1%}"
            )

            return True

        except Exception as e:
            logger.error(f"[{self.name}] ⚠️ Sync failed: {e}")
            self._dashboard_state['last_error'] = str(e)
            return False

    def _parse_float(self, value: Any, default: float = 0.0) -> float:
        """Safely parse float from API response (handles strings)."""
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and value.strip():
            try:
                return float(value)
            except ValueError:
                return default
        return default

    def _calculate_credit_from_collateral(self, flex: Dict[str, Any]) -> None:
        """Calculate credit limit based on collateral LTV."""
        total_collateral_usd = 0.0

        # Get collateral balance
        balance_value = self._parse_float(flex.get('balanceValue', 0.0))
        wallet_balance = self._parse_float(flex.get('walletBalance', 0.0))

        # Use the larger of the two
        total_collateral_usd = max(balance_value, wallet_balance)

        # Apply average LTV
        avg_ltv = sum(self.collateral_ltv.values()) / len(self.collateral_ltv)
        self.credit_limit = total_collateral_usd * avg_ltv

    def _update_collateral_portfolio(self, flex: Dict[str, Any]) -> None:
        """Update internal collateral portfolio tracking."""
        # Kraken Futures doesn't provide detailed collateral breakdown via API
        # We track total value and assume diversified collateral
        balance_value = self._parse_float(flex.get('balanceValue', 0.0))

        # Simplified: assume BTC/ETH mix
        self.collateral_portfolio = {
            'BTC': 0.6,  # 60% BTC
            'ETH': 0.4,  # 40% ETH
        }

        self.collateral_values_usd = {
            'BTC': balance_value * 0.6,
            'ETH': balance_value * 0.4,
        }

    def _calculate_ltv(self) -> None:
        """Calculate current Loan-to-Value ratio."""
        if self.credit_limit > 0:
            self.current_ltv = self.utilized / self.credit_limit
        else:
            self.current_ltv = 0.0

    def _update_dashboard_state(self) -> None:
        """Update dashboard state for monitoring."""
        self._dashboard_state = {
            'enabled': self.enabled,
            'credit_limit': round(self.credit_limit, 2),
            'utilized': round(self.utilized, 2),
            'available_credit': round(self.available_credit, 2),
            'utilization_pct': round(self.current_ltv * 100, 2),
            'interest_rate_hourly': self.interest_rate_hourly,
            'interest_rate_daily_pct': round(self.interest_rate_hourly * 24 * 100, 3),
            'ltv_pct': round(self.current_ltv * 100, 2),
            'liquidation_ltv_pct': self.liquidation_ltv * 100,
            'warning_ltv_pct': self.ltv_warning_threshold * 100,
            'emergency_loans_count': len(self.emergency_loans),
            'repayment_reserve': round(self.repayment_reserve, 2),
            'last_sync': self.last_sync,
            'status': self.get_status(),
        }

    def get_status(self) -> str:
        """Get current Flexline status."""
        if not self.enabled:
            return 'DISABLED'
        if self.current_ltv >= self.liquidation_ltv:
            return 'CRITICAL'
        if self.current_ltv >= self.ltv_warning_threshold:
            return 'WARNING'
        if self.current_ltv >= self.max_utilization:
            return 'HIGH_UTILIZATION'
        return 'HEALTHY'

    def calculate_collateral_value(self, prices: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate total borrowable value of collateral portfolio.
        
        Args:
            prices: Optional dict of asset prices (fetches from API if not provided)
        
        Returns:
            Total borrowable USD value
        """
        if prices is None:
            prices = self._fetch_collateral_prices()

        total_borrowable = 0.0

        for asset, weight in self.collateral_portfolio.items():
            if asset in prices:
                ltv = self.collateral_ltv.get(asset, 0.5)
                price = prices[asset]
                # Assume portfolio value is proportional to weight
                asset_value = self.credit_limit / 0.5 * weight if self.credit_limit > 0 else 0
                borrowable = asset_value * ltv
                total_borrowable += borrowable

        return total_borrowable

    def _fetch_collateral_prices(self) -> Dict[str, float]:
        """Fetch current prices for collateral assets."""
        prices = {}
        try:
            # Fetch BTC price
            btc_ticker = self.kraken.fetch_ticker('BTC/USDT')
            prices['BTC'] = prices['XBT'] = btc_ticker.get('last', 0)

            # Fetch ETH price
            eth_ticker = self.kraken.fetch_ticker('ETH/USDT')
            prices['ETH'] = eth_ticker.get('last', 0)

            # Stablecoins at $1
            prices['USDT'] = prices['USDC'] = 1.0
        except Exception as e:
            logger.warning(f"[{self.name}] ⚠️ Price fetch failed: {e}")
        return prices

    def borrow(self, amount: float, purpose: str = 'trading') -> Tuple[bool, str]:
        """
        Utilize Flexline credit.
        
        Args:
            amount: USD amount to borrow
            purpose: Purpose of borrow ('trading', 'emergency', 'arb')
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.enabled:
            return False, "Flexline is disabled"

        # Sync first
        self.sync_credit_line()

        # Check available credit
        if amount > self.available_credit:
            msg = f"Borrow request ${amount:.2f} exceeds available ${self.available_credit:.2f}"
            logger.warning(f"[{self.name}] ⚠️ {msg}")
            return False, msg

        # Check interest rate
        if self.interest_rate_hourly > self.max_hourly_rate:
            msg = f"Interest rate {self.interest_rate_hourly:.5f} exceeds max {self.max_hourly_rate:.5f}"
            logger.warning(f"[{self.name}] ⚠️ {msg}")
            return False, msg

        # Check LTV after borrow
        projected_ltv = (self.utilized + amount) / self.credit_limit if self.credit_limit > 0 else 1.0
        if projected_ltv > self.max_utilization:
            msg = f"Borrow would push LTV to {projected_ltv:.1%}, exceeds max {self.max_utilization:.1%}"
            logger.warning(f"[{self.name}] ⚠️ {msg}")
            return False, msg

        try:
            # Execute transfer from flex credit to margin
            # Note: Kraken API transfer syntax may vary
            self.kraken.transfer('flex', 'margin', amount, currency='USDC')

            # Update internal state
            self.utilized += amount
            self.available_credit -= amount

            # Track emergency loans separately
            if purpose == 'emergency':
                self.emergency_loans.append({
                    'amount': amount,
                    'timestamp': time.time(),
                    'repay_by': time.time() + 86400,  # 24 hours
                    'status': 'active'
                })
                logger.warning(f"[{self.name}] 🚨 EMERGENCY LOAN: ${amount:.2f} - Must repay within 24h")

            logger.info(f"[{self.name}] ✅ Borrowed ${amount:.2f} via Flexline for {purpose}")
            self._update_dashboard_state()
            return True, f"Successfully borrowed ${amount:.2f}"

        except Exception as e:
            msg = f"Borrow failed: {e}"
            logger.error(f"[{self.name}] ❌ {msg}")
            return False, msg

    def repay(self, amount: float) -> Tuple[bool, str]:
        """
        Repay Flexline credit.
        
        Args:
            amount: USD amount to repay
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.enabled:
            return False, "Flexline is disabled"

        # Sync first
        self.sync_credit_line()

        # Can't repay more than utilized
        if amount > self.utilized:
            amount = self.utilized

        if amount <= 0:
            return False, "No outstanding balance to repay"

        try:
            # Execute transfer from margin to flex
            self.kraken.transfer('margin', 'flex', amount, currency='USDC')

            # Update internal state
            self.utilized -= amount
            self.available_credit += amount

            # Remove from emergency loans if applicable
            now = time.time()
            for loan in self.emergency_loans[:]:
                if loan['status'] == 'active':
                    if loan['amount'] <= amount:
                        loan['status'] = 'repaid'
                        loan['repaid_at'] = now
                        logger.info(f"[{self.name}] ✅ Emergency loan ${loan['amount']:.2f} repaid")

            logger.info(f"[{self.name}] ✅ Repaid ${amount:.2f} to Flexline")
            self._update_dashboard_state()
            return True, f"Successfully repaid ${amount:.2f}"

        except Exception as e:
            msg = f"Repay failed: {e}"
            logger.error(f"[{self.name}] ❌ {msg}")
            return False, msg

    def get_interest_cost(self, duration_hours: float = 24) -> float:
        """
        Calculate interest cost for current utilization.
        
        Args:
            duration_hours: Duration in hours
        
        Returns:
            Interest cost in USD
        """
        return self.utilized * self.interest_rate_hourly * duration_hours

    def get_daily_interest_rate(self) -> float:
        """Get daily interest rate as percentage."""
        return self.interest_rate_hourly * 24 * 100

    def get_annual_interest_rate(self) -> float:
        """Get annual interest rate (APY) as percentage."""
        return self.interest_rate_hourly * 24 * 365 * 100

    def check_liquidation_risk(self) -> Dict[str, Any]:
        """
        Check current liquidation risk status.
        
        Returns:
            Dict with risk assessment
        """
        distance_to_liquidation = self.liquidation_ltv - self.current_ltv
        distance_to_warning = self.ltv_warning_threshold - self.current_ltv

        risk_level = 'LOW'
        if self.current_ltv >= self.liquidation_ltv:
            risk_level = 'CRITICAL'
        elif self.current_ltv >= self.ltv_warning_threshold:
            risk_level = 'HIGH'
        elif self.current_ltv >= self.max_utilization:
            risk_level = 'MEDIUM'

        return {
            'risk_level': risk_level,
            'current_ltv': self.current_ltv,
            'liquidation_ltv': self.liquidation_ltv,
            'distance_to_liquidation': distance_to_liquidation,
            'distance_to_warning': distance_to_warning,
            'utilized': self.utilized,
            'credit_limit': self.credit_limit,
            'emergency_action_required': risk_level == 'CRITICAL',
        }

    def auto_repay_if_needed(self, executor_balance: float) -> Tuple[bool, str]:
        """
        Automatically repay if LTV is dangerously high.
        
        Args:
            executor_balance: Available balance from ExecutorHolon
        
        Returns:
            Tuple of (action_taken: bool, message: str)
        """
        risk = self.check_liquidation_risk()

        if risk['risk_level'] not in ['CRITICAL', 'HIGH']:
            return False, "No auto-repay needed"

        # Calculate repay amount (use emergency reserve first)
        target_ltv = self.max_utilization * 0.8  # Target 80% of max
        target_utilized = self.credit_limit * target_ltv
        repay_amount = self.utilized - target_utilized

        # Cap at available balance
        repay_amount = min(repay_amount, executor_balance)

        if repay_amount <= 0:
            return False, "Insufficient balance for auto-repay"

        logger.warning(f"[{self.name}] 🚨 AUTO-REPAY TRIGGERED: ${repay_amount:.2f} (LTV: {self.current_ltv:.1%})")
        return self.repay(repay_amount)

    def optimize_allocation(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Allocate Flexline credit to highest risk-adjusted returns.
        
        Args:
            opportunities: List of trading opportunities with 'apy', 'max_size', 'equity'
        
        Returns:
            List of allocated opportunities with sizes
        """
        if not self.enabled or self.available_credit <= 0:
            return []

        # Calculate net APY for each opportunity
        annual_interest = self.get_annual_interest_rate()
        for opp in opportunities:
            opp['net_apy'] = opp.get('apy', 0) - annual_interest

        # Sort by net APY descending
        opportunities.sort(key=lambda x: x.get('net_apy', 0), reverse=True)

        allocated = []
        remaining = self.available_credit

        for opp in opportunities:
            if remaining <= 0:
                break

            # Skip if net APY below threshold
            if opp.get('net_apy', 0) < self.min_net_apy:
                continue

            # Calculate optimal size
            max_size = opp.get('max_size', float('inf'))
            equity_limit = opp.get('equity', 0) * 0.25  # Max 25% of equity per position
            size = min(remaining, max_size, equity_limit)

            if size > 0:
                allocated.append({
                    **opp,
                    'allocated_size': size,
                    'flexline_backed': True
                })
                remaining -= size

        return allocated

    def get_available_for_trading(self) -> float:
        """
        Get available credit for new trades (after reserves).
        
        Returns:
            Available USD for trading
        """
        # Keep emergency reserve
        reserve = self.credit_limit * self.emergency_reserve
        trading_available = self.available_credit - reserve
        return max(0, trading_available)

    def get_dashboard_state(self) -> Dict[str, Any]:
        """Return dashboard state for monitoring."""
        return self._dashboard_state.copy()

    def enable(self) -> None:
        """Enable Flexline integration."""
        self.enabled = True
        self.sync_credit_line()
        logger.info(f"[{self.name}] ✅ Flexline ENABLED")

    def disable(self) -> None:
        """Disable Flexline integration."""
        self.enabled = False
        logger.info(f"[{self.name}] ⏸️ Flexline DISABLED")

    def is_healthy(self) -> bool:
        """Check if Flexline is in healthy state."""
        return (
            self.enabled and
            self.get_status() == 'HEALTHY' and
            self.last_sync > time.time() - 300  # Synced within 5 min
        )
