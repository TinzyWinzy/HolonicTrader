"""
FlexlineAgent Unit Tests

Tests for Kraken Flexline Credit Facility Management.
Run with: python -m pytest HolonicTrader/tests/test_flexline.py -v
"""

import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from HolonicTrader.agent_flexline import FlexlineAgent
import config


class TestFlexlineAgentInit:
    """Test FlexlineAgent initialization."""

    def test_init_default(self):
        """Test default initialization."""
        agent = FlexlineAgent()
        
        assert agent.name == "FlexlineAgent"
        assert agent.enabled == getattr(config, 'FLEXLINE_ENABLED', False)
        assert agent.credit_limit == 0.0
        assert agent.utilized == 0.0
        assert agent.available_credit == 0.0
        assert agent.max_utilization == 0.50
        assert agent.emergency_reserve == 0.20
        assert agent.liquidation_ltv == 0.80
        assert agent.ltv_warning_threshold == 0.65

    def test_init_with_custom_name(self):
        """Test initialization with custom name."""
        agent = FlexlineAgent(name="TestFlexline")
        assert agent.name == "TestFlexline"

    def test_collateral_ltv_config(self):
        """Test collateral LTV configuration."""
        agent = FlexlineAgent()
        
        assert 'BTC' in agent.collateral_ltv
        assert 'ETH' in agent.collateral_ltv
        assert agent.collateral_ltv['BTC'] == 0.70
        assert agent.collateral_ltv['ETH'] == 0.65


class TestFlexlineAgentParsing:
    """Test data parsing utilities."""

    def test_parse_float_from_string(self):
        """Test parsing float from string."""
        agent = FlexlineAgent()
        
        assert agent._parse_float("100.50") == 100.50
        assert agent._parse_float("0") == 0.0
        assert agent._parse_float("") == 0.0
        assert agent._parse_float(None) == 0.0

    def test_parse_float_from_number(self):
        """Test parsing float from number."""
        agent = FlexlineAgent()
        
        assert agent._parse_float(100) == 100.0
        assert agent._parse_float(100.50) == 100.50
        assert agent._parse_float(0) == 0.0

    def test_parse_float_invalid(self):
        """Test parsing invalid float."""
        agent = FlexlineAgent()
        
        assert agent._parse_float("invalid") == 0.0
        assert agent._parse_float("abc123") == 0.0


class TestFlexlineAgentStatus:
    """Test status and health checks."""

    def test_get_status_disabled(self):
        """Test status when disabled."""
        agent = FlexlineAgent()
        agent.enabled = False
        
        assert agent.get_status() == 'DISABLED'

    def test_get_status_healthy(self):
        """Test healthy status."""
        agent = FlexlineAgent()
        agent.enabled = True
        agent.current_ltv = 0.30  # 30% utilization
        
        assert agent.get_status() == 'HEALTHY'

    def test_get_status_warning(self):
        """Test warning status."""
        agent = FlexlineAgent()
        agent.enabled = True
        agent.current_ltv = 0.70  # Above warning threshold
        
        assert agent.get_status() == 'WARNING'

    def test_get_status_critical(self):
        """Test critical status."""
        agent = FlexlineAgent()
        agent.enabled = True
        agent.current_ltv = 0.85  # Above liquidation threshold
        
        assert agent.get_status() == 'CRITICAL'

    def test_is_healthy(self):
        """Test health check."""
        agent = FlexlineAgent()
        agent.enabled = True
        agent.current_ltv = 0.30
        agent.last_sync = __import__('time').time()
        
        assert agent.is_healthy() == True

    def test_is_healthy_disabled(self):
        """Test health check when disabled."""
        agent = FlexlineAgent()
        agent.enabled = False
        
        assert agent.is_healthy() == False


class TestFlexlineAgentInterest:
    """Test interest calculations."""

    def test_get_daily_interest_rate(self):
        """Test daily interest rate calculation."""
        agent = FlexlineAgent()
        agent.interest_rate_hourly = 0.0001
        
        daily_rate = agent.get_daily_interest_rate()
        assert daily_rate == 0.24  # 0.0001 * 24 * 100

    def test_get_annual_interest_rate(self):
        """Test annual interest rate calculation."""
        agent = FlexlineAgent()
        agent.interest_rate_hourly = 0.0001
        
        annual_rate = agent.get_annual_interest_rate()
        assert annual_rate == 87.6  # 0.0001 * 24 * 365 * 100

    def test_get_interest_cost(self):
        """Test interest cost calculation."""
        agent = FlexlineAgent()
        agent.utilized = 1000.0
        agent.interest_rate_hourly = 0.0001
        
        cost_24h = agent.get_interest_cost(24)
        assert cost_24h == 2.4  # 1000 * 0.0001 * 24


class TestFlexlineAgentLiquidation:
    """Test liquidation risk monitoring."""

    def test_check_liquidation_risk_low(self):
        """Test low liquidation risk."""
        agent = FlexlineAgent()
        agent.current_ltv = 0.30
        agent.utilized = 30.0
        agent.credit_limit = 100.0
        
        risk = agent.check_liquidation_risk()
        
        assert risk['risk_level'] == 'LOW'
        assert risk['current_ltv'] == 0.30
        assert risk['distance_to_liquidation'] == 0.50

    def test_check_liquidation_risk_critical(self):
        """Test critical liquidation risk."""
        agent = FlexlineAgent()
        agent.current_ltv = 0.85
        agent.utilized = 85.0
        agent.credit_limit = 100.0
        
        risk = agent.check_liquidation_risk()
        
        assert risk['risk_level'] == 'CRITICAL'
        assert risk['emergency_action_required'] == True

    def test_check_liquidation_risk_high(self):
        """Test high liquidation risk."""
        agent = FlexlineAgent()
        agent.current_ltv = 0.70
        agent.utilized = 70.0
        agent.credit_limit = 100.0
        
        risk = agent.check_liquidation_risk()
        
        assert risk['risk_level'] == 'HIGH'


class TestFlexlineAgentAllocation:
    """Test capital allocation optimization."""

    def test_get_available_for_trading(self):
        """Test available credit for trading."""
        agent = FlexlineAgent()
        agent.credit_limit = 1000.0
        agent.utilized = 200.0
        agent.available_credit = 800.0
        agent.emergency_reserve = 0.20
        
        available = agent.get_available_for_trading()
        
        # Should subtract emergency reserve (20% of 1000 = 200)
        assert available == 600.0  # 800 - 200

    def test_get_available_for_trading_no_reserve(self):
        """Test available credit with no reserve needed."""
        agent = FlexlineAgent()
        agent.credit_limit = 100.0
        agent.utilized = 0.0
        agent.available_credit = 100.0
        agent.emergency_reserve = 0.20
        
        available = agent.get_available_for_trading()
        
        assert available == 80.0  # 100 - 20 (reserve)

    def test_optimize_allocation(self):
        """Test allocation optimization."""
        agent = FlexlineAgent()
        agent.enabled = True
        agent.available_credit = 500.0
        agent.min_net_apy = 50.0
        agent.interest_rate_hourly = 0.0001
        
        opportunities = [
            {'symbol': 'BTC/USDT', 'apy': 200.0, 'max_size': 300.0, 'equity': 1000.0},
            {'symbol': 'ETH/USDT', 'apy': 100.0, 'max_size': 400.0, 'equity': 1000.0},
            {'symbol': 'SOL/USDT', 'apy': 30.0, 'max_size': 200.0, 'equity': 1000.0},  # Below min_net_apy
        ]
        
        allocated = agent.optimize_allocation(opportunities)
        
        # Should allocate to highest APY first, skip below threshold
        assert len(allocated) >= 1
        assert allocated[0]['symbol'] == 'BTC/USDT'  # Highest APY


class TestFlexlineAgentMockAPI:
    """Test with mocked API calls."""

    @patch('HolonicTrader.agent_flexline.ccxt.krakenfutures')
    def test_sync_credit_line_mock(self, mock_ccxt):
        """Test credit line sync with mocked API."""
        # Setup mock
        mock_exchange = MagicMock()
        mock_ccxt.return_value = mock_exchange
        mock_exchange.fetch_balance.return_value = {
            'info': {
                'accounts': {
                    'flexCredit': {
                        'creditLimit': '1000.0',
                        'utilized': '200.0',
                        'interestRate': {'hourly': '0.0001'}
                    }
                }
            }
        }
        
        agent = FlexlineAgent()
        agent.enabled = True
        
        # Mock the methods that would fail without real API
        agent._calculate_credit_from_collateral = MagicMock()
        agent._update_collateral_portfolio = MagicMock()
        agent._calculate_ltv = MagicMock(return_value=None)
        agent._update_dashboard_state = MagicMock()
        
        result = agent.sync_credit_line(force=True)
        
        assert result == True
        assert agent.credit_limit == 1000.0
        assert agent.utilized == 200.0
        assert agent.available_credit >= 0

    @patch('HolonicTrader.agent_flexline.ccxt.krakenfutures')
    def test_sync_credit_line_failure(self, mock_ccxt):
        """Test credit line sync failure handling."""
        mock_ccxt.side_effect = Exception("API Error")
        
        agent = FlexlineAgent()
        agent.enabled = True
        
        result = agent.sync_credit_line(force=True)
        
        assert result == False
        assert 'last_error' in agent._dashboard_state


class TestFlexlineAgentEnableDisable:
    """Test enable/disable functionality."""

    def test_enable(self):
        """Test enabling Flexline."""
        agent = FlexlineAgent()
        agent.enabled = False
        
        agent.enable()
        
        assert agent.enabled == True

    def test_disable(self):
        """Test disabling Flexline."""
        agent = FlexlineAgent()
        agent.enabled = True
        
        agent.disable()
        
        assert agent.enabled == False


class TestFlexlineAgentBorrowRepay:
    """Test borrow and repay logic (without actual API calls)."""

    def test_borrow_disabled(self):
        """Test borrow when disabled."""
        agent = FlexlineAgent()
        agent.enabled = False
        
        success, msg = agent.borrow(100.0)
        
        assert success == False
        assert "disabled" in msg.lower()

    def test_repay_disabled(self):
        """Test repay when disabled."""
        agent = FlexlineAgent()
        agent.enabled = False
        
        success, msg = agent.repay(100.0)
        
        assert success == False
        assert "disabled" in msg.lower()

    def test_borrow_exceeds_available(self):
        """Test borrow exceeding available credit."""
        agent = FlexlineAgent()
        agent.enabled = True
        agent.available_credit = 100.0
        
        success, msg = agent.borrow(200.0)
        
        assert success == False
        assert "exceeds" in msg.lower()

    def test_repay_no_balance(self):
        """Test repay with no balance."""
        agent = FlexlineAgent()
        agent.enabled = True
        agent.utilized = 0.0
        
        success, msg = agent.repay(100.0)
        
        assert success == False
        assert "no outstanding" in msg.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
