"""
NEXUS Configuration (Phase 15) - NANO MODE EDITION

CENTRAL STORAGE for all thresholds, leverage caps, and system parameters.
UPDATED 2026-01-13 for NANO accounts (< $50)
FIXED: Position sizing, leverage, and margin calculations for $25 account
"""

from dotenv import load_dotenv
import os

load_dotenv()

KRAKEN_FUTURES_API_KEY = os.getenv('KRAKEN_FUTURES_API_KEY')
KRAKEN_FUTURES_PRIVATE_KEY = os.getenv('KRAKEN_FUTURES_PRIVATE_KEY')
KRAKEN_SPOT_KEY = os.getenv('KRAKEN_API_KEY')
KRAKEN_SPOT_SECRET = os.getenv('KRAKEN_PRIVATE_KEY')

API_KEY = KRAKEN_FUTURES_API_KEY or KRAKEN_SPOT_KEY
API_SECRET = KRAKEN_FUTURES_PRIVATE_KEY or KRAKEN_SPOT_SECRET

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
TELEGRAM_ENABLED = True

# Estimated execution costs used for sizing, simulation and decision thresholds
# These can be overridden via environment variables for live tuning.
ESTIMATED_FEE_PCT = float(os.getenv('ESTIMATED_FEE_PCT', '0.001'))      # 0.1% default
ESTIMATED_SLIPPAGE_PCT = float(os.getenv('ESTIMATED_SLIPPAGE_PCT', '0.001'))  # 0.1% default

# FIX 2026-03-19 (Helix): Execution Cost Filter — DO NOT TRADE if edge < costs
# Round-trip cost = 2 * (fee + slippage) = ~0.4%. Minimum edge must exceed this.
# On nano accounts ($15 positions), 0.4% cost = $0.06 — TP at 6% = $0.90.
# We require edge > 3x round-trip cost to ensure positive expectancy after friction.
# FIX 2026-03-23: Cost filter blocking all nano trades - relax for small accounts
EXECUTION_COST_FILTER_ENABLED = True
MIN_EDGE_MULTIPLE = 2.0   # Reduced from 3.0 to 2.0 (allow nano trades with smaller edge)
MIN_PROFIT_USD = 0.10     # Reduced from $0.20 to $0.10 (nano account viable profit)

MARKET_HOLON_TYPE = 'REAL'

SCAVENGER_THRESHOLD = 90.0
INITIAL_CAPITAL = 100.0
MISSION_TARGET = 3000.0
MISSION_NAME = "Operation Paper Centurion"
PRINCIPAL = 25.0
PAPER_TRADING = False  # LIVE MODE: System ready for real trading with patience mindset

TIMEFRAME = '15m'
DEFAULT_CYCLE_INTERVAL = 30
SIGNAL_SCAN_INTERVAL = 180

USE_ONNX = True

ENABLE_EVOLUTION = True  # FIX 2026-03-20: Reconnected evolution → live trading pipeline
# FIX 2026-03-18 (Path to Profitability): Optimized R/R for current market conditions
# OLD: 2.5% SL / 4.5% TP = 1.8:1 theoretical, but actual R/R = 0.53:1 (losses 1.87x larger)
# Issue: Low win rate (49.4%) + Poor R/R (0.53:1) = Negative expectancy (-0.36%)
# NEW: 2.0% SL / 4.0% TP = 2.0:1 (proven achievable R:R for micro-cap)
# Actual data: avg win $5.54 vs avg loss $4.69 = 1.18:1 realized R:R
# 6% TP rarely fills — lowered to 4% to capture more wins before reversal
DEFAULT_STOP_LOSS_PCT = 0.020
DEFAULT_TAKE_PROFIT_PCT = 0.040
STOP_LOSS_PCT = DEFAULT_STOP_LOSS_PCT
SOFTWARE_HARD_STOP_PCT = -0.05  # 2026-03-21: Belt & suspenders — force close at -5% if no exchange stop

# Global SL clamp bounds (used by ATR and evolved-genome sanity clamps)
MIN_STOP_LOSS_PCT = 0.015   # 1.5% floor
# FIX 2026-03-23: Increased from 3% to 4% to accommodate 2.5x ATR stops without clipping
MAX_STOP_LOSS_PCT = 0.040   # Increased from 0.030 to 0.040 (allow wider ATR-based stops)

# Take-profit floors for ATR-derived targets (prevents dust-level TP that fees erase)
MIN_TAKE_PROFIT_PCT = 0.01   # 1.0%
MAX_TAKE_PROFIT_PCT = 0.25   # 25% safety cap

# ATR-derived TP multiplier (controls how far TP is from entry in ATR terms)
# Lowering this reduces excessive TP distances when ATR is relatively large.
DEFAULT_TAKE_PROFIT_ATR_MULT = 2.0  # Was 4.0, lowered to improve R:R symmetry

# Structural Resonance: if conviction calibrates to ~0.60, only allow in ORDERED regime
STRUCTURAL_RESONANCE_LOW_CONV_THRESHOLD = 0.60

# Half-breach profit preservation tuning (ExitGuardian)
# FIX 2026-03-19 (Chronos/Helix): Old values (2%/35%/0.5%) strangled winners.
# Actual R:R collapsed from 1.8:1 → 0.53:1 because half-breach fired on normal noise.
# New values: let winners run to TP. Only protect truly extended moves.
HALF_BREACH_MIN_PEAK_PCT = 0.04          # Activate only after +4% peak (was 2%)
HALF_BREACH_RETAIN_FRACTION = 0.20       # Keep 20% of peak; allow 80% giveback (was 35%)
HALF_BREACH_MIN_LOCK_PNL_PCT = 0.015     # Don't trigger if < +1.5% current profit (was 0.5%)

EXPONENTIAL_GROWTH_MODE = False  # DISABLED: Conservative nano account mode
# GROWTH MODE PHILOSOPHY: Conservative compounding > aggressive gambling
# Risk per trade: 25% MAX (not 95%)
# Target: +5-10% monthly (not +100% weekly)
# Survival > Home runs

GROWTH_PHASE = 'CONSERVATIVE'  # Conservative: 25% risk (was AGGRESSIVE: 95%)
REINVESTMENT_RATE = 0.50  # 50% reinvest, 50% buffer (was 100%)
ARBITRAGE_MIN_APY = 150
ARBITRAGE_MAX_POSITIONS = 8

ARB_MIN_OPEN_INTEREST = 1000.0
ARB_MIN_OI_XSTOCKS = 500.0

ARB_MIN_FUNDING_VS_VOLATILITY = 2.0
ARB_ATR_PERIOD = 14

ARBITRAGE_RISK_PER_TRADE = 0.50
ARBITRAGE_MAX_LEVERAGE = 1.5
MAX_DAILY_LOSS_PCT = 0.03  # 3% max daily loss (was 10%)
LOSS_COOLDOWN_MINUTES = 15  # FIX 2026-03-14: 30→15min (reduce paralysis, allow re-entry)
MAX_DAILY_LOSSES = 5  # FIX 2026-03-14: 3→5 (allow more attempts, was too restrictive)
MONTE_CARLO_COOLDOWN_MINUTES = 30  # FIX 2026-03-14: 60→30min (reduce paralysis)
MAX_WEEKLY_LOSS_PCT = 0.20
MAX_DRAWDOWN_PCT = 0.25
FUNDING_CONVERGENCE_THRESHOLD = 0.10

MAX_POSITION_PER_ASSET_PCT = 0.15  # Max 15% per asset (was 25%)
MAX_TOTAL_EXPOSURE_PCT = 0.50  # Max 50% exposure (was 75%)
STOP_LOSS_ARB_PCT = 0.08

MIN_FUNDING_HISTORY_HOURS = 24
MIN_FUNDING_PER_8H = 0.10
MAX_FUNDING_RATE_CHANGE_PCT = 0.50
STOP_LOSS_ENABLED = True
TREND_ALIGNMENT_ENABLED = True

WHALE_THESIS_TTL = 4
STANDARD_THESIS_TTL = 2
THESIS_FAILURE_ACTION = 'FLAT'

WHALE_REQUIRES_STRUCTURE_SUPPORT = True
WHALE_ALLOW_NEUTRAL_WITH_BID_WALL = True
WHALE_STRUCTURE_GATE_ALLOW_NEUTRAL = True

PHASE1_CONCENTRATED_ASSETS = None
FORCE_EXIT_ASSETS = []

DYNAMIC_ASSET_SELECTION_ENABLED = True
MAX_CONCURRENT_ASSETS = 3
MIN_APY_THRESHOLD = 50.0
MIN_VOLUME_24H = 100000

BTC_HEDGE_ASSETS = ['ETH/USDT', 'XMR/USDT', 'XRP/USDT']

ARB_LAYER_ENABLED = True
ARB_IGNORE_MACRO_BIAS = True
ARB_EXEMPT_CORRELATION = True
ARB_EXEMPT_STACKING = True
ARB_STRUCTURE_BYPASS = True
ARB_POSITION_SIZE_PCT = 0.05  # Max 5% for arb (was 15%)
ARB_MAX_STACK_COUNT = 1
ARB_MIN_APY_THRESHOLD = 150.0

SMALL_REGIME_BEARISH_MODE = True
SIGNAL_IGNORE_VS_FAIL = True

FORCE_HARD_SYNC_ON_STARTUP = False

# FIX 2026-03-18: Added DOT, XTZ, ADA, XRP, LTC, IMX, LDO, AVAX, SHIB, STX, ARB, PYTH
# This prevents calculate_nano_position() from using 0.0 min_qty which causes tiny positions
MIN_TRADE_QTY = {
    'PAXG': 0.001, 
    'XAUT': 0.001, 
    'BTC': 0.0001, 
    'ETH': 0.001, 
    'SOL': 0.01, 
    'XMR': 0.01,
    'DOT': 0.5,      # ~$0.80 at $1.60 price
    'XTZ': 0.5,      # ~$0.50 at $1.00 price
    'ADA': 5.0,      # ~$1.25 at $0.25 price
    'XRP': 1.0,      # ~$2.00 at $2.00 price
    'LTC': 0.05,     # ~$4.00 at $80 price
    'IMX': 1.0,      # ~$1.50 at $1.50 price
    'LDO': 0.5,      # ~$1.00 at $2.00 price
    'AVAX': 0.1,     # ~$3.50 at $35 price
    'SHIB': 50.0,    # ~$0.50 at $0.00001 price
    'STX': 1.0,      # ~$2.00 at $2.00 price
    'ARB': 1.0,      # ~$2.00 at $2.00 price
    'PYTH': 1.0,     # ~$0.40 at $0.40 price
}

MAX_DAILY_FEES_PCT = 0.03
MAX_DAILY_FEES_HARD_LIMIT = 0.08
MAX_DAILY_FEES_ABSOLUTE = 5.0

MAX_TRADES_PER_HOUR = 10
MAX_TRADES_PER_DAY = 20  # Prevent overtrading (was 100)
MAX_TRADES_PER_SYMBOL_PER_DAY = 4  # 2026-03-21: DATA shows winners trade 2.5-4/day, PAXG was 14/day
MIN_TIME_BETWEEN_TRADES_SEC = 30

PIVOT_MIN_CONVICTION = 0.25
# DEFAULT_STOP_LOSS_PCT and DEFAULT_TAKE_PROFIT_PCT already defined above (line 42-43)
# REMOVED DUPLICATE: Line 118-119 was overriding 4.5% stops with 1.5% disaster

REGIME_SETTINGS = {
    'SMALL': {'allocation': 0.10, 'leverage': 2.0, 'max_pos': 3},
    'MEDIUM': {'allocation': 0.15, 'leverage': 3.0, 'max_pos': 4},
    'LARGE': {'allocation': 0.20, 'leverage': 5.0, 'max_pos': 6}
}

SMCE_TIER_SMALL = 500.0
SMCE_TIER_MEDIUM = 5000.0

SMCE_SMALL_MAX_EXPOSURE = 0.50
SMCE_SMALL_MAX_PER_ASSET = 0.20
SMCE_SMALL_MAX_CLUSTER = 0.25
SMCE_SMALL_MAX_LEVERAGE_NORMAL = 3.0
SMCE_SMALL_MAX_LEVERAGE_TRANSITION = 3.0
SMCE_SMALL_MAX_LEVERAGE_HIGH_ENTROPY = 2.0

SMCE_MICRO_THRESHOLD = 200.0
SMCE_MICRO_MAX_EXPOSURE = 0.40  # Max 40% exposure (was 65%)
SMCE_MICRO_MAX_PER_ASSET = 0.30

SMCE_DAILY_DRAWDOWN_LIMIT = 0.03
SMCE_WEEKLY_DRAWDOWN_LIMIT = 0.06
SMCE_STACKING_PRICE_BUFFER = 0.005
SMCE_DEFENSIVE_COOLDOWN_HOURS = 48

TRADING_MODE = 'FUTURES'

KRAKEN_SYMBOL_MAP = {
    'BTC/USDT': 'BTC/USD:USD',
    'ETH/USDT': 'ETH/USD:USD',
    'SOL/USDT': 'SOL/USD:USD',
    'XRP/USDT': 'XRP/USD:USD',
    'ADA/USDT': 'ADA/USD:USD',
    'DOGE/USDT': 'DOGE/USD:USD',
    'SUI/USDT': 'SUI/USD:USD',
    'UNI/USDT': 'UNI/USD:USD',
    'AAVE/USDT': 'AAVE/USD:USD',
    'SHIB/USDT': 'SHIB/USD:USD',
    'PAXG/USDT': 'PAXG/USD:USD',
    'LINK/USDT': 'LINK/USD:USD',
    'BNB/USDT': 'BNB/USD:USD',
    'LTC/USDT': 'LTC/USD:USD',
    'XMR/USDT': 'XMR/USD:USD',
    'XTZ/USDT': 'XTZ/USD:USD',
    'AVAX/USDT': 'AVAX/USD:USD',
    'DOT/USDT': 'DOT/USD:USD',
    'NEAR/USDT': 'NEAR/USD:USD',
    'PEPE/USDT': 'PEPE/USD:USD',
    'TAO/USDT': 'TAO/USD:USD',
    'XAUT/USDT': 'XAUT/USD:USD',
    'FET/USDT': 'FET/USD:USD',
    'WLD/USDT': 'WLD/USD:USD',
    'ARB/USDT': 'ARB/USD:USD',
    'OP/USDT': 'OP/USD:USD',
    'IMX/USDT': 'IMX/USD:USD',
    'STX/USDT': 'STX/USD:USD',
    'APT/USDT': 'APT/USD:USD',
    'TIA/USDT': 'TIA/USD:USD',
    'SEI/USDT': 'SEI/USD:USD',
    'INJ/USDT': 'INJ/USD:USD',
    'KAS/USDT': 'KAS/USD:USD',
    'LDO/USDT': 'LDO/USD:USD',
    'PYTH/USDT': 'PYTH/USD:USD',
    'JTO/USDT': 'JTO/USD:USD',
    'BONK/USDT': 'BONK/USD:USD',
    'WIF/USDT': 'WIF/USD:USD',
    'ORDI/USDT': 'ORDI/USD:USD',
    'SPYX/USDT': 'SPYX/USD:USD',
    'QQQX/USDT': 'QQQX/USD:USD',
    'NVDAX/USDT': 'NVDAX/USD:USD',
    'AAPLX/USDT': 'AAPLX/USD:USD',
    'GOOGLX/USDT': 'GOOGLX/USD:USD',
    'TSLAX/USDT': 'TSLAX/USD:USD',
    'MSTRX/USDT': 'MSTRX/USD:USD',
    'CRCLX/USDT': 'CRCLX/USD:USD',
    'HOODX/USDT': 'HOODX/USD:USD',
}

XSTOCKS_SYMBOLS = list(KRAKEN_SYMBOL_MAP.keys())

XSTOCKS_SYMBOLS = [s for s in XSTOCKS_SYMBOLS
                   if any(xs in s for xs in ['SPYX', 'QQQX', 'NVDAX', 'AAPLX',
                                              'GOOGLX', 'TSLAX', 'MSTRX', 'CRCLX', 'HOODX'])]

# === ASSET BLACKLIST (2026-03-20 Audit: consistent negative expectancy, large losses) ===
ASSET_BLACKLIST = {
    'PAXG/USDT',   # -$559, 18.5% WR, avg loss -$13.54
    'XMR/USDT',    # -$363, 31.6% WR, avg loss -$31.07
    'BTC/USDT',    # -$222, 38.7% WR, avg loss -$5.69
    'XAUT/USDT',   # -$210, 45.7% WR, avg loss -$6.67
    'SOL/USDT',    # -$0.95, 44.3% WR, persistent underperformer
}

ALLOWED_ASSETS = [
    # Tier 0: Capital Concentration — proven +EV over 50+ trades (2026-03-20 Audit)
    'TAO/USDT',    # +$974, 65% WR, 13:1 win/loss ratio
    'BNB/USDT',    # +$626, 71% WR, #2 performer
    'ETH/USDT',    # +$101, 64% WR, solid edge
    'AAVE/USDT',   # +$52, 59% WR, near-zero losses
    # Tier 1: Proven winners (positive PnL, high WR)
    'XTZ/USDT', 'LDO/USDT',
    # Tier 2: Solid performers (positive expectancy)
    'XRP/USDT', 'SHIB/USDT', 'PEPE/USDT',
    # Tier 3: Promising (traded profitably, smaller sample)
    'WIF/USDT', 'DOT/USDT',
    # REMOVED: LTC/USDT (3 trades, no edge), IMX/USDT (2 trades, no data)
]

# === CAPITAL CONCENTRATION WEIGHTS (2026-03-20 Audit) ===
# Tier 0 symbols get larger allocation; Tier 3 get smallest
# Governor multiplies base_notional by this weight during sizing
ASSET_ALLOCATION_WEIGHTS = {
    # Tier 0: Maximum concentration — proven +EV
    # DATA: TAO+BNB = 119/929 trades (13%) → +$1,600 (85% of all profit)
    # 2026-03-21: Moderate amplification — 2.0x → 3.0x for Tier 0
    'TAO/USDT': 3.0,   # +$974, 65% WR, avg win $12.03 — TOP PERFORMER
    'BNB/USDT': 3.0,   # +$626, 71% WR, highest win rate
    'ETH/USDT': 2.0,   # +$101, 64% WR, consistent edge
    'AAVE/USDT': 1.8,  # +$52, 59% WR, near-zero losses
    # Tier 1: Solid — moderate allocation
    'XTZ/USDT': 1.0,
    'LDO/USDT': 1.0,
    # Tier 2: Standard
    'XRP/USDT': 0.8,
    'SHIB/USDT': 0.8,
    'PEPE/USDT': 0.8,
    # Tier 3: Promising but structurally weak
    'WIF/USDT': 0.6,
    'DOT/USDT': 0.4,   # Was 0.6. 38.5% WR, loss avg 2x win avg, +$10 from 39 trades
}
ASSET_ALLOCATION_WEIGHT_DEFAULT = 0.4  # Unlisted symbols get cautious allocation (was 0.5)

ACTIVE_WATCHLIST = ALLOWED_ASSETS.copy()

ASSET_PREF_PREDATOR = [
    # 2026-03-20 Audit: Focus on proven +EV assets, blacklisted removed
    'ETH/USDT', 'TAO/USDT', 'AAVE/USDT',   # Tier 0 concentration
    'SUI/USDT', 'NEAR/USDT', 'FET/USDT', 'ARB/USDT', 'OP/USDT',
    'TIA/USDT', 'SEI/USDT', 'INJ/USDT', 'APT/USDT',
    'STX/USDT', 'IMX/USDT', 'KAS/USDT',
    'NVDAX/USDT', 'AAPLX/USDT', 'GOOGLX/USDT', 'TSLAX/USDT', 'MSTRX/USDT',
]

ASSET_PREF_SCAVENGER = [
    # 2026-03-20 Audit: Removed ADA (negative recent PnL), kept proven scavengers
    'XRP/USDT', 'DOGE/USDT', 'PEPE/USDT', 'SHIB/USDT', 'LINK/USDT', 'DOT/USDT', 'XTZ/USDT',
    'LTC/USDT', 'UNI/USDT', 'WIF/USDT', 'LDO/USDT',
    'SPYX/USDT', 'QQQX/USDT', 'CRCLX/USDT', 'HOODX/USDT',
]

MIN_TRADE_QTY = {
    'BTC': 0.0001,
    'ETH': 0.001,
    'SOL': 0.01,
    'XRP': 1.0,
    'ADA': 1.0,
    'DOGE': 10.0,
    'LTC': 0.1,
    'TAO': 0.01,
    'FET': 5.0, 'WLD': 5.0, 'ARB': 10.0, 'OP': 5.0, 'IMX': 5.0,
    'TIA': 2.0, 'SEI': 20.0, 'INJ': 0.5, 'APT': 1.0, 'SUI': 5.0,
    'LDO': 5.0, 'PYTH': 20.0, 'JTO': 5.0, 'BONK': 10000.0, 'WIF': 5.0,
    'ORDI': 0.2, 'KAS': 50.0, 'STX': 5.0,
    'SPYX': 0.1,
    'QQQX': 0.1,
    'NVDAX': 0.5,
    'AAPLX': 0.5,
    'GOOGLX': 0.5,
    'TSLAX': 0.5,
    'MSTRX': 0.5,
    'CRCLX': 1.0,
    'HOODX': 1.0,
    'BNB': 0.01,
    'SHIB': 10000.0,
    'AAVE': 0.1,
    'PAXG': 0.001,
    'NEAR': 1.0,
    'AVAX': 0.1,
    'PEPE': 10000.0,
    'XTZ': 1.0,
    'LINK': 0.5,
    'XAUT': 0.001,
    'XMR': 0.01,
    'UNI': 0.5,
    'DOT': 1.0,
}

TICK_SIZES = {
    'BTC': 0.50,
    'ETH': 0.10,
    'SOL': 0.01,
    'ADA': 0.00001,
    'XRP': 0.00001,
    'ADA': 0.00001,
    'DOGE': 0.00001,
    'SUI': 0.0001,
    'PEPE': 0.00000001,
    'SHIB': 0.00000001,
    'BONK': 0.00000001,
    'WIF': 0.0001,
    'FET': 0.0001,
    'WLD': 0.0001,
    'ARB': 0.0001,
    'OP': 0.0001,
    'TIA': 0.0001,
    'SEI': 0.0001,
    'INJ': 0.001,
    'BNB': 0.01,
    'AAVE': 0.01,
    'PAXG': 0.01,
    'NEAR': 0.0001,
    'AVAX': 0.001,
    'XTZ': 0.0001,
    'LINK': 0.001,
    'XAUT': 0.01,
    'XMR': 0.001,
    'UNI': 0.001,
    'DOT': 0.001,
}

# Execution quantity mismatch handling
# If True, downstream SL/TP logic will rely on the actual filled quantity
# returned by the exchange (safer for partial fills / rounding behavior).
ENFORCE_FILLED_QTY_FOR_SLTP = True
# Tolerance (absolute) for reporting a quantity mismatch between requested
# execution qty and the exchange-reported filled qty.
QTY_MISMATCH_TOLERANCE = 1e-6
# Enable verbose logging when execution qty != filled qty
LOG_QTY_MISMATCH = True

REGIME_SMALL_CEILING = 1000.0
REGIME_LARGE_CEILING = 5000000.0
REGIME_GRAVITY_CEILING = 50000000.0
REGIME_ATOM_FLOOR = 50000000.0

REGIME_PROMOTION_CYCLES = 3
REGIME_PROMOTION_HEALTH_THRESHOLD = 0.60
REGIME_DEMOTION_HEALTH_THRESHOLD = 0.30

# ========================================================================
# CENTRAL POSITION LIMITS (2026-03-18)
# Single source of truth for all position limits
# Used by: Atlas, Governor, Executor, Risk Management
# ========================================================================
POSITION_LIMITS_CENTRAL = {
    'NANO': {   # <$100 account
        'description': 'Nano Account: Conservative growth with leverage',
        'max_positions': 3,              # Allow diversification (was 1)
        'max_stacks': 2,                 # Max 2 stacks per position
        'max_allocation_pct': 0.30,      # 30% of margin per trade
        'max_position_size_pct': 0.30,   # Match Atlas
        'leverage_cap': 2.0,             # Max 2x leverage (reduced for safety)
        'min_trade_size_usd': 25.0,      # Atlas minimum
        'target_notional': 50.0,         # With leverage: $25 * 3x = $75
        'max_exposure_ratio': 5.0,
        'min_order_value': 15.0,
        'max_order_value_pct': 0.30,
        'cooldown_after_failure': 300,
        'allowed_pairs': [              # Added for Trinity compatibility
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT',
            'DOGE/USDT', 'AVAX/USDT', 'LINK/USDT',
            'LTC/USDT', 'SUI/USDT', 'NEAR/USDT', 'PEPE/USDT',
            'FET/USDT', 'ARB/USDT', 'OP/USDT', 'WIF/USDT', 'BONK/USDT',
            'DOT/USDT', 'XTZ/USDT', 'LDO/USDT', 'IMX/USDT', 'STX/USDT', 'SHIB/USDT',
            'TAO/USDT', 'SEI/USDT', 'APT/USDT', 'AAVE/USDT',  # 2026-03-19: Match expanded ALLOWED_ASSETS
        ],
    },
    'MICRO': {  # $100-500 account
        'description': 'Micro Account: Building consistency',
        'max_positions': 5,
        'max_stacks': 2,
        'max_allocation_pct': 0.25,      # 25% of margin
        'max_position_size_pct': 0.25,
        'leverage_cap': 3.0,
        'min_trade_size_usd': 50.0,
        'target_notional': 75.0,
        'max_exposure_ratio': 5.0,
        'min_order_value': 10.0,
        'max_order_value_pct': 0.25,
        'cooldown_after_failure': 300,
        'allowed_pairs': [
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT',
            'DOGE/USDT', 'AVAX/USDT', 'LINK/USDT',
            'LTC/USDT', 'SUI/USDT', 'NEAR/USDT', 'PEPE/USDT',
            'FET/USDT', 'ARB/USDT', 'OP/USDT', 'WIF/USDT', 'BONK/USDT',
            'DOT/USDT', 'XTZ/USDT', 'LDO/USDT', 'IMX/USDT', 'STX/USDT', 'SHIB/USDT',
            'TAO/USDT', 'SEI/USDT', 'APT/USDT', 'AAVE/USDT',  # 2026-03-19: Match expanded ALLOWED_ASSETS
        ],
    },
    'SMALL': {  # $500-5000 account
        'description': 'Small Account: Standard operations',
        'max_positions': 5,
        'max_stacks': 3,
        'max_allocation_pct': 0.20,      # 20% of margin
        'max_position_size_pct': 0.20,
        'leverage_cap': 5.0,
        'min_trade_size_usd': 100.0,
        'target_notional': 150.0,
        'max_exposure_ratio': 5.0,
        'min_order_value': 25.0,
        'max_order_value_pct': 0.20,
        'cooldown_after_failure': 300,
        'allowed_pairs': [
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT',
            'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT',
            'LTC/USDT', 'SUI/USDT', 'NEAR/USDT', 'PEPE/USDT',
            'FET/USDT', 'ARB/USDT', 'OP/USDT', 'WIF/USDT', 'BONK/USDT'
        ],
    },
    'MEDIUM': { # $5000-50000 account
        'description': 'Medium Account: Institutional scaling',
        'max_positions': 8,
        'max_stacks': 4,
        'max_allocation_pct': 0.15,
        'max_position_size_pct': 0.15,
        'leverage_cap': 4.0,
        'min_trade_size_usd': 250.0,
        'target_notional': 500.0,
        'max_exposure_ratio': 5.0,
        'min_order_value': 100.0,
        'max_order_value_pct': 0.15,
        'cooldown_after_failure': 300,
        'allowed_pairs': 'TOP_20',
    },
    'LARGE': {  # $50000+ account
        'description': 'Large Account: Full institutional',
        'max_positions': 10,
        'max_stacks': 5,
        'max_allocation_pct': 0.10,
        'max_position_size_pct': 0.10,
        'leverage_cap': 3.0,
        'min_trade_size_usd': 500.0,
        'target_notional': 1000.0,
        'max_exposure_ratio': 5.0,
        'min_order_value': 250.0,
        'max_order_value_pct': 0.10,
        'cooldown_after_failure': 300,
        'allowed_pairs': 'TOP_10',
    },
    'GRAVITY': {
        'description': 'Aggressive, Whale-Immune, Thesis-Blind',
        'max_positions': 10,
        'max_stacks': 5,
        'max_exposure_ratio': 5.0,
        'max_leverage': 5.0,
        'min_order_value': 100.0,
        'max_order_value_pct': 0.05,
        'cooldown_after_failure': 120,
        'allowed_pairs': 'TOP_15',
    },
    'ATOM': {
        'description': 'Institutional, Low-Leverage, Multi-Venue',
        'max_positions': 20,
        'max_stacks': 10,
        'max_exposure_ratio': 2.0,
        'max_leverage': 2.0,
        'min_order_value': 1000.0,
        'max_order_value_pct': 0.05,
        'cooldown_after_failure': 120,
        'allowed_pairs': 'TOP_25',
    }
}

# Backward compatibility: REGIME_PERMISSIONS now references CENTRAL limits
REGIME_PERMISSIONS = POSITION_LIMITS_CENTRAL.copy()

MEMECOIN_ASSETS = ['DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT', 'BONK/USDT', 'WIF/USDT']
MEMECOIN_PUMP_RVOL = 3.0
POLYMARKET_PATIENCE_MINUTES = 3

CRISIS_SAFE_HAVENS = ['BTC/USDT', 'PAXG/USDT', 'XAUT/USDT']
CRISIS_RISK_ASSETS = ['DOGE/USDT', 'ADA/USDT', 'SOL/USDT', 'PEPE/USDT', 'WIF/USDT']

MACRO_TICKERS = ['^GSPC', '^IXIC', '^RUT', 'USDT-USD', '^VIX']
MACRO_STACK_WEIGHT = 0.15
STRUCTURE_TARGET_MIN_RR = 1.5

DUMP_PUMP_ENABLED          = True
DUMP_GOLD_WINDOW_ET_HOUR   = 8
DUMP_BTC_WINDOW_ET_HOUR    = 10
DUMP_WINDOW_MINUTES_BEFORE = 10
DUMP_WINDOW_MINUTES_AFTER  = 25
DUMP_VELOCITY_PCT          = 0.0035
DUMP_RVOL_THRESHOLD        = 2.5
DUMP_ASK_WALL_DIST         = 0.005
DUMP_EXHAUSTION_CVD_RATIO  = 0.60

VIX_CALM_THRESHOLD         = 15.0
VIX_FEAR_THRESHOLD         = 20.0
VIX_PANIC_THRESHOLD        = 30.0

# ── ORION CTKS PROFIT NAVIGATOR (Market Path Alignment) ──
# Enhances CTKSStrategicHolon with market-path, momentum, and intermarket intelligence.
ORION_ENABLED              = True
# Path strictness: 1.0 = hard veto (no SELL in bullish path), 0.0 = disabled
# 0.5 = soft filter (conviction penalty). Recommended: 0.6-0.8
ORION_PATH_STRICTNESS      = 0.5
# Momentum must align with structure direction to enter
ORION_MOMENTUM_ALIGNMENT   = True
# Intermarket correlation: DXY rising strongly → avoid crypto BUY
ORION_INTERMARKET_FILTER   = True
# DXY daily change threshold to trigger risk-off signal (e.g., 0.5%)
ORION_DXY_RISK_OFF_PCT     = 0.005
# Yield curve (US10Y-US02Y) inversion threshold → bearish signal
ORION_YIELD_INVERSION_THRESHOLD = -0.10
# Minimum trade quality score (structure + momentum + macro aligned = 3, min 2)
ORION_MIN_ALIGNMENT_SCORE  = 2
# Additional intermarket tickers for MacroOracle
ORION_INTERMARKET_TICKERS  = ['DX-Y.NYB', '^TNX', '^IRX']  # DXY, US10Y, US2Y

MARKET_OPEN_FVG_ENABLED = True
MARKET_OPEN_FVG_WINDOWS = [(0, 0), (14, 30)]
MARKET_OPEN_FVG_DOMINANCE_THRESHOLD = 0.55

VOL_WINDOW_RISK_PCT = 0.01
VOL_WINDOW_LEVERAGE = 2
VOL_WINDOW_CYCLE_INTERVAL = 60
VOL_WINDOW_BTC_VOL_THRESHOLD = 0.50
VOL_WINDOW_FUNDING_THRESHOLD = 0.0005
VOL_WINDOW_SPREAD_THRESHOLD = 0.003
VOL_WINDOW_MIN_BALANCE_SHUTOFF = 10.0
VOL_WINDOW_TARGET_PROFIT = 0.05
VOL_WINDOW_MAX_POSITIONS = 2
VOL_WINDOW_GROSS_RISK_CAP = 0.05
VOL_WINDOW_MIN_VOLATILITY = 0.50

MAX_TOTAL_LEVERAGE = 10.0
GC_INTERVAL_CYCLES = 5
FAIR_WEATHER_MIN_BIAS = 0.35

SATELLITE_ASSETS = ['DOGE/USDT', 'ADA/USDT']  # FIX 2026-03-19: Removed SOL/USDT (15 trades, -$0.44, 33% WR — worst performer)
SATELLITE_MARGIN = 17.0
SATELLITE_LEVERAGE = 1.0
ARB_LEVERAGE = 3.0

MICRO_MAX_POSITIONS = 5

SATELLITE_RVOL_THRESHOLD = 6.72
SATELLITE_DOGE_RVOL_THRESHOLD = 6.72
SATELLITE_BBW_EXPANSION_THRESHOLD = 0.24
SATELLITE_ENTRY_RSI_CAP = 40.0

# RSI thresholds — canonical definition (updated at runtime by evolution engine)
# Removed duplicate definition that was at lines 788+ — single source of truth
STRATEGY_RSI_OVERSOLD = 35
STRATEGY_RSI_OVERBOUGHT = 65
STRATEGY_RSI_PANIC_BUY = 30.0
STRATEGY_RSI_ENTRY_MAX = 60.0

SATELLITE_BREAKEVEN_TRIGGER = 0.02
SATELLITE_TAKE_PROFIT_1 = 0.10
SATELLITE_STOP_LOSS = 0.020  # FIX 2026-03-19: 4.5%→2.0% (align with DEFAULT_STOP_LOSS_PCT; was causing avg loss 1.87x avg win)

SCAVENGER_LEVERAGE = 3.0      # Reduced from 5.0 to 3.0 (FIX 2026-03-16)
PREDATOR_LEVERAGE = 3.0       # Reduced from 5.0 to 3.0 (FIX 2026-03-16)
MICRO_HARD_LEVERAGE_LIMIT = 3.0  # Reduced from 5.0 to 3.0
PREDATOR_TRAILING_STOP_ATR_MULT = 4.5  # FIX 2026-03-14: 2.5x→4.5x (was stopping out on noise)

MIN_ORDER_VALUE = 5.0
# FIX 2026-03-16 (Chronos): Reduced position size to control risk during negative expectancy
KELLY_HARD_CAP_MARGIN = 20.0  # Reduced from $35 to $20 (was increased 2026-03-14)
SIZE_MAX_ALLOCATION = 0.15    # Reduced from 0.20 to 15% per position
MAX_RISK_PCT = 0.01           # Reduced from 0.02 to 1% risk per trade

GOVERNOR_PER_TRADE_ALLOC_REDUCTION = 0.8
GOVERNOR_CAPITAL_BUFFER_PCT = 0.10
GOVERNOR_EXPOSURE_SOFT_LIMIT = 0.25
GOVERNOR_EXPOSURE_HARD_LIMIT = 0.30
GOVERNOR_BUFFER_RELEASE_THRESHOLD = 0.50

SCAVENGER_MAX_MARGIN = 0.50
SCAVENGER_STOP_LOSS = 0.020  # FIX 2026-03-19: 4.5%→2.0% (align with DEFAULT_STOP_LOSS_PCT)
SCAVENGER_SCALP_TP = 0.04   # FIX 2026-03-19 (Helix): 2.5%→4% (old value fired in noise, strangled R:R)
PREDATOR_STOP_LOSS = 0.020  # FIX 2026-03-19: 4.5%→2.0% (align with DEFAULT_STOP_LOSS_PCT)
PREDATOR_TAKE_PROFIT = 0.06  # FIX 2026-03-14: 3%→6% (capture real moves)

PROFIT_TARGETS = {
    'WHALE_BID_WALL': {
        'rapid': 0.03,    # FIX 2026-03-18: 2%→3% (improved 1.5x)
        'normal': 0.06,   # FIX 2026-03-18: 5%→6% (let winners run longer)
        'runner': 0.12    # FIX 2026-03-18: 10%→12% (home runs)
    },
    'WHALE_ACCUMULATION': {
        'rapid': 0.03,    # FIX 2026-03-18: 2%→3%
        'normal': 0.06,   # FIX 2026-03-18: 5%→6%
        'runner': 0.12    # FIX 2026-03-18: 10%→12%
    },
    'PACK_HUNT': {
        'rapid': 0.03,    # FIX 2026-03-18: 2%→3%
        'normal': 0.05,   # FIX 2026-03-18: 4%→5%
        'runner': 0.10    # FIX 2026-03-18: 8%→10%
    },
    'DIP': {
        'rapid': 0.03,    # FIX 2026-03-18: 2%→3%
        'normal': 0.06,   # FIX 2026-03-18: 5%→6%
        'runner': 0.12    # FIX 2026-03-18: 10%→12%
    },
    'DEFAULT': {
        'rapid': 0.02,    # FIX 2026-03-18: 1%→2%
        'normal': 0.060,  # FIX 2026-03-18: 3%→6% — matches new main targets
        'runner': 0.100   # FIX 2026-03-18: 6%→10% — home runs
    }
}

EXIT_PYRAMID = {
    'aggressive': [0.3, 0.4, 0.3],
    'balanced':   [0.2, 0.5, 0.3],
    'conservative': [0.1, 0.6, 0.3]
}

IMMEDIATE_SCALP_CONFIG = {
    'rsi_overbought': 95.0,   # FIX 2026-03-19 (Helix): 90→95 (RSI 90 is normal in crypto momentum)
    'profit_target': 0.035,   # FIX 2026-03-19 (Helix): 2%→3.5% (stop scalping inside noise range)
    'position_size_ratio': 0.5
}

# FIX 2026-03-15 (Chronos v2): Raised to match main PROFIT_TARGETS scale.
# Old ny_session.normal=1.0% averaged with DEFAULT 2.5% → 1.75% effective TP.
# Against 4.5% SL this was R/R=0.39:1 (structurally negative expectancy at any win rate).
# New: ny_session.normal=3.0% → avg(3.0%+3.0%)=3.0% effective TP → R/R≈0.67:1
TIME_BASED_PROFIT_TARGETS = {
    'asian_session':  {'rapid': 0.008, 'normal': 0.020, 'runner': 0.040},
    'london_session': {'rapid': 0.012, 'normal': 0.030, 'runner': 0.060},
    'ny_session':     {'rapid': 0.012, 'normal': 0.030, 'runner': 0.060},
    'weekend':        {'rapid': 0.005, 'normal': 0.015, 'runner': 0.030}
}

ICT_ASIAN_SESSION = "00:00-08:00"
ICT_LONDON_OPEN = "08:00-10:00"
ICT_NY_OPEN = "13:30-15:30"
ICT_SWEEP_TOLERANCE_PCT = 0.005
STACK_PROFIT_TARGETS = {
    1: {'target': 0.005, 'stop': 0.002},
    2: {'target': 0.010, 'stop': 0.004},
    3: {'target': 0.020, 'stop': 0.010},
    4: {'target': 0.035, 'stop': 0.018}
}

STACK_TRAILING_ENABLED = True
STACK_TRAILING_PNL_THRESHOLD = 0.015
STACK_TRAILING_ATR_MULT = 2.0
STACK_TRAILING_STEP = 0.005

COMPOUNDING_ENABLED = True
COMPOUNDING_MIN_PROFIT_USD = 2.0
COMPOUNDING_REINVEST_PCT = 0.50
COMPOUNDING_MAX_POSITION_ADD = 0.10

MARKET_ADJUSTMENTS = {
    'crisis_score': {
        'critical': 0.8,
        'reduce_targets': 0.7
    },
    'sentiment': {
        'bearish': 0.0,
        'bullish': 0.2
    }
}

POSITION_LIMITS = {
    'max_positions': 8,
    'max_correlation': 0.1,
    'size_limits': {
        'DEFAULT': 0.05,
        'LARGE_CAP': 0.10,
        'MEME': 0.03
    },
    'daily_turnover_limit': 1.0
}

MICRO_GUARD_PORTFOLIO_NOTIONAL_MULT = 2.0
MICRO_GUARD_SINGLE_NOTIONAL_MULT = 1.0
MICRO_GUARD_GROSS_LEVERAGE_MULT = 2.0
MICRO_GUARD_GROSS_LEVERAGE = 2.0

MICRO_GUARD_CASH_PRESERVATION_THRESHOLD = 10.0
MICRO_GUARD_CASH_PRESERVATION_LEVERAGE = 2.0

ORDER_FAILURE_COOLDOWN = {
    'insufficientAvailableFunds': 300,
    'wouldNotReducePosition': 60,
    'network': 10,
    'default': 120,
}

MAX_CONSECUTIVE_FAILURES = 3

DEFAULT_ORDER_TYPE = 'limit'
MARKET_ORDER_ONLY_URGENT = True
PRICE_COLLAR_BPS = 50
TIGHT_COLLAR_ENABLED = True
THIN_BOOK_COLLAR_BPS = 25
SLIPPAGE_CHECK_ENABLED = True
MAX_ALLOWED_SLIPPAGE_BPS = 100
LIQUIDITY_DEPTH_MULT = 3.0

MAX_DAILY_FEES_PCT = 0.05
MAX_DAILY_FEES_HARD_LIMIT = 0.10
FEE_TRACKING_ENABLED = True
ESTIMATED_MAKER_FEE = 0.0002
ESTIMATED_TAKER_FEE = 0.0005
ORDER_CANCELLATION_LIMIT = 10

REGIME_CHANGE_RETRY_DELAY = 30
VETO_RELAX_RETRY_DELAY = 45
ETH_USDT_PRIORITY_RETRY = True
REGIME_RETRY_MAX_ATTEMPTS = 3
REGIME_RETRY_BACKOFF = 1.5

SCOUT_CANDIDATES = ALLOWED_ASSETS
SCOUT_RVOL_THRESHOLD = 2.5
SCOUT_HYPE_THRESHOLD = 0.5
SCOUT_CACHE_TTL = 60
SCOUT_CYCLE_INTERVAL = 60          # Faster rotation for better asset turnover (was 120s)
SCOUT_BATCH_SIZE = 16               # Assets scanned per cycle (3 batches cover 48 assets)
SCOUT_BATCH_ROTATION = True         # Enable batch rotation for full universe coverage
SCOUT_ALLOW_EXPANSION = True        # Allow scout to promote from expanded ALLOWED_ASSETS universe
STRICT_ASSET_UNIVERSE = True        # Enforce ALLOWED_ASSETS across scout/trader pipelines

# (RSI thresholds defined above — single source of truth, updated by evolution at runtime)
# FIX 2026-03-18: Raised model thresholds to achieve positive expectancy
# OLD: LSTM=0.52, XGB=0.50 → accepting near-random predictions → negative expectancy (-0.36%)
# NEW: LSTM=0.65, XGB=0.68 → stricter filtering for isolated BUY edge
# ATLAS: With SELL disabled, need higher confidence to maintain win rate
STRATEGY_LSTM_THRESHOLD = 0.65  # Was 0.52 (accepting random predictions)
STRATEGY_XGB_THRESHOLD = 0.68   # Was 0.50 (purely random)
STRATEGY_POST_EXIT_COOLDOWN_CANDLES = 2

# 2026-03-20 AUDIT: Multi-signal confirmation — minimum independent signals required to enter
# Prevents single-signal dependency (e.g. whale-only entries in CHAOTIC regime)
# Signals counted: TREND, DIP, PANIC, QUANTUM, WHALE, KALMAN_VALUE, LSTM_AGREE
# FIX 2026-03-23: Confluence requirement of 2 signals blocking ALL trades
# Analysis: Logs show most assets only trigger 1 signal (DIP), getting vetoed
MIN_BUY_CONFIRMATIONS = 1  # Reduced from 2 to 1 (allow single high-quality signal)

# ATLAS PROFIT ARCHITECT: Disable SELL strategy to isolate BUY edge
# BUY strategy: +$0.0005/trade expectancy (positive but masked)
# SELL strategy: -$0.008/trade expectancy (strongly negative)
# Combined: -$0.0036/trade (losses bury wins)
BUY_ONLY_MODE = True  # Disable all SELL signals

# ATLAS: Additional filtering for isolated BUY edge
# 2026-03-20 FIX: Audit thresholds were calibrated for SampleEntropy (0.4-1.2 scale)
# but applied to Shannon entropy (1.5-2.1 scale on 10-bin ln histogram). 0.80 blocked 100%.
# Now: allow ORDERED + TRANSITION. CHAOTIC (Shannon > 1.85) remains blocked.
ENTRY_REGIME_FILTER = ['ORDERED', 'TRANSITION']  # Trade in structured + transitional markets

# Hard entropy ceiling: Block entries when Shannon entropy > threshold (last-resort safety)
# Shannon 10-bin ln scale: max=2.303, normal markets=1.5-2.1, CHAOTIC boundary=1.85
# Set to 2.1 (matching Rust guard) — blocks >91% of max entropy (truly uniform/random)
# At 2.0 half the watchlist is blocked including #1/#2 performers (TAO=2.016, BNB=2.049)
ENTRY_MAX_ENTROPY = 2.1  # Shannon entropy scale (was 0.80 — wrong scale!)
ENTRY_REGIME_ALIASES = {
    'BULLISH': 'ORDERED',
    'NEUTRAL': 'TRANSITION',
    'BEARISH': 'CHAOTIC'
}

# Symbol-specific conviction floors
# 2026-03-20 FIX: PPO brain outputs max ~0.60 conviction (sigmoid learned behavior).
# Previous floors (0.68-0.83) created permanent entry veto — zero trades possible.
# Lowered to sit below PPO output range while preserving tiered selectivity.
SYMBOL_CONVICTION_FLOORS = {
    # Tier 0: Capital Concentration — lowest bar for top earners
    # TAO (+$974, 65% WR), ETH (+$101, 64% WR), AAVE (+$52, 59% WR), BNB (+$626, 71% WR)
    'TAO/USDT': 0.45,
    'ETH/USDT': 0.45,
    'AAVE/USDT': 0.45,
    'BNB/USDT': 0.45,

    # Tier 1: Proven winners — moderate barrier
    'XTZ/USDT': 0.50,
    'LDO/USDT': 0.50,

    # Tier 2: Solid performers — standard barrier
    'XRP/USDT': 0.53,
    'SHIB/USDT': 0.53,
    'PEPE/USDT': 0.53,

    # Tier 3: Promising — highest bar within PPO range
    'WIF/USDT': 0.55,
    'DOT/USDT': 0.58,  # Was 0.55. 38.5% WR, loss avg 2x win avg — demand stronger signal
}
CONVICTION_FLOOR_DEFAULT = 0.50  # 2026-03-20: Lowered from 0.70 — aligned with PPO output range (~0.60 max)

# ========================================================================
# DYNAMIC CONVICTION ENGINE (2026-03-24)
# Adaptive conviction floors based on confluence, performance, and regime
# Instead of static floors, conviction requirements adapt to market conditions
# ========================================================================

# Base conviction floor (starting point before adjustments)
CONVICTION_FLOOR_BASE = 0.50

# Confluence bonus: Each additional signal beyond 1 reduces floor
# Example: 3 signals = -0.10 floor reduction (0.50 → 0.40)
CONFLUENCE_BONUS_PER_SIGNAL = 0.05
CONFLUENCE_MAX_BONUS = 0.15  # Cap at 3 extra signals

# Performance adjustment: Recent win rate affects floor
# Hot streak (>60% WR) = lower floor, Cold streak (<40% WR) = higher floor
PERFORMANCE_BONUS_THRESHOLD = 0.60  # Win rate for floor discount
PERFORMANCE_PENALTY_THRESHOLD = 0.40  # Win rate for floor premium
PERFORMANCE_ADJUSTMENT_PCT = 0.10  # 10% discount/premium

# Regime alignment bonus: Strategy matching regime gets lower floor
REGIME_ALIGNMENT_BONUS = 0.05  # 5% reduction for aligned strategy
REGIME_MISALIGNMENT_PENALTY = 0.10  # 10% premium for misaligned strategy

# Strategy-Regime alignment matrix (which strategies work in which regimes)
STRATEGY_REGIME_ALIGNMENT = {
    'DIP': ['LOW_VOL_MEAN_REVERT', 'EXPANSION'],  # DIP works in ranging/growth
    'TREND': ['EXPANSION', 'ORDERED'],  # Trend works in trending markets
    'PANIC': ['CHAOTIC', 'DEFENSIVE'],  # Panic buys in crashes
    'WHALE': ['LOW_VOL_MEAN_REVERT', 'EXPANSION', 'ORDERED'],  # Whales work in stable markets
    'STRUCTURAL_RESONANCE': ['ORDERED'],  # Resonance needs ordered markets
    'VOLATILITY_SQUEEZE': ['EXPANSION', 'TRANSITION'],  # Squeeze works in volatile markets
}

# Hard floor: Absolute minimum regardless of all bonuses (safety net)
CONVICTION_FLOOR_HARD_MINIMUM = 0.35  # Never go below 0.35

# Soft ceiling: Maximum floor after penalties (prevents over-blocking)
CONVICTION_FLOOR_SOFT_MAXIMUM = 0.70  # Rarely exceed 0.70

# Lookback period for recent win rate calculation
CONVICTION_WINRATE_LOOKBACK = 10  # Last 10 trades for performance check

GOVERNOR_COOLDOWN_SECONDS = 0
# FIX 2026-03-16 (Chronos): SOL/USDT stacking issue - prevents too-close entries
# Old: 0.15% caused constant "Stack Too Close" blocks and overlapping positions
# New: 2% minimum distance between stacked positions
GOVERNOR_MIN_STACK_DIST = 0.02  # Increased from 0.0015 (0.15%) to 2%
# FIX 2026-03-23: Stack timeout was 5min (300s) - too short for swing trades
# Analysis: Positions need 30-60min to develop thesis, 5min timeout forced premature exits
STACK_TIMEOUT_SECONDS = 3600  # Increased from 300 (5min) to 3600 (60min) for swing trades
GOVERNOR_MAX_MARGIN_PCT = 0.25  # Match winning strategy (was 95%!)
GOVERNOR_STACK_DECAY = 0.8
GOVERNOR_MAX_TREND_AGE_HOURS = 24.0
GOVERNOR_TREND_DECAY_START = 12.0

MAX_POSITIONS_PER_ASSET = 5

MAX_SIMULTANEOUS_POSITIONS = 8  # Increased from 5 for better slot utilization (8 slots available)

POOL_A_SLOTS = 4  # Increased from 3 to match new position limit
POOL_A_ALLOCATION_PCT = 0.50
POOL_A_COOLDOWN_SEC = 60
POOL_A_ENTRIES_PER_CYCLE = 1

POOL_B_SLOTS = 12  # Increased from 10 for more arb opportunities
POOL_B_ALLOCATION_PCT = 0.60

MANAGEMENT_MODE_MAX_DURATION_SEC = 3600  # Extended from 1800 (60min vs 30min) - Give positions more time to work
MANAGEMENT_MODE_COOLDOWN_SEC = 120
MANAGEMENT_MODE_AUTO_EXIT_THRESHOLD = 0.80
POOL_B_COOLDOWN_SEC = 90
POOL_B_ENTRIES_PER_CYCLE = 1

POOL_RESERVE_PCT = 0.10

DYNAMIC_SLOT_SCALING = {
    100: {'pool_a': 4, 'pool_b': 8},
    500: {'pool_a': 6, 'pool_b': 10},
    1000: {'pool_a': 8, 'pool_b': 12},
    5000: {'pool_a': 12, 'pool_b': 18},
}

SLOT_POOL_DIRECTIONAL = POOL_A_SLOTS
SLOT_POOL_ARB = POOL_B_SLOTS
SIZE_MAX_ALLOCATION = 0.35
MAX_POSITION_SIZE_PCT = 0.30  # Aligned with Atlas (was 0.35)
MAX_ENTRIES_PER_CYCLE = 1  # Reduced from 2 - Prevent rapid accumulation, reduce management mode triggers
MIN_SECONDS_BETWEEN_STACKS = 60
STACK_SNOOZE_DURATION = 120

# === REGIME-AWARE STACK DISTANCE (FIX 2026-03-14) ===
# Distance buffer varies by regime to optimize stacking behavior
STACK_DISTANCE_BUFFERS = {
    'HARVEST':    0.003,  # 0.3% - Tight (ordered market, allow closer stacks)
    'EXPANSION':  0.005,  # 0.5% - Moderate (trending, balanced approach)
    'TRANSITION': 0.010,  # 1.0% - Wide (uncertain, require more separation)
    'DEFENSIVE':  0.020,  # 2.0% - Very wide (high risk, stacks rarely allowed)
}
SMCE_STACKING_PRICE_BUFFER = 0.005  # Default fallback (5min)

HYGIENE_ENABLED = True
HYGIENE_TOXIC_FUNDING_APY = -50.0
HYGIENE_CONVICTION_DECAY_THRESHOLD = 0.60
HYGIENE_CONVICTION_DECAY_CYCLES = 3
HYGIENE_OPPORTUNITY_COST_PCT = 25.0
HYGIENE_MIN_AGE_MINUTES = 5.0

GRADUATION_SIPHON_PCT = 0.10
GRADUATION_MAX_SLOT_BONUS = 4
REGIME_PROMOTION_CYCLES = 5

ATR_PERIOD = 14
ATR_STORM_MULTIPLIER = 3.0
# FIX 2026-03-23: Stops too tight (1.8x ATR) causing frequent stop-outs on noise wicks
# Analysis: Crypto wicks are 1-3 ATR normal, need 2.5x minimum to avoid premature stop-outs
ATR_STOP_LOSS_MULTIPLIER = 2.5  # Increased from 1.8 to 2.5 (give trades room to breathe)
ATR_STOP_LOSS_MIN = 2.0         # Minimum ATR multiplier (never go below 2x)
ATR_STOP_LOSS_MAX = 4.0         # Maximum ATR multiplier (cap risk on extreme volatility)
KELLY_LOOKBACK = 20
VOL_SCALAR_MIN = 0.5
VOL_SCALAR_MAX = 2.0

BB_PERIOD = 20
BB_STD = 2

SENTIMENT_SOURCES = [
    'https://cointelegraph.com/rss',
    'https://www.coindesk.com/arc/outboundfeeds/rss/',
    'https://cryptopanic.com/news/rss/',
    'https://www.reddit.com/r/CryptoCurrency/top/.rss?t=hour',
    'https://www.reddit.com/r/Bitcoin/hot/.rss',
    'https://www.reddit.com/r/solana/hot/.rss',
    'https://www.cnbc.com/id/100727362/device/rss/rss.html',
    'https://www.investing.com/rss/commodities.rss'
]

PHYSICS_CORRELATION_THRESHOLD = 0.75
PHYSICS_MIN_RVOL = 1.0
PHYSICS_MAX_ENTROPY = 1.35
PHYSICS_OU_STRETCH_THRESHOLD = 2.0
PHYSICS_OU_LAMBDA_THRESHOLD = 10.0
PHYSICS_MAX_RUIN_PROBABILITY = 0.60  # BLOCK toxic trades (>60% ruin prob) - relaxed from 0.35 to allow valid setups

CORRELATION_CHECK = True

WHALE_RVOL_THRESHOLD = 3.0
WHALE_SENTIMENT_WEIGHT = 0.2
WHALE_ACCUMULATION_RVOL = 2.0
WHALE_ACCUMULATION_ATR_FACTOR = 0.8
WHALE_DEFENSE_RVOL = 2.5
WHALE_ORDER_IMBALANCE_RATIO = 8.0    # 2026-03-20 Audit: Raised from 5.0 — thin books fire BID_WALL too easily
WHALE_FUNDING_SQUEEZE_THRESHOLD = -0.00005
# 2026-03-20 Audit: Require multiple whale factors to confirm (not just book skew alone)
# BID_WALL only fired 5,334 times as sole signal — fragile single-signal dependency
WHALE_MIN_FACTORS = 2  # Require at least 2 of: ROCKET, ACCUMULATION, DEFENSE, BID_WALL, SQUEEZE
SENTIMENT_WEIGHT = 0.3
SENTIMENT_THRESHOLD_BULL = 0.2
SENTIMENT_THRESHOLD_BEAR = -0.2
ENABLE_SHORT_SELLING = True

GMB_THRESHOLD = 0.40
GMB_EXIT_HYSTERESIS = 0.10

SOLVENCY_PANIC_THRESHOLD = 0.80
SOLVENCY_PANIC_REDUCTION = 0.10

CONSOLIDATION_WEIGHT_PNL = 0.30
CONSOLIDATION_WEIGHT_CONVICTION = 0.25
CONSOLIDATION_WEIGHT_LIQUIDITY = 0.15
CONSOLIDATION_WEIGHT_AGE = 0.10
CONSOLIDATION_WEIGHT_CORRELATION = 0.20

CONSOLIDATION_MIN_AGE_MINUTES = 5
CONSOLIDATION_DUST_THRESHOLD = 1.0
CONSOLIDATION_STALE_HOURS = 24.0
CONSOLIDATION_MARGIN_BUFFER = 1.5
CONSOLIDATION_HARD_BUFFER = 1.2

ACC_RISK_FLOOR = 0.5
ACC_RISK_CEILING = 1.0
ACC_DRAWDOWN_LIMIT = 0.05
ACC_HARD_STOP_LIMIT = 0.10
ACC_SANITY_THRESHOLD = 0.05

ENABLE_ARBITRAGE = True
ARB_SPATIAL_THRESHOLD = 0.005
ARB_FUNDING_THRESHOLD = 0.0005
ARB_MAX_EXPOSURE = 0.10
ARB_COOLDOWN_SECONDS = 120
ARB_MAX_STALENESS = 10.0
ARB_MIN_PROFIT_SPREAD = 0.002

EXEC_MAX_POV = 0.08
EXEC_VWAP_WINDOW = 25
EXEC_TWAP_MAX_DURATION = 7200
EXEC_IMPACT_THRESHOLD = 0.15

PPO_LEARNING_RATE = 0.0005
PPO_CLIP_RATIO = 0.25
PPO_REWARD_DRAWDOWN_PENALTY = 1.5

CCXT_RATE_LIMIT = True
CCXT_POOL_SIZE = 3
TRADER_MAX_WORKERS = 2
TRADER_MAX_CYCLE_ENTRIES = 1

# =============================================================================
# AEGIS QUANTSEC: WebSocket Health Monitoring Configuration
# =============================================================================
# Enhanced keepalive settings to prevent timeout issues (H-01 fix)
# Reference: WebSocket timeout errors in live_trading_session_*.log

# Enable WebSocket health monitoring
AEGIS_WS_HEALTH_ENABLED = True

# Ping interval - AEGIS recommends 15s for HFT systems (vs default 30s)
# Shorter interval detects connection issues faster
AEGIS_WS_PING_INTERVAL = 15.0  # seconds

# Pong timeout - how long to wait for response before marking unhealthy
AEGIS_WS_PONG_TIMEOUT = 10.0  # seconds

# Consecutive timeouts before marking connection as CRITICAL
AEGIS_WS_MAX_TIMEOUTS = 3

# Health check interval - how often to evaluate connection health
AEGIS_WS_HEALTH_CHECK_INTERVAL = 5.0  # seconds

# Minimum messages per minute before marking as DEGRADED
AEGIS_WS_MIN_MESSAGES_PER_MINUTE = 10.0

# Enable automatic REST fallback when WebSocket is unhealthy
AEGIS_WS_REST_FALLBACK_ENABLED = True

# Cooldown between REST fallback fetches (prevent rate limit)
AEGIS_WS_REST_FALLBACK_COOLDOWN = 5.0  # seconds

USE_INTEL_GPU = True
USE_OPENVINO = True

FAMILY_L1 = ['SOL/USDT', 'ADA/USDT', 'AVAX/USDT']
FAMILY_PAYMENT = ['XRP/USDT', 'LTC/USDT']
FAMILY_MEME = ['DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT', 'BONK/USDT', 'WIF/USDT']

IMMUNE_MAX_DAILY_DRAWDOWN = 0.08  # FIX 2026-03-19 (Helix): 5%→8%. SMCE handles 3% daily via doctrine.
                                   # Monitor fever at 5% competed with SMCE, causing double-lockdown.
                                   # Raise to 8% so only Monitor catches REAL catastrophic drawdown.
                                   # Normal daily DD (3-5%) is managed by SMCE alone (no paralysis).
IMMUNE_MAX_LEVERAGE_RATIO = 5.0

MAINTENANCE_MARGIN_RATE = 0.50

USE_WEBSOCKETS = True
WS_BUFFER_SIZE = 1000

MICRO_CAPITAL_MODE = False
MICRO_MAX_LEVERAGE = 5.0
MICRO_MAX_EXPOSURE_RATIO = 5.0

STACKING_MIN_EQUITY = 100.0
STACKING_BUFFER_MULTIPLIER = 5.0

NANO_MAX_POSITIONS = 2
NANO_COOLDOWN_AFTER_FAILURE = 86400

# OPTIMIZATION 2026-03-09: Increase minimum value to 15.0 to overcome Kraken fee drag
MIN_ORDER_VALUE = 15.0  # Nano account friendly (was $15)
MAX_ORDER_VALUE = 15.0  # 2026-03-21 FIX: DATA shows trades >$15 notional = -$982 PnL / 31.6% WR. (was $50)

NANO_CAPITAL_THRESHOLD = 50.0
KELLY_HARD_CAP_MARGIN = 35.0  # INCREASED 2026-03-14: $25→$35 for "perfect signal" concentration (was conservative)
NANO_ALLOCATION_PCT = 0.30    # FIX 2026-03-18: Match Atlas config (was 0.15) - 30% of available margin
NANO_MAX_LEVERAGE = 2.0       # 2026-03-20 FIX: Match POSITION_LIMITS_CENTRAL NANO tier (was 5.0)
NANO_MAX_POSITIONS = 2
NANO_COOLDOWN_AFTER_FAILURE = 86400

PERSONALITY_BTC_ATR_FILTER = 0.5
PERSONALITY_SOL_RSI_LONG = 50.0
PERSONALITY_SOL_RSI_SHORT = 45.0
PERSONALITY_DOGE_RVOL = 1.2

ASSET_PROFILES = {
    'BTC/USDT': {
        'description': 'Profiled (Fit 25.5)',
        'rsi_buy': 24.86,
        'rsi_sell': 88.57,
        'stop_loss': 0.042,
        'take_profit': 0.088,
        'satellite_rvol': 4.95,
        'satellite_stop': 0.042,
        'rvol_threshold': 1.0,
        'leverage_cap': 4.33
    },
    'ETH/USDT': {
        'description': 'Profiled (Fit 32.3)',
        'rsi_buy': 45.35,
        'rsi_sell': 95.00,
        'stop_loss': 0.021,
        'take_profit': 0.119,
        'satellite_rvol': 2.38,
        'satellite_stop': 0.021,
        'rvol_threshold': 1.0,
        'leverage_cap': 1.06
    },
    'SOL/USDT': {
        'description': 'Trend King (Fit 57.9) — RESTRICTED 2026-03-19 (15 trades, -$0.44, 33% WR)',
        'rsi_buy': 50.64,
        'rsi_sell': 93.77,
        'stop_loss': 0.020,   # FIX 2026-03-19: 5%→2% (was 1.2:1 R:R catastrophe)
        'take_profit': 0.090,  # FIX 2026-03-19: 6%→9% (minimum 4.5:1 R:R required)
        'satellite_rvol': 3.21,
        'satellite_stop': 0.020,  # FIX 2026-03-19: 5%→2%
        'rvol_threshold': 1.0,
        'leverage_cap': 1.0    # FIX 2026-03-19: 4.15x→1x (no leverage while restricted)
    },
    'XRP/USDT': {
        'description': 'Profiled (Fit 0.0 - Fallback)',
        'rsi_buy': 25.47,
        'rsi_sell': 79.57,
        'stop_loss': 0.019,
        'take_profit': 0.056,
        'satellite_rvol': 2.51,
        'satellite_stop': 0.032,
        'rvol_threshold': 1.0,
        'leverage_cap': 3.58
    },
    'ADA/USDT': {
        'description': 'Deep Diver (Fit 33.8)',
        'rsi_buy': 20.02,
        'rsi_sell': 95.00,
        'stop_loss': 0.043,
        'take_profit': 0.128,
        'satellite_rvol': 2.15,
        'satellite_stop': 0.043,
        'rvol_threshold': 1.0,
        'leverage_cap': 5.0
    },
    'DOGE/USDT': {
        'description': 'Dual Mode (Fit 11.7)',
        'rsi_buy': 47.24,
        'rsi_sell': 85.98,
        'stop_loss': 0.028,
        'take_profit': 0.089,
        'satellite_rvol': 3.68,
        'satellite_stop': 0.028,
        'rvol_threshold': 1.2,
        'leverage_cap': 2.92
    },
    'SUI/USDT': {
        'description': 'Profiled (Fit 0.0 - Fallback)',
        'rsi_buy': 25.33,
        'rsi_sell': 82.90,
        'stop_loss': 0.038,
        'take_profit': 0.102,
        'satellite_rvol': 3.34,
        'satellite_stop': 0.038,
        'rvol_threshold': 1.0,
        'leverage_cap': 1.61
    },
    'UNI/USDT': {
        'description': 'Profiled (Fit 4.6)',
        'rsi_buy': 36.46,
        'rsi_sell': 77.79,
        'stop_loss': 0.050,
        'take_profit': 0.182,
        'satellite_rvol': 2.12,
        'satellite_stop': 0.050,
        'rvol_threshold': 1.0,
        'leverage_cap': 4.73
    },
    'AAVE/USDT': {
        'description': 'Profiled (Fit 62.0)',
        'rsi_buy': 23.01,
        'rsi_sell': 81.14,
        'stop_loss': 0.052,
        'take_profit': 0.104,
        'satellite_rvol': 3.17,
        'satellite_stop': 0.052,
        'rvol_threshold': 1.0,
        'leverage_cap': 1.53
    },
    'SHIB/USDT': {
        'description': 'Profiled (Fit 90.6!)',
        'rsi_buy': 25.79,
        'rsi_sell': 95.00,
        'stop_loss': 0.025,
        'take_profit': 0.068,
        'satellite_rvol': 3.01,
        'satellite_stop': 0.025,
        'rvol_threshold': 1.0,
        'leverage_cap': 3.78
    },
    'PAXG/USDT': {
        'description': 'Profiled (Fit 37.7)',
        'rsi_buy': 39.74,
        'rsi_sell': 95.00,
        'stop_loss': 0.024,
        'take_profit': 0.127,
        'satellite_rvol': 3.27,
        'satellite_stop': 0.024,
        'rvol_threshold': 1.0,
        'leverage_cap': 4.96,
        'max_allocation': 0.15
    },
    'LINK/USDT': {
        'description': 'Profiled (Fit 45.1)',
        'rsi_buy': 34.43,
        'rsi_sell': 75.96,
        'stop_loss': 0.040,
        'take_profit': 0.076,
        'take_profit': 0.108,
        'satellite_rvol': 2.38,
        'satellite_stop': 0.040,
        'rvol_threshold': 1.0,
        'leverage_cap': 4.61
    },
    'BNB/USDT': {
        'description': 'Profiled (Fit 33.8)',
        'rsi_buy': 18.96,
        'rsi_sell': 90.82,
        'stop_loss': 0.032,
        'take_profit': 0.055,
        'satellite_rvol': 3.64,
        'satellite_stop': 0.032,
        'rvol_threshold': 1.0,
        'leverage_cap': 1.11
    },
    'LTC/USDT': {
        'description': 'Profiled (Fit 8.24)',
        'rsi_buy': 47.77,
        'rsi_sell': 94.18,
        'stop_loss': 0.030,
        'take_profit': 0.055,
        'satellite_rvol': 2.21,
        'satellite_stop': 0.030,
        'rvol_threshold': 1.0,
        'leverage_cap': 2.27
    },
    'XMR/USDT': {
        'description': 'Profiled (Fit 75.1)',
        'rsi_buy': 21.31,
        'rsi_sell': 83.97,
        'stop_loss': 0.014,
        'take_profit': 0.196,
        'satellite_rvol': 4.01,
        'satellite_stop': 0.014,
        'rvol_threshold': 1.0,
        'leverage_cap': 5.00
    },
    'XTZ/USDT': {
        'description': 'Profiled (Fit 90.0!)',
        'rsi_buy': 33.21,
        'rsi_sell': 80.65,
        'stop_loss': 0.010,
        'take_profit': 0.058,
        'satellite_rvol': 2.41,
        'satellite_stop': 0.030,
        'rvol_threshold': 1.0,
        'leverage_cap': 3.30
    },
    'AVAX/USDT': {
        'description': 'Profiled (Fit 29.2)',
        'rsi_buy': 46.02,
        'rsi_sell': 80.30,
        'stop_loss': 0.014,
        'take_profit': 0.129,
        'satellite_rvol': 4.07,
        'satellite_stop': 0.014,
        'rvol_threshold': 1.0,
        'leverage_cap': 2.95
    },
    'DOT/USDT': {
        'description': 'Profiled (Fit 75.7)',
        'rsi_buy': 36.77,
        'rsi_sell': 95.00,
        'stop_loss': 0.013,
        'take_profit': 0.109,
        'satellite_rvol': 2.18,
        'satellite_stop': 0.030,
        'rvol_threshold': 1.0,
        'leverage_cap': 2.13
    },
    'NEAR/USDT': {
        'description': 'Profiled (Fit 47.4)',
        'rsi_buy': 23.04,
        'rsi_sell': 95.00,
        'stop_loss': 0.044,
        'take_profit': 0.169,
        'satellite_rvol': 2.45,
        'satellite_stop': 0.044,
        'rvol_threshold': 1.0,
        'leverage_cap': 1.11
    },
    'PEPE/USDT': {
        'description': 'Profiled (Fit 0.0 - Fallback)',
        'rsi_buy': 47.31,
        'rsi_sell': 80.20,
        'stop_loss': 0.026,
        'take_profit': 0.141,
        'satellite_rvol': 4.62,
        'satellite_stop': 0.026,
        'rvol_threshold': 1.0,
        'leverage_cap': 1.21
    },
    'TAO/USDT': {
        'description': 'Profiled (Fit 50.9)',
        'rsi_buy': 19.54,
        'rsi_sell': 80.92,
        'stop_loss': 0.022,
        'take_profit': 0.056,
        'satellite_rvol': 2.33,
        'satellite_stop': 0.035,
        'rvol_threshold': 1.0,
        'leverage_cap': 1.59
    },
    'XAUT/USDT': {
        'description': 'Profiled (Fit 38.7)',
        'rsi_buy': 45.04,
        'rsi_sell': 89.89,
        'stop_loss': 0.039,
        'take_profit': 0.139,
        'satellite_rvol': 4.90,
        'satellite_stop': 0.039,
        'rvol_threshold': 1.0,
        'leverage_cap': 3.60
    }
}

# === KRAKEN FLEXLINE & AGGRESSIVE GROWTH CONFIGURATION (Moved to Root) ===
FLEXLINE_MAX_UTILIZATION = 0.00  # NO BORROWING on nano account 
LOAN_DETAILS = {
    'ACTIVE_LOAN_AMOUNT': 100.0,
    'LOAN_REPAYMENT_DATE': None,
    'REPAYMENT_RESERVE_PCT': 0.25,
}
AGGRESSIVE_GROWTH_MODE = False  # DISABLED: Nano account safety
# ========================================================================

CONVICTION_DECAY_BASE_HOURS = 48.0
CONVICTION_DECAY_CAPITAL_MULTIPLIER = 5.0

IRON_BANK_ENABLED = True
IRON_BANK_MIN_RESERVE = 50.0
IRON_BANK_RATCHET_PCT = 0.00
IRON_BANK_BUFFER_PCT = 0.05

TOPOLOGY_WARNING_THRESHOLD = 0.0001
TOPOLOGY_WINDOW_SIZE = 50

GC_INTERVAL_CYCLES = 10
GC_STALE_ORDER_TIMEOUT = 180
GC_LOG_VERBOSE = True

API_MAX_RETRIES = 15
API_HIBERNATION_TIME = 60
API_RETRY_JITTER_MIN = 1.0
API_RETRY_JITTER_MAX = 3.0
API_RATE_LIMIT_COOL = 10.0

import json
try:
    _user_cfg_path = os.path.join(os.getcwd(), 'user_config.json')
    if os.path.exists(_user_cfg_path):
        with open(_user_cfg_path, 'r') as f:
            _user_cfg = json.load(f)

        if 'max_allocation' in _user_cfg:
            GOVERNOR_MAX_MARGIN_PCT = float(_user_cfg['max_allocation'])
        if 'leverage_cap' in _user_cfg:
            PREDATOR_LEVERAGE = float(_user_cfg['leverage_cap'])
        if 'micro_mode' in _user_cfg:
            MICRO_CAPITAL_MODE = bool(_user_cfg['micro_mode'])

except Exception as e:
    print(f">> [Config] Warning: Failed to load user_config.json: {e}")

# FIX 2026-03-09: Removed "SYSTEM RESCUE" that was forcing Micro-Mode OFF
# MICRO_CAPITAL_MODE is now allowed to be controlled by users/logic.

def calculate_nano_position(balance: float, symbol: str, price: float) -> dict:
    """
    Calculates safety-first position size for Nano/Micro accounts.
    FIX 2026-03-21: Hard cap notional at MAX_ORDER_VALUE ($15).
    DATA: trades >$15 notional = -$982 PnL / 31.6% WR. Smaller = better.
    Uses NANO_MAX_LEVERAGE (2x) and ASSET_ALLOCATION_WEIGHTS for per-symbol scaling.
    """
    leverage = NANO_MAX_LEVERAGE  # 2.0 — constitutional cap

    # Per-symbol allocation weight (Tier 0 winners get 2x, Tier 3 get 0.4x)
    weight = ASSET_ALLOCATION_WEIGHTS.get(symbol, ASSET_ALLOCATION_WEIGHT_DEFAULT)

    # Base notional: 15% of equity, scaled by symbol weight
    base_notional = balance * 0.15 * weight * leverage

    # Hard cap: MAX_ORDER_VALUE ($15) — DATA shows >$15 loses money
    hard_cap = MAX_ORDER_VALUE
    target_notional = min(base_notional, hard_cap)

    # Exchange minimum floor
    base_asset = symbol.split('/')[0]
    min_qty = MIN_TRADE_QTY.get(base_asset, 0.0)
    exchange_min_notional = min_qty * price
    target_notional = max(target_notional, exchange_min_notional)

    # Re-apply hard cap after exchange minimum (if exchange min > cap, skip trade)
    if target_notional > hard_cap * 1.2:  # 20% grace for exchange minimums
        return {
            'quantity': 0, 'leverage': leverage, 'notional': 0, 'margin': 0,
            'meets_atlas_minimum': False, 'atlas_min_trade': 0, 'atlas_min_notional': 0,
        }

    # Final calculations
    quantity = target_notional / price if price > 0 else 0
    margin = target_notional / max(leverage, 1.0)

    return {
        'quantity': quantity,
        'leverage': leverage,
        'notional': target_notional,
        'margin': margin,
        'meets_atlas_minimum': target_notional >= MIN_ORDER_VALUE,
        'atlas_min_trade': MIN_ORDER_VALUE,
        'atlas_min_notional': target_notional,
    }

RISK_MIN_MARGIN_LEVEL = 1.5
RISK_MAX_MICRO_LEVERAGE = 1.5
RISK_MIN_BASE_NOTIONAL = 25.0  # FIX 2026-03-18: Match Atlas minimum (was 10.0)
RISK_MAX_ENTROPY_VETO = 2.1    # 2026-03-21: Aligned with ENTRY_MAX_ENTROPY. 2.0 blocked top performers (TAO, BNB).

# ============================================================================
# ASSET ENTROPY TIERS  (2026-03-19 — empirical from live scout snapshot)
# Source: SampleEntropy on close prices, 15-asset universe
# Tiers recalibrated periodically by ChronosHolon._recalibrate_entropy_tiers()
# ============================================================================
ASSET_ENTROPY_TIERS = {
    # --- ORDERED (SampleEntropy ≤ 0.70) ---
    # Structured / trending assets. Full size, lower conviction floor.
    # Scouter action: trend_following
    'ORDERED': ['PEPE/USDT', 'DOT/USDT', 'LDO/USDT', 'WIF/USDT', 'ADA/USDT'],

    # --- TRANSITION (0.70 < SampleEntropy ≤ 1.10) ---
    # Normal tradeable conditions. Standard ops, moderate conviction.
    # Scouter action: standard
    'TRANSITION': ['SEI/USDT', 'BONK/USDT', 'XRP/USDT', 'LTC/USDT', 'SHIB/USDT',
                   'APT/USDT', 'AAVE/USDT', 'IMX/USDT', 'TAO/USDT'],

    # --- CHAOTIC (SampleEntropy > 1.10) ---
    # High structural complexity. Dampened size, higher conviction floor.
    # Scouter action: mean_reversion_or_pass
    'CHAOTIC': ['XTZ/USDT'],
}

# Reverse lookup: symbol -> tier
ASSET_ENTROPY_TIER_MAP: dict = {
    sym: tier
    for tier, syms in ASSET_ENTROPY_TIERS.items()
    for sym in syms
}

# Tier-based conviction floor overrides (stacked on SYMBOL_CONVICTION_FLOORS)
ASSET_TIER_CONVICTION_MODIFIER = {
    'ORDERED':     -0.05,   # Slightly lower floor — structure is reliable
    'TRANSITION':   0.00,   # Neutral
    'CHAOTIC':     +0.05,   # Higher bar — need stronger signal to enter
}

# Tier-based position size multipliers
ASSET_TIER_SIZE_MODIFIER = {
    'ORDERED':    1.10,   # 10% bonus size in trending conditions
    'TRANSITION': 1.00,   # Standard
    'CHAOTIC':    0.70,   # 30% reduction in high-complexity conditions
}

FLEXLINE_ENABLED = False

FLEXLINE_MAX_UTILIZATION = 0.50
FLEXLINE_EMERGENCY_RESERVE = 0.20
FLEXLINE_MIN_NET_APY = 50.0
FLEXLINE_MAX_HOURLY_RATE = 0.0002

FLEXLINE_COLLATERAL_LTV = {
    'BTC': 0.70,
    'XBT': 0.70,
    'ETH': 0.65,
    'USDT': 0.90,
    'USDC': 0.90,
}

FLEXLINE_LIQUIDATION_LTV = 0.80
FLEXLINE_WARNING_LTV = 0.65
FLEXLINE_AUTO_REPAY_LTV = 0.70

FLEXLINE_ARB_ALLOCATION_PCT = 0.50
FLEXLINE_DIRECTIONAL_ALLOCATION_PCT = 0.30
FLEXLINE_EMERGENCY_ALLOCATION_PCT = 0.20

FLEXLINE_INTEREST_SYNC_INTERVAL = 3600
FLEXLINE_COMPOUND_INTERVAL = 86400

FLEXLINE_DASHBOARD_UPDATE_INTERVAL = 60

FLEXLINE_INJECT_INTO_GOVERNOR = True
FLEXLINE_INJECT_INTO_ARB = True
FLEXLINE_INJECT_INTO_EXECUTOR = True

FLEXLINE_AUTO_REPAY_ENABLED = True
FLEXLINE_LIQUIDATION_ALERT_ENABLED = True
FLEXLINE_RATE_ALERT_ENABLED = True

# ========================================================================
# KRAKEN FLEXLINE & AGGRESSIVE GROWTH CONFIGURATION (Moved to Root) ===
FLEXLINE_MAX_UTILIZATION = 0.00  # NO BORROWING on nano account 
LOAN_DETAILS = {
    'ACTIVE_LOAN_AMOUNT': 100.0,
    'LOAN_REPAYMENT_DATE': None,
    'REPAYMENT_RESERVE_PCT': 0.25,
}
AGGRESSIVE_GROWTH_MODE = False  # DISABLED: Nano account safety
# ========================================================================

# ========================================================================
# SMCE v1 - SOVEREIGN MICRO-COMPOUNDING ENGINE (CONSTITUTIONAL CONSTANTS)
# ========================================================================
SMCE_TIER_SMALL_MAX   = 500.0
SMCE_TIER_MEDIUM_MAX  = 5000.0

SMCE_SMALL_MAX_TOTAL_EXPOSURE = 0.30
SMCE_SMALL_MAX_PER_ASSET      = 0.12
SMCE_SMALL_MAX_CLUSTER        = 0.15
SMCE_SMALL_MAX_LEVERAGE = {"HARVEST":3.0,"EXPANSION":3.0,"TRANSITION":2.0,"DEFENSIVE":1.0}

SMCE_MEDIUM_MAX_TOTAL_EXPOSURE = 0.40
SMCE_MEDIUM_MAX_PER_ASSET      = 0.15
SMCE_MEDIUM_MAX_CLUSTER        = 0.25
SMCE_MEDIUM_MAX_LEVERAGE = {"HARVEST":4.0,"EXPANSION":5.0,"TRANSITION":3.0,"DEFENSIVE":1.0}

SMCE_DAILY_DD_LIMIT       = 0.03
SMCE_WEEKLY_DD_LIMIT      = 0.06
SMCE_DEFENSIVE_COOLDOWN_H = 48
SMCE_RISK_MULT_DEFENSIVE  = 0.5

SMCE_HARVEST_MIN_SCORE    = 5
SMCE_EXPANSION_MIN_SCORE  = 6
SMCE_TRANSITION_MIN_SCORE = 4
SMCE_TRANSITION_SIZE_MOD  = 0.5

SMCE_MC_PATHS              = 1000
SMCE_MC_VETO_DRAWDOWN_PROB = 0.10
SMCE_MC_VETO_CVAR          = 0.04
SMCE_MC_VETO_LIQ_PROB      = 0.01

SMCE_SCALING_MIN_CLEAN_DAYS  = 60
SMCE_SCALING_MAX_DRAWDOWN    = 0.08
SMCE_SCALING_MAX_WK_VARIANCE = 0.05
SMCE_SCALING_ALLOC_BOOST     = 0.02
# ========================================================================

# ========================================================================
# ARBITRAGE & GOLD CONFIGURATION
# ========================================================================
ENABLE_XSTOCKS_ARB = False
GOLD_LEAD_LAG_ENABLED = True
GOLD_LEAD_LAG_MIN_SPREAD = 0.004
GOLD_LEAD_LAG_MAX_SPREAD = 0.012
GOLD_LEAD_LAG_MIN_MOVE = 0.003
GOLD_LEAD_LAG_COOLDOWN = 120
GOLD_LEAD_LAG_MAX_LAG = 45

PAXG_BTC_ENABLED = True
PAXG_BTC_ZSCORE_ENTRY = 2.0
PAXG_BTC_ZSCORE_EXIT = 0.5
PAXG_BTC_ZSCORE_STOP = 3.0
PAXG_BTC_LOOKBACK_DAYS = 90
PAXG_BTC_MIN_DATA_POINTS = 1000
PAXG_BTC_TRADE_COOLDOWN = 3600
PAXG_BTC_MAX_POSITION_PCT = 0.10
PAXG_BTC_EXCHANGE = 'kucoin'
# ========================================================================

FORCE_HARD_SYNC_ON_STARTUP = False

# ========================================================================
# CHRONOS FORENSICS FIXES (2026-03-15)
# ========================================================================
# REST-Only Mode: Disable WebSocket entirely for stable data feeds
# (WebSocket latencies exceeded 27 minutes in production)
WS_FORCE_REST_ONLY = True  # Force REST API for all price feeds
WS_UNHEALTHY_THRESHOLD = 0.5  # % of unhealthy connections to trigger REST fallback

# Dynamic Stop-Loss: Asset-specific risk based on volatility
# High-volatility assets (BTC, ETH, SOL) get tighter stops
DYNAMIC_STOP_LOSS_ENABLED = True
# FIX 2026-03-16 (Chronos): SOL/USDT -302% loss - stops too tight causing frequent stop-outs
# Old: 0.5% stop on 5x leverage = 2.5% equity risk per trade (too tight for SOL volatility)
# New: 1.5% stop on 3x leverage = 4.5% equity risk (more realistic)
DYNAMIC_STOP_LOSS_HIGH_VOL = 0.015  # Increased from 0.005 (0.5%) to 1.5% for BTC, ETH, SOL
DYNAMIC_STOP_LOSS_MED_VOL = 0.015  # 1.5% for mid-cap alts
DYNAMIC_STOP_LOSS_LOW_VOL = 0.025  # 2.5% for stable/low-vol assets
HIGH_VOLATILITY_ASSETS = {'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT'}

# Conviction Threshold: Filter marginal signals
# 2026-03-23 FIX: Reduced from 0.65 to 0.50 to address 0% signal pass rate
# Analysis: 4839 signals blocked, 0 executed - system was over-protected
# Structure Boss and Governor already filter weak signals, so base threshold can be lower
MINIMUM_CONVICTION_THRESHOLD = 0.50  # 2026-03-23 Fix: Allow more signals through for testing

# Loss Streak Protection (2026-03-23)
# Assets with chronic loss streaks get additional scrutiny
LOSS_STREAK_SUSPEND_THRESHOLD = 5  # Suspend after 5 consecutive losses on same asset
LOSS_STREAK_SUSPEND_HOURS = 24     # Suspension duration
CHRONIC_LOSS_ASSETS = {
    'XTZ/USDT': {'max_consecutive': 3, 'suspended': False},  # 14 loss streak observed
    'WIF/USDT': {'max_consecutive': 3, 'suspended': False},  # 12 loss streak observed
    'DOT/USDT': {'max_consecutive': 3, 'suspended': False},  # 11 loss streak observed
    'LDO/USDT': {'max_consecutive': 4, 'suspended': False},  # 8 loss streak observed
}

# Daily Loss Limit: Circuit breaker to prevent death-by-a-thousand-cuts
DAILY_LOSS_LIMIT_ENABLED = True
DAILY_LOSS_LIMIT_USD = 5.0  # Hard dollar limit
DAILY_LOSS_LIMIT_PCT = 0.05  # 5% of equity
DAILY_LOSS_COOLDOWN_HOURS = 24  # Trading halt after hitting limit

# High-volatility stop-loss override (overrides DEFAULT_STOP_LOSS_PCT)
# FIX 2026-03-16 (Chronos): Aligned with DYNAMIC_STOP_LOSS_HIGH_VOL
HIGH_VOL_STOP_LOSS_PCT = 0.015  # Increased from 0.0075 (0.75%) to 1.5% for BTC/ETH/SOL

# SOL/USDT Specific Risk Controls (FIX 2026-03-16: -302% loss investigation)
# SOL had massive losses due to: 1) tight stops (0.5%), 2) high leverage (5x), 3) stacked positions
SOL_USDT_MAX_LEVERAGE = 2.0     # Special cap for SOL (overrides PREDATOR_LEVERAGE)
SOL_USDT_MAX_POSITIONS = 1      # Only 1 position at a time (no stacking)
SOL_USDT_STOP_LOSS = 0.02       # 2% stop specifically for SOL
# ========================================================================

# ========================================================================
# QUANT-OPS MULTI-AGENT ARCHITECTURE (2026-03-19)
# ========================================================================
# Coordinated intelligence loop: Chronos → Aegis → Helix → Atlas
# Runs after every QUANTOPS_CYCLE_INTERVAL trading cycles.
QUANTOPS_ENABLED = True
QUANTOPS_CYCLE_INTERVAL = 5       # Run intelligence cycle every N trade cycles
QUANTOPS_MEMORY_DEPTH = 10        # Number of prior reports to feed as context
QUANTOPS_LLM_MODE = False         # False = deterministic Python agents; True = optional LLM narratives
QUANTOPS_OUTPUT_DIR = "quant_ops_reports"  # Directory for timestamped cycle JSON reports

# Atlas capital allocation (mutated at runtime by QuantOps/Atlas agent)
ATLAS_BUY_ALLOCATION = 0.80       # Fraction of capital allocated to BUY strategy
ATLAS_RESERVE_ALLOCATION = 0.20   # Fraction held in reserve
# ========================================================================
