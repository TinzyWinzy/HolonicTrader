"""
NEXUS Configuration (Phase 15) - NANO MODE EDITION

CENTRAL STORAGE for all thresholds, leverage caps, and system parameters.
UPDATED 2026-01-13 for NANO accounts (< $50)
FIXED: Position sizing, leverage, and margin calculations for $25 account
"""

from dotenv import load_dotenv
import os

load_dotenv()

# === API KEYS ===
KRAKEN_FUTURES_API_KEY = os.getenv('KRAKEN_FUTURES_API_KEY')
KRAKEN_FUTURES_PRIVATE_KEY = os.getenv('KRAKEN_FUTURES_PRIVATE_KEY')
KRAKEN_SPOT_KEY = os.getenv('KRAKEN_API_KEY')
KRAKEN_SPOT_SECRET = os.getenv('KRAKEN_PRIVATE_KEY')

# Active Keys
API_KEY = KRAKEN_FUTURES_API_KEY or KRAKEN_SPOT_KEY
API_SECRET = KRAKEN_FUTURES_PRIVATE_KEY or KRAKEN_SPOT_SECRET

# === TELEGRAM INTEGRATION ===
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

# === ACCOUNT PARAMETERS ===
SCAVENGER_THRESHOLD = 90.0                # Balance <= this = Scavenger Mode
INITIAL_CAPITAL = 300.0                   # Reset for Paper Trading (2026-01-14)
MISSION_TARGET = 3000.0                   # GOAL: Grow to $3000
MISSION_NAME = "Operation Paper Centurion"
PRINCIPAL = 100.0                         # Floor for capital protection (Lowered for buying power)
PAPER_TRADING = True                      # PAPER MODE for test run

# === TIME SETTINGS ===
TIMEFRAME = '15m'                         # Default trading timeframe
DEFAULT_CYCLE_INTERVAL = 30               # Faster Response (User Request)

# === SYSTEM FLAGGING ===
ENABLE_EVOLUTION = False                  # DANGER: Disable "Brain Transplants" in Live Mode
DEFAULT_STOP_LOSS_PCT = 0.05              # 5% Hard Stop per position
DEFAULT_TAKE_PROFIT_PCT = 0.10            # 10% Take Profit target


# === TRADING MODE ===
TRADING_MODE = 'FUTURES'

# === KRAKEN SYMBOL MAPPING (FUTURES) ===
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
}

# === ASSET CONSTRAINTS (MOVED UP FOR DEPENDENCIES) ===
# UNLEASHED: Allow ALL mapped assets
ALLOWED_ASSETS = list(KRAKEN_SYMBOL_MAP.keys())
# ALLOWED_ASSETS = [
#     'BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'SOL/USDT', 'DOGE/USDT',
#     'ADA/USDT', 'LINK/USDT', 'LTC/USDT', 'XTZ/USDT', 'AVAX/USDT',
#     'DOT/USDT', 'PAXG/USDT', 'TAO/USDT', 'XAUT/USDT',
# ]

# === SCOUT ARCHITECTURE ===
# The "Hot" List: Assets actively being traded/analyzed deeply
ACTIVE_WATCHLIST = ALLOWED_ASSETS.copy()

# === TRINITY STRATEGY PREFERENCES ===
# User Control: Define which assets play which role in the strategy
ASSET_PREF_PREDATOR = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'SUI/USDT', 'TAO/USDT', 'XMR/USDT', 'AVAX/USDT', 'BNB/USDT', 'NEAR/USDT', 'AAVE/USDT']   # High Conviction Trenders
ASSET_PREF_SCAVENGER = ['XRP/USDT', 'DOGE/USDT', 'PEPE/USDT', 'SHIB/USDT', 'ADA/USDT', 'LINK/USDT', 'DOT/USDT', 'XTZ/USDT', 'LTC/USDT', 'UNI/USDT', 'PAXG/USDT'] # Volatility Harvesters & Defensive

# Kraken Futures Minimums (verified)
MIN_TRADE_QTY = {
    'BTC': 0.0001,   # ~$9 at $90k BTC
    'ETH': 0.001,    # ~$3 at $3k ETH
    'SOL': 0.01,     # ~$0.15 at $15 SOL
    'XRP': 1.0,      # ~$0.50
    'ADA': 1.0,      # ~$0.50
    'DOGE': 10.0,    # ~$0.50
    'LTC': 0.1,      # ~$7
}

TICK_SIZES = {
    'BTC': 0.50,
    'ETH': 0.10,
    'SOL': 0.01,
    'ADA': 0.00001,
    'XRP': 0.00001,
}

# === REGIME SYSTEM (CAPITAL-BASED) ===
# Updated thresholds for realistic progression
REGIME_NANO_CEILING = 49.0                # NANO: $0 - $49 (YOU ARE HERE)
REGIME_MICRO_CEILING = 249.0              # MICRO: $50 - $249
REGIME_SMALL_CEILING = 999.0              # SMALL: $250 - $999
REGIME_MEDIUM_CEILING = 9999.0            # MEDIUM: $1000+

# Promotion/Demotion Rules
REGIME_PROMOTION_STABILITY_HOURS = 72
REGIME_PROMOTION_HEALTH_THRESHOLD = 0.95
REGIME_PROMOTION_MIN_TRADES = 20
REGIME_DEMOTION_BUFFER = 5.0
REGIME_DEMOTION_DRAWDOWN_PCT = 0.25

# === REGIME-SPECIFIC PERMISSIONS (UPDATED FOR NANO SAFETY) ===
# === REGIME-SPECIFIC PERMISSIONS (UPDATED FOR NANO SAFETY) ===
REGIME_PERMISSIONS = {
    'NANO': {  # FOR $0-$49 ACCOUNTS (AGGRESSIVE MODE)
        'max_positions': 5,               # Allow Multi-Asset (5 slots)
        'max_stacks': 0,                  # No stacking allowed
        'max_exposure_ratio': 1.0,        # Max 1x total exposure (SAFETY CAP)
        'max_leverage': 1.0,              # HARD CAP: 1x (No Leverage)
        'allowed_pairs': ALLOWED_ASSETS,  # Trade EVERYTHING
        'correlation_check': True,        # Enable correlation check
        'min_order_value': 2.0,           # $2 minimum position
        'max_order_value_pct': 0.40,      # Max 40% of capital per position
        'cooldown_after_failure': 60,     # Fast retry
    },
    'MICRO': {  # UNLOCKED FOR ACTIVE TRADING
        'max_positions': 5,               # 5 positions allowed (was 2)
        'max_stacks': 0,                  # Still no pyramiding
        'max_exposure_ratio': 1.0,        # Max 1x total exposure (SAFETY CAP)
        'max_leverage': 1.0,              # HARD CAP: 1x (No Leverage)
        'allowed_pairs': ALLOWED_ASSETS,  # Trade all assets (was limited)
        'correlation_check': True,        # Keep cluster risk
        'min_order_value': 3.0,           # $3 minimum
        'max_order_value_pct': 0.30,      # 30% of capital
        'cooldown_after_failure': 120,
    },
    'SMALL': {
        'max_positions': 15,              # Increased from 12 for better busket utilization
        'max_stacks': 10,                 # Increased from 6 to capture trending moves (e.g. XMR)
        'max_exposure_ratio': 1.0,        # Max 1x total exposure (SAFETY CAP)
        'max_leverage': 1.0,              # HARD CAP: 1x (No Leverage)
        'allowed_pairs': ALLOWED_ASSETS,
        'correlation_check': True,
        'min_order_value': 10.0,
        'max_order_value_pct': 0.25,
        'cooldown_after_failure': 120,
    },
    'MEDIUM': {
        'max_positions': 24,
        'max_stacks': 2,
        'max_exposure_ratio': 1.0,        # Max 1x total exposure (SAFETY CAP)
        'max_leverage': 1.0,              # HARD CAP: 1x (No Leverage)
        'allowed_pairs': ALLOWED_ASSETS,
        'correlation_check': True,
        'min_order_value': 20.0,
        'max_order_value_pct': 0.30,
        'cooldown_after_failure': 60,
    },
}

# === SECTOR DEFINITIONS (Holistic Phase 5b) ===
MEMECOIN_ASSETS = ['DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT']
MEMECOIN_PUMP_RVOL = 3.0
POLYMARKET_PATIENCE_MINUTES = 3 

# === MACRO STRATEGY LISTS ===
CRISIS_SAFE_HAVENS = ['BTC/USDT', 'PAXG/USDT']
CRISIS_RISK_ASSETS = ['DOGE/USDT', 'ADA/USDT', 'SOL/USDT']

# === VOLATILITY WINDOW (High-Entropy Regime) ===
VOL_WINDOW_RISK_PCT = 0.02           # 2% Risk per trade
VOL_WINDOW_LEVERAGE = 5              # Max 5x Leverage
VOL_WINDOW_CYCLE_INTERVAL = 15       # 15s Cycle
VOL_WINDOW_BTC_VOL_THRESHOLD = 0.45  # > 45% Annualized Volatility
VOL_WINDOW_FUNDING_THRESHOLD = 0.0003 # > 0.03% per 8h
VOL_WINDOW_SPREAD_THRESHOLD = 0.004  # < 0.4% Spread required
VOL_WINDOW_MIN_BALANCE_SHUTOFF = 5.0 # Stop if balance < $5
VOL_WINDOW_TARGET_PROFIT = 0.39      # +39% Daily Target
VOL_WINDOW_MAX_POSITIONS = 3         # Max 3 concurrent positions
VOL_WINDOW_GROSS_RISK_CAP = 0.06     # Max 6% NAV Gross Risk
VOL_WINDOW_MIN_VOLATILITY = 0.40       # Minimum Volatility to activate window


# === SATELLITE COMMANDER (TIER 2) ===
# 🧬 APEX PREDATOR (NANO CHALLENGE 2026-01-14)
# Goal: $17.90 -> $100.00 | Achieved in Sim: $40.26 (+125% ROI)
SATELLITE_ASSETS = ['SOL/USDT', 'DOGE/USDT', 'ADA/USDT'] 
SATELLITE_MARGIN = 17.0      # Approx All-In on $17 Account (Nano Mode)
SATELLITE_LEVERAGE = 1.0     # HARD CAP: 1x (Safety First)

# Tuned for Tighter Stacking
GOVERNOR_MIN_STACK_DIST = 0.0025 # Reduced to 0.25% (Was 0.5%)

# Reduced Concentration for Micro Accounts
MICRO_MAX_POSITIONS = 3

# Evolved Genome (RELAXED FOR RANGING MARKETS)
# Evolved Genome (DOGE SNIPER PROFILE 2026-01-23)
SATELLITE_RVOL_THRESHOLD = 6.72   # Evolved: Only trigger Satellite on MASSIVE PUMPS
SATELLITE_DOGE_RVOL_THRESHOLD = 6.72  # Match global for now, specific to DOGE
SATELLITE_BBW_EXPANSION_THRESHOLD = 0.24 # Evolved: 0.24
SATELLITE_ENTRY_RSI_CAP = 40.0  # Relaxed from 33.0 to catch more dips

# === STRATEGY SETTINGS (RELAXED FOR RANGING MARKETS) ===
STRATEGY_RSI_OVERSOLD = 40            # Relaxed from 35 (More Entries)
STRATEGY_RSI_OVERBOUGHT = 65          # Relaxed from 65
STRATEGY_RSI_PANIC_BUY = 30.0         # Relaxed from 40.0
STRATEGY_RSI_ENTRY_MAX = 65.0         # Relaxed from 60.0 (Allow momentum entries)


# Risk Management
SATELLITE_BREAKEVEN_TRIGGER = 0.02 # Move SL at +2%
SATELLITE_TAKE_PROFIT_1 = 0.10 # Evolved: +10% Target
SATELLITE_STOP_LOSS = 0.026 # Evolved: -2.6% Stop (Tight)

# === LEVERAGE SETTINGS (SAFETY FIRST) ===
# Global maximums - actual limits set by regime
SCAVENGER_LEVERAGE = 1.0                  # Cap at 1x
PREDATOR_LEVERAGE = 1.0                     # Cap at 1x
MICRO_HARD_LEVERAGE_LIMIT = 1.0             # Cap at 1x
PREDATOR_TRAILING_STOP_ATR_MULT = 3.5     # 3.5x ATR Trailing Stop (Widened from 2.0 for volatility)


# === POSITION SIZING & RISK (UNLEASHED) ===
MIN_ORDER_VALUE = 2.0                     # Lowered from 5.0 for NANO accounts
MAX_RISK_PCT = 0.05                       # UNLEASHED: 5% Risk per trade (Was 2%)
SCAVENGER_MAX_MARGIN = 0.85               # UNLEASHED: Max 85% Margin (Was 80%)
SCAVENGER_STOP_LOSS = 0.10                # UNLEASHED: 10% stop loss (Was 5%)
SCAVENGER_SCALP_TP = 0.06                 # UNLEASHED: 6% take profit (Was 4%) to Capture bigger swings
PREDATOR_STOP_LOSS = 0.08                 # UNLEASHED: 8% stop loss (Was 5%)
PREDATOR_TAKE_PROFIT = 0.12               # UNLEASHED: 12% take profit (Was 6%) - Let winners RUN

# === RAPID PROFIT TAKING ("SCALP THE WHALES") ===
# Enhanced Targets (Phase 3 Optimization)
PROFIT_TARGETS = {
    'WHALE_BID_WALL': {
        'rapid': 0.010,    # 1.0% (Was 0.5%) - Whales move big
        'normal': 0.025,   # 2.5%
        'runner': 0.050    # 5.0%
    },
    'WHALE_ACCUMULATION': {
        'rapid': 0.015,    # 1.5%
        'normal': 0.040,   # 4.0%
        'runner': 0.080    # 8.0%
    },
    'PACK_HUNT': {
        'rapid': 0.005,    # 0.5% (Was 0.3%) - Increased base
        'normal': 0.012,   # 1.2%
        'runner': 0.025    # 2.5%
    },
    'DIP': {
        'rapid': 0.008,    # 0.8% (Was 0.4%)
        'normal': 0.020,   # 2.0%
        'runner': 0.040    # 4.0%
    },
    'DEFAULT': {           # Fallback
        'rapid': 0.005,    # 0.5% Base Scalp
        'normal': 0.015,   # 1.5%
        'runner': 0.030    # 3.0%
    }
}

EXIT_PYRAMID = {
    'aggressive': [0.3, 0.4, 0.3],  # 30%, 40%, 30%
    'balanced':   [0.2, 0.5, 0.3],
    'conservative': [0.1, 0.6, 0.3]
}

IMMEDIATE_SCALP_CONFIG = {
    'rsi_overbought': 80.0,
    'profit_target': 0.003,
    'position_size_ratio': 0.5
}

TIME_BASED_PROFIT_TARGETS = {
    'asian_session': {'rapid': 0.003, 'normal': 0.008, 'runner': 0.015},   # 0-8h: Quiet, tight scalps
    'london_session': {'rapid': 0.005, 'normal': 0.012, 'runner': 0.025},  # 8-16h: Volatility, wider targets
    'ny_session': {'rapid': 0.004, 'normal': 0.010, 'runner': 0.020},      # 16-24h: Mixed, medium targets
    'weekend': {'rapid': 0.002, 'normal': 0.005, 'runner': 0.010}          # Weekend: Very tight
}

# Adjusted Stacking Targets (Phase 3)
STACK_PROFIT_TARGETS = {
    1: {'target': 0.008, 'stop': 0.004},   # 0.8% (Was 0.5%)
    2: {'target': 0.015, 'stop': 0.008},   # 1.5% (Was 0.8%)
    3: {'target': 0.025, 'stop': 0.015},   # 2.5%
    4: {'target': 0.040, 'stop': 0.020}    # 4.0%
}

# Market Condition Adjustments
MARKET_ADJUSTMENTS = {
    'crisis_score': {
        'critical': 0.8,
        'reduce_targets': 0.7  # Reduce targets by 30% in crisis
    },
    'sentiment': {
        'bearish': 0.0,
        'bullish': 0.2
    }
}

# Position Management Limits
POSITION_LIMITS = {
    'max_positions': 8,
    'max_correlation': 0.3,
    'size_limits': {
        # Using simple symbol checks or generic logic for now
        'DEFAULT': 0.10,     # Max 10% NV per asset
        'LARGE_CAP': 0.15,   # BTC/ETH/SOL can be 15%
        'MEME': 0.05         # PEPE/DOGE max 5%
    },
    'daily_turnover_limit': 3.0
}

# === MICRO-GUARD HARD LIMITS (UNLEASHED) ===
MICRO_GUARD_PORTFOLIO_NOTIONAL_MULT = 3.0  # Restored to 3.0 (Was 1.5)
MICRO_GUARD_SINGLE_NOTIONAL_MULT = 1.5     # Increased to 1.5 (Was 0.75)
MICRO_GUARD_GROSS_LEVERAGE_MULT = 3.0      # Restored to 3.0 (Was 1.5)
MICRO_GUARD_GROSS_LEVERAGE = 10.0          # UNLEASHED: 10x Cap (Was 3.0)


# Emergency protection
MICRO_GUARD_CASH_PRESERVATION_THRESHOLD = 10.0
MICRO_GUARD_CASH_PRESERVATION_LEVERAGE = 2.0

# === ORDER FAILURE PROTECTION ===
ORDER_FAILURE_COOLDOWN = {
    'insufficientAvailableFunds': 300,     # 5 minutes
    'wouldNotReducePosition': 60,          # 1 minute
    'network': 10,                         # 10 seconds
    'default': 120,                        # 2 minutes for other errors
}

MAX_CONSECUTIVE_FAILURES = 3               # Stop after 3 consecutive failures

# === SCOUT SETTINGS ===
SCOUT_CANDIDATES = ALLOWED_ASSETS  # Keep it simple for NANO
SCOUT_RVOL_THRESHOLD = 2.5
SCOUT_HYPE_THRESHOLD = 0.5
SCOUT_CACHE_TTL = 60
SCOUT_CYCLE_INTERVAL = 300  # Increased to 5min to prevent system hangs

# === STRATEGY SETTINGS (RELAXED FOR RANGING MARKETS) ===
STRATEGY_RSI_OVERSOLD = 35            # Relaxed from 45
STRATEGY_RSI_OVERBOUGHT = 65          # Relaxed from 70
STRATEGY_RSI_PANIC_BUY = 30.0         # Relaxed from 40.0
STRATEGY_RSI_ENTRY_MAX = 60.0         # Relaxed from 70.0
STRATEGY_LSTM_THRESHOLD = 0.52        # Slightly raised for more entries
STRATEGY_XGB_THRESHOLD = 0.50         # Slightly raised from 0.48
STRATEGY_POST_EXIT_COOLDOWN_CANDLES = 2  # Faster re-entry

# === GOVERNANCE / RISK ===
GOVERNOR_COOLDOWN_SECONDS = 60    # Relaxed from 300s (5m) to 60s (1m) for faster re-entry
GOVERNOR_MIN_STACK_DIST = 0.001   # 0.1% (Base). Now scaled dynamically by Governor (ATR).
STACK_TIMEOUT_SECONDS = 900       # 15 minutes (Relaxed from 5m) - Max time to hold before force-reduce check
GOVERNOR_MAX_MARGIN_PCT = 0.85  # UNLEASHED: 85% Utiliization
GOVERNOR_STACK_DECAY = 0.8
GOVERNOR_MAX_TREND_AGE_HOURS = 24.0
GOVERNOR_TREND_DECAY_START = 12.0
# === VOLATILITY ===
ATR_PERIOD = 14
ATR_STORM_MULTIPLIER = 3.0
ATR_STOP_LOSS_MULTIPLIER = 3.0 # UNLEASHED: 3.0x ATR Wide Stop (Was 2.0)
KELLY_LOOKBACK = 20 # Number of trades to calculate Win Rate for Kelly Criterion
VOL_SCALAR_MIN = 0.5 # Min Volatility Scalar (High Vol = 0.5x size)
VOL_SCALAR_MAX = 2.0 # Max Volatility Scalar (Low Vol = 2.0x size)

# === BOLLINGER BANDS ===
BB_PERIOD = 20
BB_STD = 2

# === SENTIMENT & NEWS ===
# === SENTIMENT & NEWS ===
SENTIMENT_SOURCES = [
    'https://cointelegraph.com/rss',
    'https://www.coindesk.com/arc/outboundfeeds/rss/',
    'https://cryptopanic.com/news/rss/',
    'https://beincrypto.com/feed/',
    # === REDDIT (THE HIVE MIND) ===
    'https://www.reddit.com/r/CryptoCurrency/top/.rss?t=hour',  # Breaking & Viral
    'https://www.reddit.com/r/Bitcoin/hot/.rss',                # King Sentiment
    'https://www.reddit.com/r/solana/hot/.rss',                 # Degen Sentiment
    # === MACRO SOURCES ===
    'https://www.cnbc.com/id/100727362/device/rss/rss.html',
    'https://www.investing.com/rss/commodities.rss'
]

# === PHASE 39: MARKET PHYSICS (VALIDATION LAYER) ===
PHYSICS_CORRELATION_THRESHOLD = 0.75  # Veto Longs if Corr > 0.75 & BTC Bearish
PHYSICS_MIN_RVOL = 1.0                # Relaxed from 1.1x for better sensitivity in low-vol periods
PHYSICS_MAX_ENTROPY = 1.35            # Veto All Signals if Entropy > 1.35 (Chaos)
PHYSICS_OU_STRETCH_THRESHOLD = 2.0   # Quantum Reversion: Trigger if p > 2 sigma from OU Mean
PHYSICS_OU_LAMBDA_THRESHOLD = 10.0   # Reversion Strength: Consider "High" if lambda > 10
PHYSICS_MAX_RUIN_PROBABILITY = 0.60  # Relaxed from 0.45: Permit high-conviction whale setups

# === CORRELATION GUARD ===
CORRELATION_CHECK = True              # Enable/Disable Correlation Matrix Checks

# === WHALE TRACKING (Project Ahab) ===
WHALE_RVOL_THRESHOLD = 3.0            # 300% Volume Spike = Whale
WHALE_SENTIMENT_WEIGHT = 0.2          # Boost sentiment if whales mentioned
# Ahab Phase 1: Stealth Accumulation
WHALE_ACCUMULATION_RVOL = 2.0         # 200% Volume (Quiet Buying)
WHALE_ACCUMULATION_ATR_FACTOR = 0.8   # Low Volatility (Price Contained)
# Ahab Phase 2: Defense & Squeeze
WHALE_DEFENSE_RVOL = 2.5              # 250% Volume (Defending Level)
WHALE_ORDER_IMBALANCE_RATIO = 2.0     # 2x Buying Pressure (Bid Wall)
WHALE_FUNDING_SQUEEZE_THRESHOLD = -0.00005 # -0.005% Funding (Shorts trapped)
SENTIMENT_WEIGHT = 0.3
SENTIMENT_THRESHOLD_BULL = 0.2
SENTIMENT_THRESHOLD_BEAR = -0.2
ENABLE_SHORT_SELLING = True           # CRITICAL: Enabled for Bear Market Regime

# === CROSS-ASSET CORRELATION ===
GMB_THRESHOLD = 0.40  # Lowered from 0.50 for neutral markets
GMB_EXIT_HYSTERESIS = 0.10  # Exit only if GMB < GMB_THRESHOLD - 0.10 (Fix for DOGE noise)

# === SAFETY THRESHOLDS (Governor) ===
SOLVENCY_PANIC_THRESHOLD = 0.98       # UNLEASHED: 98% (Was 0.95)
SOLVENCY_PANIC_REDUCTION = 0.20       # Reduce to 20% size if panic hit

# === CONSOLIDATION ENGINE ===
CONSOLIDATION_WEIGHT_PNL = 0.30
CONSOLIDATION_WEIGHT_CONVICTION = 0.25
CONSOLIDATION_WEIGHT_LIQUIDITY = 0.15
CONSOLIDATION_WEIGHT_AGE = 0.10
CONSOLIDATION_WEIGHT_CORRELATION = 0.20

CONSOLIDATION_DUST_THRESHOLD = 1.0
CONSOLIDATION_STALE_HOURS = 24.0
CONSOLIDATION_MARGIN_BUFFER = 1.5
CONSOLIDATION_HARD_BUFFER = 1.2

# === ACCUMULATOR ===
ACC_RISK_FLOOR = 0.5                      # Min Risk Multiplier (Defensive)
ACC_RISK_CEILING = 2.0                    # Max Risk Multiplier (Aggressive)
ACC_DRAWDOWN_LIMIT = 0.25                 # Drawdown limit to stop trading (Hard Stop)
ACC_HARD_STOP_LIMIT = 0.40                # Catastrophic Stop (Overrides Session Recovery)
ACC_SANITY_THRESHOLD = 0.30               # >30% instant drop = Likely Data Glitch protection)

# === ARBITRAGE PARAMETERS ===
ARB_SPATIAL_THRESHOLD = 0.003          # 0.3% spread between exchanges (KuCoin vs Kraken)
ARB_FUNDING_THRESHOLD = 0.0001         # 0.01% per 8h (Basis trigger)
ARB_MAX_EXPOSURE = 0.50                # Max 50% of capital for pure arb plays

# === ADVANCED EXECUTION PARAMETERS ===
EXEC_MAX_POV = 0.05                    # Max 5% of Volume (POV algorithm)
EXEC_VWAP_WINDOW = 20                  # Lookback window for VWAP profile
EXEC_TWAP_MAX_DURATION = 3600          # Max 1 hour for TWAP slices
EXEC_IMPACT_THRESHOLD = 0.10           # Max 10% of top-level book liquidity

# === PPO SOVEREIGN BRAIN ===
PPO_LEARNING_RATE = 0.0003
PPO_CLIP_RATIO = 0.2
PPO_REWARD_DRAWDOWN_PENALTY = 2.0

# === CONCURRENCY & RATE LIMITING ===
CCXT_RATE_LIMIT = True
CCXT_POOL_SIZE = 10         # Reduced for NANO accounts
TRADER_MAX_WORKERS = 8      # Reduced for NANO accounts
TRADER_MAX_CYCLE_ENTRIES = 3 # Max new positions per 15m cycle (Crisis Prevention)

# === INTEL GPU ACCELERATION ===
USE_INTEL_GPU = True
USE_OPENVINO = True

# === IMMUNE SYSTEM & PERSONALITY ===
FAMILY_L1 = ['SOL/USDT', 'ADA/USDT', 'AVAX/USDT']
FAMILY_PAYMENT = ['XRP/USDT', 'LTC/USDT']
FAMILY_MEME = ['DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT']

IMMUNE_MAX_DAILY_DRAWDOWN = 0.20        # UNLEASHED: 20% (Was 10%)
IMMUNE_MAX_LEVERAGE_RATIO = 5.0         # Absolute Notional Cap (5x Equity)NANO

# === LIQUIDATION ENGINE ===
MAINTENANCE_MARGIN_RATE = 0.50

# === UNIFIED CONTROL PROTOCOL ===
MICRO_CAPITAL_MODE = True
MICRO_MAX_LEVERAGE = 3.0
MICRO_MAX_EXPOSURE_RATIO = 3.5
# MICRO_MAX_POSITIONS handled above

# Stacking restrictions
STACKING_MIN_EQUITY = 100.0
STACKING_BUFFER_MULTIPLIER = 5.0

# MicroMode refinements
MICRO_STARTUP_MAX_POSITIONS = 2

# === NANO-MODE CONFIG (REALITY ANCHORED) ===
NANO_CAPITAL_THRESHOLD = 50.0
NANO_ALLOCATION_PCT = 0.05           # 5% Max Position Size ($0.77 at $15)
NANO_MAX_LEVERAGE = 1.0              # 1.0x Safety Cap
NANO_MAX_POSITIONS = 1               # Max 1 Position
NANO_COOLDOWN_AFTER_FAILURE = 86400  # 24 Hours (Extreme Punishment)

# === PERSONALITY PARAMETERS ===
PERSONALITY_BTC_ATR_FILTER = 0.5
PERSONALITY_SOL_RSI_LONG = 50.0 # Relaxed from 55.0 for better sensitivity
PERSONALITY_SOL_RSI_SHORT = 45.0
PERSONALITY_DOGE_RVOL = 0.9  # Evolved: Lowered to 0.9 to catch steady trends (Standard Mode)

# === EVOLVED ASSET PROFILES (2026-01-23) ===
ASSET_PROFILES = {
    'BTC/USDT': {
        'description': 'Profiled (Fit 25.5)',
        'rsi_buy': 24.86,
        'rsi_sell': 88.57,
        'stop_loss': 0.042,
        'take_profit': 0.088,
        'satellite_rvol': 4.95,
        'satellite_stop': 0.042, 
        'rvol_threshold': 1.0, # Baseline
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
        'description': 'Trend King (Fit 57.9)',
        'rsi_buy': 50.64,
        'rsi_sell': 93.77,
        'stop_loss': 0.050,
        'take_profit': 0.060,
        'satellite_rvol': 3.21,
        'satellite_stop': 0.050,
        'rvol_threshold': 1.0,
        'leverage_cap': 4.15
    },
    'XRP/USDT': {
        'description': 'Profiled (Fit 0.0 - Fallback)', 
        'rsi_buy': 25.47,
        'rsi_sell': 79.57,
        'stop_loss': 0.019,
        'take_profit': 0.056,
        'satellite_rvol': 2.51,
        'satellite_stop': 0.019,
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
        'rvol_threshold': 0.9,
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
        'leverage_cap': 4.96
    },
    'LINK/USDT': {
        'description': 'Profiled (Fit 45.1)',
        'rsi_buy': 34.43,
        'rsi_sell': 75.96,
        'stop_loss': 0.040,
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
        'satellite_stop': 0.010,
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
        'satellite_stop': 0.013,
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
        'satellite_stop': 0.022,
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

# === CONVICTION DECAY ===
CONVICTION_DECAY_BASE_HOURS = 48.0
CONVICTION_DECAY_CAPITAL_MULTIPLIER = 5.0

# === IRON BANK (Capital Preservation) ===
IRON_BANK_ENABLED = True
IRON_BANK_MIN_RESERVE = 100.0  # Floor (Lowered for buying power)
IRON_BANK_RATCHET_PCT = 0.00   # LOCKED: Profit Ratchet Disabled.
IRON_BANK_BUFFER_PCT = 0.05    # Keep 5% buffer above Floor before Ratcheting

# === MARKET TOPOLOGY (TDA) ===
TOPOLOGY_WARNING_THRESHOLD = 0.0001  # Validated Low (Normal: 0.003-0.02)
TOPOLOGY_WINDOW_SIZE = 50



# === GARBAGE COLLECTOR ===
GC_INTERVAL_CYCLES = 10
GC_STALE_ORDER_TIMEOUT = 180
GC_LOG_VERBOSE = True

# === API RESILIENCE ===
API_MAX_RETRIES = 15
API_HIBERNATION_TIME = 60
API_RETRY_JITTER_MIN = 1.0
API_RETRY_JITTER_MAX = 3.0
API_RATE_LIMIT_COOL = 10.0

# === USER CONFIG OVERRIDES (JSON) ===
# Added 2026-01-22 for Dashboard Persistence
import json
try:
    _user_cfg_path = os.path.join(os.getcwd(), 'user_config.json')
    if os.path.exists(_user_cfg_path):
        with open(_user_cfg_path, 'r') as f:
            _user_cfg = json.load(f)
            
        # Apply Overrides
        if 'max_allocation' in _user_cfg:
            GOVERNOR_MAX_MARGIN_PCT = float(_user_cfg['max_allocation'])
        if 'leverage_cap' in _user_cfg:
            PREDATOR_LEVERAGE = float(_user_cfg['leverage_cap'])
        if 'micro_mode' in _user_cfg:
            MICRO_CAPITAL_MODE = bool(_user_cfg['micro_mode'])
            
        # print(f">> [Config] Loaded User Config: Alloc={GOVERNOR_MAX_MARGIN_PCT}, Lev={PREDATOR_LEVERAGE}, Micro={MICRO_CAPITAL_MODE}")
except Exception as e:
    print(f">> [Config] Warning: Failed to load user_config.json: {e}")

# === SYSTEM RESCUE OVERRIDE (2026-01-26) ===
# User requested disabling Micro-Mode to prevent nano-transactions.
MICRO_CAPITAL_MODE = False
print(">> [Config] SYSTEM RESCUE: Micro-Mode FORCED OFF.")

def calculate_nano_position(balance: float, symbol: str, price: float) -> dict:
    """
    Calculates safety-first position size for Nano accounts.
    Strictly enforces 1.0x max leverage and minimum order sizes.
    """
    # 1. Determine Risk-Based Size (5% of Equity)
    # Reference global constant directly
    risk_alloc = balance * NANO_ALLOCATION_PCT
    
    # 2. Get Minimum Trade Size
    base_asset = symbol.split('/')[0]
    min_qty = MIN_TRADE_QTY.get(base_asset, 0.0)
    min_notional = min_qty * price
    
    # 3. Enforce Minimums
    final_notional = risk_alloc
    if risk_alloc < min_notional:
        if min_notional <= (balance * 0.10): # Allow if <= 10% of account
            final_notional = min_notional
        else:
            # Cannot afford minimum size safely
            return {
                'quantity': 0.0,
                'leverage': 1.0,
                'notional': 0.0,
                'margin': 0.0,
                'error': 'Insufficient capital for min order'
            }
            
    # 4. Calculate Quantity
    quantity = final_notional / price
    
    # 5. strict 1x Leverage
    leverage = 1.0
    margin = final_notional / leverage
    
    return {
        'quantity': quantity,
        'leverage': leverage,
        'notional': final_notional,
        'margin': margin
    }

