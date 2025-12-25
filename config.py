"""
NEXUS Configuration (Phase 15)

Central storage for all thresholds, leverage caps, and system parameters.
"""

from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv('KRAKEN_API_KEY')
API_SECRET = os.getenv('KRAKEN_PRIVATE_KEY')

# === TELEGRAM INTEGRATION ===
TELEGRAM_ENABLED = True
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '8127012252:AAGn4mSzbhHhR2cqInKmEnabSwgPKF9LKLo')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'holonictraderbot')

SCAVENGER_THRESHOLD = 90.0  # Balance <= this = Scavenger Mode (PREDATOR @ $100)
INITIAL_CAPITAL = 100.0     # Starting simulation balance
PRINCIPAL = 80.0            # Protect $80 of the $100 (allows $20 risk buffer)

PAPER_TRADING = True        # Set to False to enable real exchange execution

# === LEVERAGE SETTINGS ===
SCAVENGER_LEVERAGE = 20      # WARP SPEED: Max Alt Leverage
PREDATOR_LEVERAGE = 50       # WARP SPEED: Max BTC Leverage

# === POSITION SIZING & RISK ===
SCAVENGER_MAX_MARGIN = 8.0   # WARP SPEED: Risk 80% of Equity
SCAVENGER_STOP_LOSS = 0.03   # 3% wiggle room
SCAVENGER_SCALP_TP = 0.01    # 1% Quick Scalp target

PREDATOR_STOP_LOSS = 0.05    # 5% wider stop for trends
PREDATOR_TAKE_PROFIT = 0.03  # 3% target (Reduced from 5% for tighter exits)

# === TIME GATING (Kill Zones - UTC) ===
KILL_ZONES = [] # WARP SPEED: Trade 24/7

# === VOLATILITY ===
ATR_PERIOD = 14
ATR_STORM_MULTIPLIER = 3.0   # If current ATR > 3x avg, it's a storm

# === BOLLINGER BANDS ===
BB_PERIOD = 20
BB_STD = 2

# === PREDATOR TRAILING STOP ===
PREDATOR_TRAILING_STOP_ATR_MULT = 2.0

# === ASSET CONSTRAINTS ===
ALLOWED_ASSETS = [
    'ADA/USDT', 'BNB/USDT', 'BTC/USDT', 'DOGE/USDT', 'ETH/USDT', 'SOL/USDT', 'SUI/USDT', 'XRP/USDT',
    'SHIB/USDT', 'PAXG/USDT', 'LTC/USDT', 'LINK/USDT', 'XMR/USDT', 'ALGO/USDT', 'UNI/USDT', 'AAVE/USDT'
]
FORBIDDEN_ASSETS = []

# === STRATEGY SETTINGS (Centralized) ===
STRATEGY_RSI_OVERSOLD = 45
STRATEGY_RSI_OVERBOUGHT = 70
STRATEGY_RSI_PANIC_BUY = 40.0
STRATEGY_RSI_ENTRY_MAX = 65.0       # RECALIBRATION: Increased from 60
STRATEGY_LSTM_THRESHOLD = 0.51      # RECALIBRATION: Lowered from 0.52
STRATEGY_XGB_THRESHOLD = 0.48    # Confidence required for XGBoost (Ensemble)
STRATEGY_POST_EXIT_COOLDOWN_CANDLES = 3 # Wait 3 candles before re-entry

# === GOVERNANCE / RISK ===
GOVERNOR_COOLDOWN_SECONDS = 60
GOVERNOR_MIN_STACK_DIST = 0.002     # RECALIBRATION: Lowered from 0.005
GOVERNOR_MAX_MARGIN_PCT = 0.20      # RECALIBRATION: Increased from 0.10
GOVERNOR_STACK_DECAY = 0.8 # Reduce size by 20% each stack
GOVERNOR_MAX_TREND_AGE_HOURS = 24.0 # Trends > 24h are exhausted
GOVERNOR_TREND_DECAY_START = 12.0 # Start reducing risk after 12h

# === PHASE 12: INSTITUTIONAL RISK MANAGEMENT ===
# Minimax Constraint (Game Theory)
# PRINCIPAL is now defined above to avoid duplication.
MAX_RISK_PCT = 0.02  # RECALIBRATION: Increased from 0.01

# Modified Kelly Criterion (Half-Kelly)
KELLY_RISK_REWARD = 2.0  # Adjusted risk/reward for win-rate calculation
KELLY_LOOKBACK = 50  # Number of trades to calculate win rate
KELLY_MIN_FRACTION = 0.05  # Minimum Kelly fraction (floor)
KELLY_MAX_FRACTION = 0.25  # Maximum Kelly fraction (ceiling)

# Volatility Scalar (Inverse Variance Weighting)
VOL_SCALAR_PERIOD = 14  # ATR reference period
VOL_SCALAR_MIN = 0.5  # Minimum scalar (max position reduction)
VOL_SCALAR_MAX = 2.0  # Maximum scalar (max position increase)

# === CROSS-ASSET CORRELATION ===
GMB_THRESHOLD = 0.35  # RECALIBRATION: Lowered from 0.5

# === PHASE 22: PPO SOVEREIGN BRAIN ===
PPO_LEARNING_RATE = 0.0003
PPO_CLIP_RATIO = 0.2
PPO_REWARD_DRAWDOWN_PENALTY = 2.0 # Penalty multiplier for drawdown

# === PHASE 31: CONCURRENCY & RATE LIMITING ===
CCXT_RATE_LIMIT = True
CCXT_POOL_SIZE = 20         # Matches/exceeds TRADER_MAX_WORKERS
TRADER_MAX_WORKERS = 16    # Parallel analysis threads

# === PHASE 33: INTEL GPU ACCELERATION ===
USE_INTEL_GPU = True
USE_INTEL_GPU = True
USE_OPENVINO = True

# === PHASE 25: SATELLITE COMMANDER (TIER 2) ===
SATELLITE_ASSETS = ['XRP/USDT', 'DOGE/USDT', 'ADA/USDT', 'LINK/USDT']
SATELLITE_MARGIN = 10.0      # Fixed $10 Margin per trade
SATELLITE_LEVERAGE = 10.0    # 10x Fixed Leverage (Position = $100)
SATELLITE_RVOL_THRESHOLD = 1.5   # 1.5x Volume required
SATELLITE_DOGE_RVOL_THRESHOLD = 2.0 # 2.0x for DOGE specifically
SATELLITE_BBW_EXPANSION_THRESHOLD = 0.20 # 20% Expansion required
SATELLITE_BREAKEVEN_TRIGGER = 0.015  # +1.5% Move -> Move SL to BE
SATELLITE_TAKE_PROFIT_1 = 0.03       # +3.0% Move -> Close 50%

# === PHASE 35: IMMUNE SYSTEM & PERSONALITY ===
# Asset Families (Cluster Risk)
FAMILY_L1 = ['ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'AVAX/USDT']
FAMILY_PAYMENT = ['XRP/USDT', 'LTC/USDT', 'BCH/USDT']
FAMILY_MEME = ['DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT']

# Health Thresholds
IMMUNE_MAX_DAILY_DRAWDOWN = 0.05     # 5% Daily Loss Limit
IMMUNE_MAX_LEVERAGE_RATIO = 10.0     # Max 10x Total Account Leverage

# Personality Parameters
PERSONALITY_BTC_ATR_FILTER = 0.5     # Ignore if ATR < 50% of 30d Avg
PERSONALITY_SOL_RSI_LONG = 55.0      # Min RSI to Long SOL
PERSONALITY_SOL_RSI_SHORT = 45.0     # Max RSI to Short SOL
PERSONALITY_DOGE_RVOL = 2.0          # Higher RVOL for DOGE



