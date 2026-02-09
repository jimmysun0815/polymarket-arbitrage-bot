"""
Configuration file for Prediction Market Arbitrage Bot
Edit these settings to customize bot behavior
"""

# ============================================================================
# SCANNING CONFIGURATION
# ============================================================================

# Maximum number of markets to scan ('full' for all active markets, or integer)
SCAN_RANGE = 'full'

# How often the FRONTEND runs detection (seconds)
SCAN_INTERVAL = 30

# Number of markets to process per batch
BATCH_SIZE = 200

# Background scanner full refresh interval (seconds)
BACKGROUND_SCAN_INTERVAL = 300  # 5 minutes

# Rate limit: sleep seconds on 429 response
RATE_LIMIT_RETRY_SLEEP = 10

# Rate limit: max retries on 429 response
RATE_LIMIT_MAX_RETRIES = 3

# Events API sort field for Step B (id desc = newest events first)
EVENTS_ORDER_FIELD = "id"

# Max events to fetch per scan (each event contains multiple markets)
EVENTS_FETCH_LIMIT = 500

# Number of NegRisk events to fetch (priority scan)
NEGRISK_LIMIT = 50

# SQLite database file for market data
DB_FILE = "market_db.sqlite"


# ============================================================================
# PROFIT THRESHOLDS & DETECTION
# ============================================================================

# Mid price pre-screening thresholds (skip orderbook if sum in [0.99, 1.01])
# NOTE: Only effective for NegRisk (3+ tokens). Binary markets always sum to 1.0.
MID_SUM_THRESHOLD = 0.99   # sum < 0.99 -> buy candidate
MID_SUM_UPPER = 1.01       # sum > 1.01 -> sell candidate

# Binary markets (2 tokens) always have mid_sum=1.0, cannot be mid-screened.
# Instead, check top N by volume for orderbook spread arbitrage.
BINARY_TOP_N_BY_VOLUME = 100

# NegRisk: CLOB GET /markets/{condition_id} returns token prices that are NORMALIZED
# (sum ≈ 1.0), so mid_price_sum rarely passes [<0.99 or >1.01]. When no NegRisk pass
# mid-screen, we still pull orderbook for top N NegRisk by volume (fallback).
NEGRISK_TOP_N_BY_VOLUME = 50

# Minimum profit threshold (in dollars)
MIN_PROFIT_THRESHOLD = 0.005  # $0.005

# Maximum profit threshold (filter stale/broken markets)
MAX_PROFIT_THRESHOLD = 0.50  # $0.50

# Minimum liquidity per outcome (dollars)
MIN_LIQUIDITY = 100  # $100

# NegRisk capital efficiency multiplier (from IMDEA research)
CAPITAL_EFFICIENCY_MULTIPLIER = 29

# Polymarket trading fee
POLYMARKET_FEE_PCT = 0.02  # 2%

# Gas cost buffer
GAS_BUFFER_PCT = 0.001  # 0.1%

# Minimum expected profit to alert (dollars)
MIN_ALERT_PROFIT = 10.0

# ROI thresholds for urgency classification
HIGH_URGENCY_ROI = 0.10    # 10%+ = HIGH urgency
MEDIUM_URGENCY_ROI = 0.025  # 2.5%+ = MEDIUM urgency


# ============================================================================
# WHALE TRACKING CONFIGURATION
# ============================================================================

# Minimum trade size to classify as "whale" (dollars)
WHALE_THRESHOLD = 5000

# Minimum order flow imbalance to trigger whale alert
WHALE_IMBALANCE_THRESHOLD = 0.4

# Number of recent trades to analyze for whale activity
WHALE_LOOKBACK_TRADES = 50


# ============================================================================
# RISK MANAGEMENT
# ============================================================================

# Risk score thresholds (0-1 scale)
MAX_RISK_SCORE = 0.7

# Days before resolution to consider high risk
HIGH_RISK_DAYS = 2
MEDIUM_RISK_DAYS = 7

# Oracle risk premium weights
ORACLE_BASE_RISK = 0.02
SUBJECTIVE_MULTIPLIER = 3.0
OBJECTIVE_MULTIPLIER = 1.0


# ============================================================================
# STRATEGY SELECTION
# ============================================================================

ENABLE_SINGLE_CONDITION = True
ENABLE_NEGRISK = True
ENABLE_WHALE_TRACKING = True
ENABLE_EVENT_DRIVEN = False

# NegRisk minimum conditions (outcomes)
NEGRISK_MIN_CONDITIONS = 3


# ============================================================================
# API CONFIGURATION
# ============================================================================

POLYMARKET_CLOB_URL = "https://clob.polymarket.com"
POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com"
# Market channel: must use /ws/market (connecting to /ws/ returns 404)
POLYMARKET_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# Request timeouts (seconds)
API_TIMEOUT = 30

# Rate limiting delays (seconds)
DELAY_BETWEEN_MARKETS = 0.05
DELAY_BETWEEN_ORDERBOOKS = 0.05
DELAY_BETWEEN_BATCHES = 0.5


# ============================================================================
# NOTIFICATION CONFIGURATION
# ============================================================================

# Telegram bot settings (leave empty to disable)
TELEGRAM_TOKEN = ""
TELEGRAM_CHAT_ID = ""

# Discord webhook URL (leave empty to disable)
DISCORD_WEBHOOK_URL = ""

# Active notification methods: 'console', 'telegram', 'discord'
NOTIFICATION_METHODS = ['console', 'telegram']

# Rate limit: same opportunity alert cooldown (seconds)
ALERT_RATE_LIMIT_SECONDS = 60


# ============================================================================
# WEBSOCKET CONFIGURATION
# ============================================================================

# Enable WebSocket real-time price updates
WS_ENABLED = True

# Maximum number of token subscriptions
WS_SUBS_LIMIT = 200

# Minimum volume for WS watchlist inclusion ($)
MIN_WATCHLIST_VOLUME = 1000

# Reconnect delay after disconnect (seconds)
WS_RECONNECT_DELAY = 5

# Maximum reconnect attempts before fallback to polling
WS_MAX_RECONNECT_ATTEMPTS = 3

# Polling interval when WebSocket is down (seconds)
WS_FALLBACK_INTERVAL = 30

# Cooldown: same market WS mid-screen pass log + detection at most once per N seconds
WS_MID_SCREEN_COOLDOWN_SECONDS = 60


# ============================================================================
# ORDERBOOK / SLIPPAGE CONFIGURATION
# ============================================================================

# Maximum allowed slippage percentage
SLIPPAGE_TOLERANCE = 0.02  # 2%

# Orderbook cache TTL (seconds)
ORDERBOOK_CACHE_TTL = 15

# Default trade size for slippage estimation (dollars)
DEFAULT_TRADE_SIZE = 100

# Orderbook depth to analyze (top N levels)
ORDERBOOK_DEPTH = 5


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = "INFO"
LOG_FILE = "arbitrage_bot.log"
VERBOSE_CONSOLE = True


# ============================================================================
# ALERT CONFIGURATION
# ============================================================================

# Minimum time between duplicate alerts for same market (seconds)
DUPLICATE_ALERT_WINDOW = 300

SHOW_SCAN_SUMMARY = True
ALERT_ONLY_HIGH_URGENCY = False
MIN_CAPITAL_REQUIRED = 100


# ============================================================================
# SAFETY LIMITS
# ============================================================================

MAX_POSITION_SIZE = 10000
MAX_MARKETS_PER_SCAN = 5000
SCAN_TIMEOUT = 300

# DB cleanup: delete inactive markets older than this (seconds)
DB_CLEANUP_AGE = 7 * 86400  # 7 days
