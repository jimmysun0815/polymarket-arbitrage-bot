"""
Configuration file for Prediction Market Arbitrage Bot
Edit these settings to customize bot behavior
"""

# ============================================================================
# SCANNING CONFIGURATION
# ============================================================================

# How often to scan for opportunities (seconds)
# 60 = 1 minute (recommended for starting)
# 30 = 30 seconds (more aggressive, higher API usage)
# 300 = 5 minutes (conservative, less frequent alerts)
SCAN_INTERVAL = 60

# Number of markets to monitor per scan
# 50 = Good balance (recommended)
# 100 = Maximum coverage
# 20 = Conservative (faster scans)
TOP_MARKETS = 50


# ============================================================================
# PROFIT THRESHOLDS
# ============================================================================

# Minimum profit threshold (in dollars)
# Research shows 2¢ ($0.02) covers Polygon gas costs
MIN_PROFIT_THRESHOLD = 0.02

# Minimum expected profit to alert (dollars)
# Set higher to reduce noise
MIN_ALERT_PROFIT = 10.0

# ROI thresholds for urgency classification
HIGH_URGENCY_ROI = 0.10    # 10%+ = HIGH urgency
MEDIUM_URGENCY_ROI = 0.05  # 5-10% = MEDIUM urgency
# <5% = LOW urgency


# ============================================================================
# WHALE TRACKING CONFIGURATION
# ============================================================================

# Minimum trade size to classify as "whale" (dollars)
# Research used $5,000 threshold
WHALE_THRESHOLD = 5000

# Minimum order flow imbalance to trigger whale alert
# 0.4 = 40% imbalance (60/40 buy/sell ratio)
# 0.6 = 60% imbalance (80/20 ratio) - more conservative
WHALE_IMBALANCE_THRESHOLD = 0.4

# Number of recent trades to analyze for whale activity
WHALE_LOOKBACK_TRADES = 50


# ============================================================================
# RISK MANAGEMENT
# ============================================================================

# Risk score thresholds (0-1 scale)
MAX_RISK_SCORE = 0.7  # Don't alert on opportunities with risk > 0.7

# Days before resolution to consider high risk
HIGH_RISK_DAYS = 2   # <2 days = oracle manipulation risk
MEDIUM_RISK_DAYS = 7  # <7 days = elevated risk

# Oracle risk premium weights
ORACLE_BASE_RISK = 0.02  # 2% base risk
SUBJECTIVE_MULTIPLIER = 3.0  # 3× multiplier for subjective markets
OBJECTIVE_MULTIPLIER = 1.0   # No multiplier for objective markets


# ============================================================================
# STRATEGY SELECTION
# ============================================================================

# Enable/disable specific strategies
ENABLE_SINGLE_CONDITION = True  # YES+NO≠$1.00
ENABLE_NEGRISK = True           # Σprices≠1.00 (most profitable)
ENABLE_WHALE_TRACKING = True    # Follow large traders
ENABLE_EVENT_DRIVEN = False     # Coming soon

# NegRisk minimum conditions
# 3 = Include markets with 3+ outcomes
# 4 = Only markets with 4+ outcomes (more capital efficient per research)
NEGRISK_MIN_CONDITIONS = 3


# ============================================================================
# API CONFIGURATION
# ============================================================================

# Polymarket CLOB endpoints (no authentication required)
POLYMARKET_BASE_URL = "https://clob.polymarket.com"
POLYMARKET_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# Request timeouts (seconds)
API_TIMEOUT = 10
WEBSOCKET_TIMEOUT = 30

# Rate limiting delays (seconds)
DELAY_BETWEEN_MARKETS = 0.2  # Delay between analyzing each market
DELAY_BETWEEN_ORDERBOOKS = 0.1  # Delay between fetching orderbooks


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Log level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL = "INFO"

# Log file name
LOG_FILE = "arbitrage_bot.log"

# Console output verbosity
VERBOSE_CONSOLE = True  # Show detailed opportunity information


# ============================================================================
# ALERT CONFIGURATION
# ============================================================================

# Minimum time between duplicate alerts for same market (seconds)
DUPLICATE_ALERT_WINDOW = 300  # 5 minutes

# Show summary after each scan
SHOW_SCAN_SUMMARY = True

# Alert filters
ALERT_ONLY_HIGH_URGENCY = False  # If True, only alert on high urgency
MIN_CAPITAL_REQUIRED = 100  # Don't alert if capital required < $100


# ============================================================================
# ADVANCED CONFIGURATION
# ============================================================================

# NegRisk capital efficiency multiplier from research
# Research showed 29× advantage over single-condition
NEGRISK_EFFICIENCY_MULTIPLIER = 29

# Orderbook depth to analyze (top N levels)
ORDERBOOK_DEPTH = 5

# Historical data window for analysis
HISTORICAL_WINDOW_SIZE = 100  # Keep last 100 snapshots per market


# ============================================================================
# SAFETY LIMITS
# ============================================================================

# Maximum capital to consider per opportunity
# This is for display purposes only - bot doesn't execute trades
MAX_POSITION_SIZE = 10000

# Maximum markets to process per scan (safety limit)
MAX_MARKETS_PER_SCAN = 100

# Scan timeout - abort if scan takes longer than this (seconds)
SCAN_TIMEOUT = 300  # 5 minutes
