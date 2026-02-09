#!/usr/bin/env python3

"""
Prediction Market Arbitrage Bot - Enhanced Full Market Scanner

Based on IMDEA Networks research: $39.59M arbitrage extraction (Apr 2024-Apr 2025)

Strategies Implemented:
1. Single-Condition Arbitrage (YES + NO != $1.00) - $10.58M extracted
2. NegRisk Rebalancing (sum(prices) != 1.00) - $28.99M extracted (29x capital efficiency)

Enhancements:
- 3-step scan: NegRisk events first -> hot markets by volume -> merge & dedup
- Best ask price detection with fee/gas deduction
- Complete set cost calculation with merge advice
- Orderbook slippage estimation
- Telegram + Discord notifications
- WebSocket real-time price updates
- JSON structured logging
"""

import asyncio
import aiohttp
import json
import time
import sqlite3
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import logging

from config import (
    # Scanning
    SCAN_RANGE, SCAN_INTERVAL, BATCH_SIZE,
    RATE_LIMIT_RETRY_SLEEP, RATE_LIMIT_MAX_RETRIES,
    NEGRISK_LIMIT,
    # Detection
    MIN_PROFIT_THRESHOLD, MAX_PROFIT_THRESHOLD, MIN_LIQUIDITY,
    CAPITAL_EFFICIENCY_MULTIPLIER, POLYMARKET_FEE_PCT, GAS_BUFFER_PCT,
    HIGH_URGENCY_ROI, MEDIUM_URGENCY_ROI,
    # Mid price screening
    MID_SUM_THRESHOLD, MID_SUM_UPPER, BINARY_TOP_N_BY_VOLUME, NEGRISK_TOP_N_BY_VOLUME,
    NEGRISK_MIN_CONDITIONS,
    # API
    POLYMARKET_CLOB_URL, POLYMARKET_GAMMA_URL, API_TIMEOUT,
    DELAY_BETWEEN_MARKETS, DELAY_BETWEEN_ORDERBOOKS, DELAY_BETWEEN_BATCHES,
    # Notifications
    NOTIFICATION_METHODS, ALERT_RATE_LIMIT_SECONDS,
    # WebSocket
    WS_ENABLED, WS_SUBS_LIMIT, WS_FALLBACK_INTERVAL,
    # Slippage
    SLIPPAGE_TOLERANCE, DEFAULT_TRADE_SIZE, ORDERBOOK_DEPTH,
    # DB & Background
    DB_FILE, BACKGROUND_SCAN_INTERVAL,
    # Logging
    LOG_LEVEL, LOG_FILE,
)

import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# JSON LOGGING
# ============================================================================

class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging"""
    def format(self, record):
        log_entry = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage()
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


json_formatter = JsonFormatter()

file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(json_formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ArbitrageOpportunity:
    """Represents a detected arbitrage opportunity"""
    market_id: str
    market_name: str
    opportunity_type: str  # 'single_condition', 'negrisk'
    expected_profit: float
    roi: float
    capital_required: float
    risk_score: float
    urgency: str  # 'high', 'medium', 'low'
    details: Dict
    timestamp: datetime
    # Enhanced fields
    net_profit_after_fees: float = 0.0
    merge_advice: str = ""
    estimated_slippage: float = 0.0
    net_profit_after_slippage: float = 0.0


# ============================================================================
# POLYMARKET CLIENT - ORDERBOOK ONLY (scanning offloaded to scanner_process)
# ============================================================================

class PolymarketClient:
    """Lightweight Polymarket API client for orderbook fetching only.
    Market scanning is handled by the background scanner_process."""

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.diagnostics = {
            'orderbooks_fetched': 0,
            'orderbooks_with_data': 0,
            'markets_with_tokens': 0,
            'markets_analyzed': 0,
        }

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=API_TIMEOUT)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_orderbook(self, token_id: str) -> Optional[Dict]:
        """Get orderbook for a specific outcome token"""
        if not token_id:
            return None
        try:
            url = f"{POLYMARKET_CLOB_URL}/book"
            params = {'token_id': token_id}
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; ArbitrageBot/2.0)'}

            async with self.session.get(url, params=params, headers=headers) as resp:
                self.diagnostics['orderbooks_fetched'] += 1
                if resp.status == 200:
                    book = await resp.json()
                    if book and (book.get('asks') or book.get('bids')):
                        self.diagnostics['orderbooks_with_data'] += 1
                    return book
                return None
        except Exception as e:
            logger.debug(f"Orderbook fetch error for {token_id}: {e}")
            return None


# ============================================================================
# ORDERBOOK HELPERS
# ============================================================================

def _get_best_ask(asks) -> float:
    """Get the lowest ask price from an asks list."""
    if not asks:
        return 0.0
    try:
        if isinstance(asks[0], dict):
            return float(asks[0].get('price', 0))
        elif isinstance(asks[0], (list, tuple)):
            return float(asks[0][0])
        else:
            return float(asks[0])
    except (ValueError, TypeError, IndexError):
        return 0.0


def _get_best_bid(bids) -> float:
    """Get the highest bid price from a bids list."""
    if not bids:
        return 0.0
    try:
        # bids may be sorted ascending or descending; take max as best bid
        bid_prices = []
        for b in bids:
            if isinstance(b, dict):
                bid_prices.append(float(b.get('price', 0)))
            elif isinstance(b, (list, tuple)):
                bid_prices.append(float(b[0]))
            else:
                bid_prices.append(float(b))
        return max(bid_prices) if bid_prices else 0.0
    except (ValueError, TypeError, IndexError):
        return 0.0


def _parse_tokens(market: Dict) -> list:
    """Parse tokens list from a market dict, handling JSON strings and clob_token_ids fallback."""
    tokens = market.get('tokens') or []
    if isinstance(tokens, str):
        try:
            tokens = json.loads(tokens)
        except Exception:
            tokens = []

    if not tokens:
        clob_token_ids = market.get('clob_token_ids')
        outcomes_str = market.get('outcomes') or market.get('options') or []

        if clob_token_ids and outcomes_str:
            if isinstance(outcomes_str, str):
                try:
                    outcomes_list = json.loads(outcomes_str)
                except Exception:
                    outcomes_list = []
            else:
                outcomes_list = outcomes_str

            if isinstance(clob_token_ids, str):
                try:
                    token_ids_list = json.loads(clob_token_ids)
                except Exception:
                    token_ids_list = []
            else:
                token_ids_list = clob_token_ids if isinstance(clob_token_ids, list) else []

            if len(token_ids_list) == len(outcomes_list):
                tokens = [
                    {'outcome': out, 'token_id': tid}
                    for out, tid in zip(outcomes_list, token_ids_list)
                ]

    return tokens if isinstance(tokens, list) else []


# ============================================================================
# ARBITRAGE DETECTOR - ENHANCED
# ============================================================================

class ArbitrageDetector:
    """Enhanced arbitrage detection with best-ask, fee deduction, slippage"""

    def __init__(self):
        self.opportunities: List[ArbitrageOpportunity] = []
        self.diagnostics = {
            'single_condition_checked': 0,
            'single_condition_found': 0,
            'negrisk_checked': 0,
            'negrisk_found': 0,
            'no_orderbook': 0,
            'no_prices': 0,
            'filtered_low_liquidity': 0,
            'filtered_high_slippage': 0,
        }

    # ------------------------------------------------------------------
    # Buy/Sell arb cost calculations
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_buy_arb(ask_prices: List[float]) -> Dict:
        """
        Buy arb: buy all outcomes at best ask, merge for $1.
        Profit = 1.0 - sum(asks) - fees - gas.
        """
        ask_sum = sum(ask_prices)
        fees = ask_sum * POLYMARKET_FEE_PCT
        gas = ask_sum * GAS_BUFFER_PCT
        total_cost = ask_sum + fees + gas
        net_profit = 1.0 - total_cost
        roi = (net_profit / total_cost * 100) if total_cost > 0 else 0
        return {
            'ask_sum': ask_sum,
            'fees': fees,
            'gas': gas,
            'total_cost': total_cost,
            'capital_required': total_cost,
            'net_profit': net_profit,
            'roi_pct': roi,
        }

    @staticmethod
    def calculate_sell_arb(bid_prices: List[float]) -> Dict:
        """
        Sell arb: mint complete set for $1, sell all at best bid.
        Profit = sum(bids) - 1.0 - fees - gas.
        """
        bid_sum = sum(bid_prices)
        fees = bid_sum * POLYMARKET_FEE_PCT
        gas = bid_sum * GAS_BUFFER_PCT
        total_income = bid_sum - fees - gas
        net_profit = total_income - 1.0
        roi = (net_profit / 1.0 * 100) if net_profit > 0 else 0
        return {
            'bid_sum': bid_sum,
            'fees': fees,
            'gas': gas,
            'total_income': total_income,
            'capital_required': 1.0,  # mint cost
            'net_profit': net_profit,
            'roi_pct': roi,
        }

    # ------------------------------------------------------------------
    # Single-Condition (Binary) Arbitrage
    # Buy arb: YES_best_ask + NO_best_ask < 1.0 -> buy both + merge
    # Sell arb: YES_best_bid + NO_best_bid > 1.0 -> mint + sell both
    # ------------------------------------------------------------------
    def detect_single_condition_arbitrage(
        self,
        market: Dict,
        yes_orderbook: Optional[Dict],
        no_orderbook: Optional[Dict],
        slippage_estimator=None,
    ) -> Optional[ArbitrageOpportunity]:
        """
        Binary market arbitrage (YES + NO != $1.00).
        - Buy arb: ask_sum < 1.0 -> buy both sides + merge for $1 profit
        - Sell arb: bid_sum > 1.0 -> mint for $1 + sell both sides for profit
        """
        self.diagnostics['single_condition_checked'] += 1

        if not yes_orderbook or not no_orderbook:
            self.diagnostics['no_orderbook'] += 1
            return None

        try:
            if market.get('closed') or market.get('archived') or not market.get('active'):
                return None

            yes_asks = yes_orderbook.get('asks', [])
            no_asks = no_orderbook.get('asks', [])
            yes_bids = yes_orderbook.get('bids', [])
            no_bids = no_orderbook.get('bids', [])

            yes_best_ask = _get_best_ask(yes_asks)
            no_best_ask = _get_best_ask(no_asks)
            yes_best_bid = _get_best_bid(yes_bids)
            no_best_bid = _get_best_bid(no_bids)

            logger.debug(
                f"Binary arb check: YES ask={yes_best_ask:.4f} bid={yes_best_bid:.4f}, "
                f"NO ask={no_best_ask:.4f} bid={no_best_bid:.4f}")

            market_name = (market.get('question') or market.get('title') or
                            market.get('description') or 'Unknown Market')[:80]
            market_id = (market.get('condition_id') or market.get('id') or
                          market.get('market_id') or 'unknown')
            market_slug = market.get('market_slug') or market.get('slug', '')

            # --- Buy arb: ask_sum < 1.0 ---
            if yes_best_ask > 0 and no_best_ask > 0:
                calc = self.calculate_buy_arb([yes_best_ask, no_best_ask])
                net_profit_per_unit = calc['net_profit']

                if (net_profit_per_unit > MIN_PROFIT_THRESHOLD
                        and net_profit_per_unit < MAX_PROFIT_THRESHOLD):
                    # Liquidity check
                    yes_liq = sum(float(a.get('size', 0)) for a in yes_asks[:ORDERBOOK_DEPTH])
                    no_liq = sum(float(a.get('size', 0)) for a in no_asks[:ORDERBOOK_DEPTH])
                    min_liq = min(yes_liq, no_liq)

                    if min_liq >= MIN_LIQUIDITY:
                        expected_profit = net_profit_per_unit * min_liq
                        capital_required = calc['capital_required'] * min_liq
                        roi = calc['roi_pct'] / 100.0

                        # Slippage check
                        est_slippage = 0.0
                        net_after_slippage = expected_profit
                        if slippage_estimator:
                            slip_yes = slippage_estimator.estimate_slippage_from_book(
                                yes_asks, DEFAULT_TRADE_SIZE)
                            slip_no = slippage_estimator.estimate_slippage_from_book(
                                no_asks, DEFAULT_TRADE_SIZE)
                            est_slippage = max(
                                slip_yes.get('slippage_pct', 0),
                                slip_no.get('slippage_pct', 0))
                            if est_slippage > SLIPPAGE_TOLERANCE:
                                self.diagnostics['filtered_high_slippage'] += 1
                            else:
                                slippage_cost = est_slippage * capital_required
                                net_after_slippage = expected_profit - slippage_cost
                                if net_after_slippage > 0:
                                    self.diagnostics['single_condition_found'] += 1
                                    return ArbitrageOpportunity(
                                        market_id=str(market_id),
                                        market_name=market_name,
                                        opportunity_type='BUY_ARB',
                                        expected_profit=expected_profit,
                                        roi=roi,
                                        capital_required=capital_required,
                                        risk_score=self._calculate_risk_score(
                                            market, 'single_condition'),
                                        urgency=('high' if roi > HIGH_URGENCY_ROI
                                                  else 'medium' if roi > MEDIUM_URGENCY_ROI
                                                  else 'low'),
                                        details={
                                            'strategy': 'binary',
                                            'ask_sum': calc['ask_sum'],
                                            'yes_ask': yes_best_ask,
                                            'no_ask': no_best_ask,
                                            'fees': calc['fees'],
                                            'liquidity': min_liq,
                                            'market_slug': market_slug,
                                        },
                                        timestamp=datetime.now(),
                                        net_profit_after_fees=expected_profit,
                                        merge_advice="Buy YES + NO shares, merge for $1 via CTF contract",
                                        estimated_slippage=est_slippage,
                                        net_profit_after_slippage=net_after_slippage,
                                    )
                        else:
                            # No slippage estimator
                            self.diagnostics['single_condition_found'] += 1
                            return ArbitrageOpportunity(
                                market_id=str(market_id),
                                market_name=market_name,
                                opportunity_type='BUY_ARB',
                                expected_profit=expected_profit,
                                roi=roi,
                                capital_required=capital_required,
                                risk_score=self._calculate_risk_score(
                                    market, 'single_condition'),
                                urgency=('high' if roi > HIGH_URGENCY_ROI
                                          else 'medium' if roi > MEDIUM_URGENCY_ROI
                                          else 'low'),
                                details={
                                    'strategy': 'binary',
                                    'ask_sum': calc['ask_sum'],
                                    'yes_ask': yes_best_ask,
                                    'no_ask': no_best_ask,
                                    'fees': calc['fees'],
                                    'liquidity': min_liq,
                                    'market_slug': market_slug,
                                },
                                timestamp=datetime.now(),
                                net_profit_after_fees=expected_profit,
                                merge_advice="Buy YES + NO shares, merge for $1 via CTF contract",
                                estimated_slippage=0.0,
                                net_profit_after_slippage=expected_profit,
                            )
                    else:
                        self.diagnostics['filtered_low_liquidity'] += 1

            # --- Sell arb: bid_sum > 1.0 ---
            if yes_best_bid > 0 and no_best_bid > 0:
                calc = self.calculate_sell_arb([yes_best_bid, no_best_bid])
                net_profit_per_unit = calc['net_profit']

                if (net_profit_per_unit > MIN_PROFIT_THRESHOLD
                        and net_profit_per_unit < MAX_PROFIT_THRESHOLD):
                    # Liquidity check (use bids for sell)
                    yes_liq = sum(float(b.get('size', 0)) for b in yes_bids[:ORDERBOOK_DEPTH])
                    no_liq = sum(float(b.get('size', 0)) for b in no_bids[:ORDERBOOK_DEPTH])
                    min_liq = min(yes_liq, no_liq)

                    if min_liq >= MIN_LIQUIDITY:
                        expected_profit = net_profit_per_unit * min_liq
                        capital_required = calc['capital_required'] * min_liq
                        roi = calc['roi_pct'] / 100.0

                        # Slippage check
                        est_slippage = 0.0
                        net_after_slippage = expected_profit
                        if slippage_estimator:
                            slip_yes = slippage_estimator.estimate_slippage_from_book(
                                yes_bids, DEFAULT_TRADE_SIZE)
                            slip_no = slippage_estimator.estimate_slippage_from_book(
                                no_bids, DEFAULT_TRADE_SIZE)
                            est_slippage = max(
                                slip_yes.get('slippage_pct', 0),
                                slip_no.get('slippage_pct', 0))
                            if est_slippage > SLIPPAGE_TOLERANCE:
                                self.diagnostics['filtered_high_slippage'] += 1
                            else:
                                slippage_cost = est_slippage * capital_required
                                net_after_slippage = expected_profit - slippage_cost
                                if net_after_slippage > 0:
                                    self.diagnostics['single_condition_found'] += 1
                                    return ArbitrageOpportunity(
                                        market_id=str(market_id),
                                        market_name=market_name,
                                        opportunity_type='SELL_ARB',
                                        expected_profit=expected_profit,
                                        roi=roi,
                                        capital_required=capital_required,
                                        risk_score=self._calculate_risk_score(
                                            market, 'single_condition'),
                                        urgency=('high' if roi > HIGH_URGENCY_ROI
                                                  else 'medium' if roi > MEDIUM_URGENCY_ROI
                                                  else 'low'),
                                        details={
                                            'strategy': 'binary',
                                            'bid_sum': calc['bid_sum'],
                                            'yes_bid': yes_best_bid,
                                            'no_bid': no_best_bid,
                                            'fees': calc['fees'],
                                            'liquidity': min_liq,
                                            'market_slug': market_slug,
                                        },
                                        timestamp=datetime.now(),
                                        net_profit_after_fees=expected_profit,
                                        merge_advice="Mint complete set for $1, sell YES + NO shares",
                                        estimated_slippage=est_slippage,
                                        net_profit_after_slippage=net_after_slippage,
                                    )
                        else:
                            # No slippage estimator
                            self.diagnostics['single_condition_found'] += 1
                            return ArbitrageOpportunity(
                                market_id=str(market_id),
                                market_name=market_name,
                                opportunity_type='SELL_ARB',
                                expected_profit=expected_profit,
                                roi=roi,
                                capital_required=capital_required,
                                risk_score=self._calculate_risk_score(
                                    market, 'single_condition'),
                                urgency=('high' if roi > HIGH_URGENCY_ROI
                                          else 'medium' if roi > MEDIUM_URGENCY_ROI
                                          else 'low'),
                                details={
                                    'strategy': 'binary',
                                    'bid_sum': calc['bid_sum'],
                                    'yes_bid': yes_best_bid,
                                    'no_bid': no_best_bid,
                                    'fees': calc['fees'],
                                    'liquidity': min_liq,
                                    'market_slug': market_slug,
                                },
                                timestamp=datetime.now(),
                                net_profit_after_fees=expected_profit,
                                merge_advice="Mint complete set for $1, sell YES + NO shares",
                                estimated_slippage=0.0,
                                net_profit_after_slippage=expected_profit,
                            )
                    else:
                        self.diagnostics['filtered_low_liquidity'] += 1

            return None

        except Exception as e:
            logger.debug(f"Error in single-condition detection: {e}")
            return None

    # ------------------------------------------------------------------
    # NegRisk Event-Level Arbitrage
    # Buy arb: sum(YES best asks) < 1.0 -> buy all YES + merge
    # Sell arb: sum(YES best bids) > 1.0 -> mint + sell all YES
    # ------------------------------------------------------------------
    def detect_negrisk_arbitrage(
        self,
        event_id: str,
        event_title: str,
        conditions_data: List,
        slippage_estimator=None,
    ) -> Optional[ArbitrageOpportunity]:
        """
        Event-level NegRisk arbitrage detection.

        Args:
            event_id: The event identifier
            event_title: Human-readable event title
            conditions_data: list of (condition_dict, yes_token_id, yes_orderbook) tuples
            slippage_estimator: optional slippage estimator

        Buy arb: sum of all YES best asks across conditions < 1.0
        Sell arb: sum of all YES best bids across conditions > 1.0
        """
        self.diagnostics['negrisk_checked'] += 1

        if len(conditions_data) < NEGRISK_MIN_CONDITIONS:
            return None

        try:
            ask_prices = []
            bid_prices = []
            liquidities_ask = []
            liquidities_bid = []
            all_asks_lists = []
            all_bids_lists = []
            condition_names = []

            for condition, yes_tid, book in conditions_data:
                if not book:
                    continue

                asks = book.get('asks', [])
                bids = book.get('bids', [])

                best_ask = _get_best_ask(asks)
                best_bid = _get_best_bid(bids)

                cond_name = (condition.get('question') or condition.get('title') or '')
                condition_names.append(cond_name[:40])

                if best_ask > 0:
                    ask_prices.append(best_ask)
                    all_asks_lists.append(asks)
                    ask_liq = sum(float(a.get('size', 0)) for a in asks[:ORDERBOOK_DEPTH])
                    liquidities_ask.append(ask_liq)

                if best_bid > 0:
                    bid_prices.append(best_bid)
                    all_bids_lists.append(bids)
                    bid_liq = sum(float(b.get('size', 0)) for b in bids[:ORDERBOOK_DEPTH])
                    liquidities_bid.append(bid_liq)

            ask_sum = sum(ask_prices) if ask_prices else 0
            bid_sum = sum(bid_prices) if bid_prices else 0

            logger.debug(
                f"Event {event_id}: YES ask prices={[f'{p:.4f}' for p in ask_prices]} "
                f"ask_sum={ask_sum:.4f}, bid_sum={bid_sum:.4f}")

            # Get a slug for URL (use first condition's slug or event slug)
            first_cond = conditions_data[0][0] if conditions_data else {}
            market_slug = (first_cond.get('_event_slug') or
                           first_cond.get('market_slug') or
                           first_cond.get('slug', ''))

            # --- Buy arb: ask_sum < 1.0 ---
            if (ask_prices and len(ask_prices) >= NEGRISK_MIN_CONDITIONS
                    and ask_sum < 1.0):
                calc = self.calculate_buy_arb(ask_prices)
                net_profit_per_unit = calc['net_profit']

                if (net_profit_per_unit > MIN_PROFIT_THRESHOLD
                        and net_profit_per_unit < MAX_PROFIT_THRESHOLD):
                    min_liq = min(liquidities_ask) if liquidities_ask else 0
                    if min_liq < MIN_LIQUIDITY:
                        self.diagnostics['filtered_low_liquidity'] += 1
                    else:
                        expected_profit = net_profit_per_unit * min_liq
                        capital_required = calc['capital_required'] * min_liq
                        roi = calc['roi_pct'] / 100.0

                        # Slippage
                        est_slippage = 0.0
                        net_after_slippage = expected_profit
                        if slippage_estimator:
                            max_slip = max(
                                slippage_estimator.estimate_slippage_from_book(
                                    al, DEFAULT_TRADE_SIZE).get('slippage_pct', 0)
                                for al in all_asks_lists
                            ) if all_asks_lists else 0
                            est_slippage = max_slip
                            if est_slippage > SLIPPAGE_TOLERANCE:
                                self.diagnostics['filtered_high_slippage'] += 1
                                return None
                            slippage_cost = est_slippage * capital_required
                            net_after_slippage = expected_profit - slippage_cost
                            if net_after_slippage <= 0:
                                return None

                        self.diagnostics['negrisk_found'] += 1
                        return ArbitrageOpportunity(
                            market_id=str(event_id),
                            market_name=event_title[:80] or 'Unknown Event',
                            opportunity_type='BUY_ARB',
                            expected_profit=expected_profit,
                            roi=roi,
                            capital_required=capital_required,
                            risk_score=self._calculate_risk_score(
                                first_cond, 'negrisk'),
                            urgency=('high' if roi > HIGH_URGENCY_ROI
                                      else 'medium' if roi > MEDIUM_URGENCY_ROI
                                      else 'low'),
                            details={
                                'strategy': 'negrisk',
                                'num_conditions': len(conditions_data),
                                'ask_sum': calc['ask_sum'],
                                'yes_ask_prices': [f"{p:.4f}" for p in ask_prices],
                                'fees': calc['fees'],
                                'min_liquidity': min_liq,
                                'capital_efficiency': f'{CAPITAL_EFFICIENCY_MULTIPLIER}x',
                                'event_title': event_title,
                                'market_slug': market_slug,
                            },
                            timestamp=datetime.now(),
                            net_profit_after_fees=expected_profit,
                            merge_advice=(
                                "Buy full set of YES shares and merge "
                                "via CTF/NegRisk contract for instant profit"),
                            estimated_slippage=est_slippage,
                            net_profit_after_slippage=net_after_slippage,
                        )

            # --- Sell arb: bid_sum > 1.0 ---
            if (bid_prices and len(bid_prices) >= NEGRISK_MIN_CONDITIONS
                    and bid_sum > 1.0):
                calc = self.calculate_sell_arb(bid_prices)
                net_profit_per_unit = calc['net_profit']

                if (net_profit_per_unit > MIN_PROFIT_THRESHOLD
                        and net_profit_per_unit < MAX_PROFIT_THRESHOLD):
                    min_liq = min(liquidities_bid) if liquidities_bid else 0
                    if min_liq < MIN_LIQUIDITY:
                        self.diagnostics['filtered_low_liquidity'] += 1
                    else:
                        expected_profit = net_profit_per_unit * min_liq
                        capital_required = calc['capital_required'] * min_liq
                        roi = calc['roi_pct'] / 100.0

                        # Slippage
                        est_slippage = 0.0
                        net_after_slippage = expected_profit
                        if slippage_estimator:
                            max_slip = max(
                                slippage_estimator.estimate_slippage_from_book(
                                    bl, DEFAULT_TRADE_SIZE).get('slippage_pct', 0)
                                for bl in all_bids_lists
                            ) if all_bids_lists else 0
                            est_slippage = max_slip
                            if est_slippage > SLIPPAGE_TOLERANCE:
                                self.diagnostics['filtered_high_slippage'] += 1
                                return None
                            slippage_cost = est_slippage * capital_required
                            net_after_slippage = expected_profit - slippage_cost
                            if net_after_slippage <= 0:
                                return None

                        self.diagnostics['negrisk_found'] += 1
                        return ArbitrageOpportunity(
                            market_id=str(event_id),
                            market_name=event_title[:80] or 'Unknown Event',
                            opportunity_type='SELL_ARB',
                            expected_profit=expected_profit,
                            roi=roi,
                            capital_required=capital_required,
                            risk_score=self._calculate_risk_score(
                                first_cond, 'negrisk'),
                            urgency=('high' if roi > HIGH_URGENCY_ROI
                                      else 'medium' if roi > MEDIUM_URGENCY_ROI
                                      else 'low'),
                            details={
                                'strategy': 'negrisk',
                                'num_conditions': len(conditions_data),
                                'bid_sum': calc['bid_sum'],
                                'yes_bid_prices': [f"{p:.4f}" for p in bid_prices],
                                'fees': calc['fees'],
                                'min_liquidity': min_liq,
                                'capital_efficiency': f'{CAPITAL_EFFICIENCY_MULTIPLIER}x',
                                'event_title': event_title,
                                'market_slug': market_slug,
                            },
                            timestamp=datetime.now(),
                            net_profit_after_fees=expected_profit,
                            merge_advice=(
                                "Mint complete set for $1 via NegRisk Adapter, "
                                "sell all YES shares for profit"),
                            estimated_slippage=est_slippage,
                            net_profit_after_slippage=net_after_slippage,
                        )

            return None

        except Exception as e:
            logger.debug(f"Error in NegRisk detection: {e}")
            return None

    # ------------------------------------------------------------------
    # Risk scoring
    # ------------------------------------------------------------------
    def _calculate_risk_score(self, market: Dict, strategy_type: str) -> float:
        """Calculate risk score (0-1, lower is better)"""
        try:
            risk = 0.0

            end_date_str = (market.get('end_date_iso') or market.get('end_date') or
                            market.get('close_time'))
            if end_date_str:
                try:
                    end_date = datetime.fromisoformat(str(end_date_str).replace('Z', '+00:00'))
                    days = (end_date - datetime.now()).days
                    if days < 2:
                        risk += 0.4
                    elif days < 7:
                        risk += 0.2
                except Exception:
                    pass

            if strategy_type == 'negrisk':
                num_tokens = len(market.get('tokens', []))
                risk += min(0.2, num_tokens * 0.03)

            question = (market.get('question') or market.get('title') or
                        market.get('description') or '').lower()

            subjective_kw = ['best', 'winner', 'better', 'more popular', 'succeed', 'who will']
            objective_kw = ['election', 'vote', 'score', 'price above', 'gdp', 'temperature']

            if any(k in question for k in subjective_kw):
                risk += 0.3
            if any(k in question for k in objective_kw):
                risk -= 0.1  # Lower risk for objective markets

            return max(0.0, min(1.0, risk))
        except Exception:
            return 0.5


# ============================================================================
# ALERT MANAGER - WITH NOTIFICATION INTEGRATION
# ============================================================================

class AlertManager:
    """Manage and display opportunities with multi-channel notifications"""

    def __init__(self, notification_manager=None):
        self.displayed_opportunities = {}  # {opp_key: timestamp}
        self.notification_manager = notification_manager

    def display_opportunity(self, opp: ArbitrageOpportunity):
        """Display BUY_ARB / SELL_ARB opportunity and send notifications"""

        opp_key = f"{opp.market_id}_{opp.opportunity_type}"

        # Rate limit: same opportunity within ALERT_RATE_LIMIT_SECONDS
        now = time.time()
        if opp_key in self.displayed_opportunities:
            last_time = self.displayed_opportunities[opp_key]
            if now - last_time < ALERT_RATE_LIMIT_SECONDS:
                return
        self.displayed_opportunities[opp_key] = now

        # Console output
        urgency_symbol = ("🔴" if opp.urgency == 'high'
                           else "🟡" if opp.urgency == 'medium'
                           else "🟢")

        is_buy = opp.opportunity_type == 'BUY_ARB'
        strategy = opp.details.get('strategy', 'unknown')

        # Build Polymarket URL
        slug = opp.details.get('market_slug', '')
        if strategy == 'negrisk' and slug:
            market_url = f"https://polymarket.com/event/{slug}"
        elif slug:
            market_url = f"https://polymarket.com/market/{slug}"
        else:
            market_url = ''

        print("\n" + "=" * 80)
        print(f"{urgency_symbol} {opp.opportunity_type} OPPORTUNITY ({strategy})")
        print("=" * 80)
        print(f"Market: {opp.market_name}")
        if market_url:
            print(f"URL: {market_url}")

        if is_buy:
            ask_sum = opp.details.get('ask_sum', 0)
            print(f"sum (best ask): {ask_sum:.4f} < 1.00")
        else:
            bid_sum = opp.details.get('bid_sum', 0)
            print(f"sum (best bid): {bid_sum:.4f} > 1.00")

        print(f"Net Profit: ${opp.net_profit_after_fees:.4f} (ROI: {opp.roi * 100:.2f}%)")
        print(f"Capital: ${opp.capital_required:.2f}" +
              (" (mint)" if not is_buy else ""))

        if opp.merge_advice:
            print(f"Action: {opp.merge_advice}")

        if opp.estimated_slippage > 0:
            print(f"Est. Slippage: {opp.estimated_slippage * 100:.2f}%")
            print(f"Net After Slippage: ${opp.net_profit_after_slippage:.4f}")

        print(f"Risk: slippage + {'gas' if is_buy else 'mint gas'}")

        print(f"\nDetails:")
        for key, value in opp.details.items():
            if key in ('market_slug', 'strategy'):
                continue
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print(f"\nTimestamp: {opp.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80 + "\n")

        logger.info(
            f"OPPORTUNITY: {opp.opportunity_type} ({strategy}) - "
            f"${opp.net_profit_after_fees:.4f} profit, {opp.roi * 100:.1f}% ROI")

        # Send external notifications
        if self.notification_manager:
            try:
                asyncio.get_event_loop().create_task(
                    self.notification_manager.send(opp)
                )
            except RuntimeError:
                pass

    def generate_summary(self, opportunities: List[ArbitrageOpportunity],
                          client_diag: Dict, detector_diag: Dict):
        """Generate summary with diagnostics"""

        print("\n" + "=" * 80)
        print("SCAN DIAGNOSTICS")
        print("=" * 80)
        print(f"Markets fetched: {client_diag['markets_fetched']}")
        print(f"  NegRisk events: {client_diag['negrisk_events_fetched']}")
        print(f"  NegRisk markets extracted: {client_diag['negrisk_markets_extracted']}")
        print(f"Markets with tokens: {client_diag['markets_with_tokens']}")
        print(f"Orderbooks fetched: {client_diag['orderbooks_fetched']}")
        print(f"Orderbooks with data: {client_diag['orderbooks_with_data']}")
        print(f"\nSingle-Condition checked: {detector_diag['single_condition_checked']}")
        print(f"Single-Condition found: {detector_diag['single_condition_found']}")
        print(f"NegRisk checked: {detector_diag['negrisk_checked']}")
        print(f"NegRisk found: {detector_diag['negrisk_found']}")
        print(f"\nFiltered (low liquidity): {detector_diag['filtered_low_liquidity']}")
        print(f"Filtered (high slippage): {detector_diag['filtered_high_slippage']}")
        print(f"No orderbook: {detector_diag['no_orderbook']}")
        print(f"No prices: {detector_diag['no_prices']}")
        print("=" * 80)

        if not opportunities:
            print("\nNo opportunities detected in this scan.\n")
            return

        total_profit = sum(o.expected_profit for o in opportunities)
        total_capital = sum(o.capital_required for o in opportunities)
        avg_roi = np.mean([o.roi for o in opportunities]) * 100

        by_type = defaultdict(list)
        for o in opportunities:
            by_type[o.opportunity_type].append(o)

        print("\n" + "=" * 80)
        print("OPPORTUNITIES SUMMARY")
        print("=" * 80)
        print(f"Total Opportunities: {len(opportunities)}")
        print(f"Total Expected Profit: ${total_profit:.4f}")
        print(f"Total Capital Required: ${total_capital:.2f}")
        print(f"Average ROI: {avg_roi:.2f}%")
        print(f"\nBy Strategy:")
        for strategy, opps in by_type.items():
            sp = sum(o.expected_profit for o in opps)
            print(f"  {strategy}: {len(opps)} opportunities, ${sp:.4f} profit")
        print("=" * 80 + "\n")


# ============================================================================
# MAIN BOT
# ============================================================================

class PredictionMarketBot:
    """Main bot orchestrator: reads from SQLite DB, mid-price screens, fetches orderbooks only for candidates."""

    def __init__(self):
        self.detector = ArbitrageDetector()
        self.notification_manager = None
        self.alert_manager = None
        self.slippage_estimator = None
        self.ws_handler = None
        self.scanner_process = None
        self.scan_count = 0

    async def initialize(self):
        """Initialize notification manager, slippage estimator"""
        # Notifications
        try:
            from notifications import NotificationManager
            self.notification_manager = NotificationManager()
            logger.info("Notification manager initialized")
        except ImportError:
            logger.info("notifications.py not found, using console only")
        except Exception as e:
            logger.warning(f"Failed to init notifications: {e}")

        self.alert_manager = AlertManager(notification_manager=self.notification_manager)

        # Slippage estimator
        try:
            from orderbook_utils import SlippageEstimator
            self.slippage_estimator = SlippageEstimator()
            logger.info("Slippage estimator initialized")
        except ImportError:
            logger.info("orderbook_utils.py not found, skipping slippage estimation")
        except Exception as e:
            logger.warning(f"Failed to init slippage estimator: {e}")

    def _read_markets_from_db(self) -> List[Dict]:
        """Read all active markets from SQLite DB (instant)."""
        try:
            conn = sqlite3.connect(DB_FILE, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT condition_id, data, mid_price_sum, num_tokens, is_negrisk, "
                "event_id, event_title, volume FROM markets WHERE active=1"
            ).fetchall()
            conn.close()

            markets = []
            for row in rows:
                try:
                    market_data = json.loads(row['data'])
                    market_data['_db_mid_price_sum'] = row['mid_price_sum']
                    market_data['_db_num_tokens'] = row['num_tokens']
                    market_data['_db_is_negrisk'] = row['is_negrisk']
                    market_data['_db_volume'] = row['volume']
                    if row['is_negrisk']:
                        market_data['_is_negrisk'] = True
                        market_data['_event_id'] = row['event_id']
                        market_data['_event_title'] = row['event_title']
                    markets.append(market_data)
                except (json.JSONDecodeError, Exception) as e:
                    logger.debug(f"Failed to parse market data: {e}")
                    continue
            return markets
        except Exception as e:
            logger.error(f"Failed to read from DB: {e}")
            return []

    def _mid_price_screen(self, markets: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Mid-price pre-screening with separate logic for binary vs NegRisk.

        NegRisk (is_negrisk=1):
          - Group by event_id, sum YES mid prices across all conditions in the event
          - If event_sum < MID_SUM_THRESHOLD or > MID_SUM_UPPER -> all conditions are candidates
          - Require >= NEGRISK_MIN_CONDITIONS per event
          - Fallback: top N events by total volume

        Binary (is_negrisk=0):
          - mid_price_sum = YES_mid + NO_mid for each condition
          - If < MID_SUM_THRESHOLD (buy) or > MID_SUM_UPPER (sell) -> candidate
          - Fallback: top N by volume

        Returns dict with 'negrisk', 'binary', and counts.
        """
        negrisk_events = defaultdict(list)  # event_id -> [market_dicts]
        binary_pool = []

        for m in markets:
            if m.get('_db_is_negrisk'):
                event_id = m.get('_event_id', '')
                if event_id:
                    negrisk_events[event_id].append(m)
            else:
                n_tokens = m.get('_db_num_tokens', 0)
                if n_tokens >= 2:
                    binary_pool.append(m)

        # --- NegRisk screening: event-level YES sum ---
        negrisk_cands = []
        negrisk_events_passed = 0
        negrisk_all_events = []  # [(event_id, conditions)]

        for event_id, conditions in negrisk_events.items():
            if len(conditions) < NEGRISK_MIN_CONDITIONS:
                continue
            negrisk_all_events.append((event_id, conditions))

            event_sum = sum(m.get('_db_mid_price_sum', 0) for m in conditions)
            if event_sum <= 0:
                continue

            logger.debug(
                f"NegRisk event {event_id}: {len(conditions)} conditions, "
                f"YES sum={event_sum:.4f}")

            if event_sum < MID_SUM_THRESHOLD or event_sum > MID_SUM_UPPER:
                negrisk_events_passed += 1
                negrisk_cands.extend(conditions)

        # NegRisk fallback: top N events by total volume
        negrisk_total_conditions = sum(
            len(c) for _, c in negrisk_all_events) if negrisk_all_events else 0
        if not negrisk_cands and negrisk_all_events:
            def _event_volume(ev):
                return sum(m.get('_db_volume', 0) for m in ev[1])
            negrisk_all_events.sort(key=_event_volume, reverse=True)
            for _, conditions in negrisk_all_events[:NEGRISK_TOP_N_BY_VOLUME]:
                negrisk_cands.extend(conditions)

        # --- Binary screening: YES_mid + NO_mid ---
        binary_cands = []
        binary_mid_passed = 0

        for m in binary_pool:
            mid_sum = m.get('_db_mid_price_sum', 0)
            if mid_sum <= 0:
                continue
            if mid_sum < MID_SUM_THRESHOLD or mid_sum > MID_SUM_UPPER:
                binary_mid_passed += 1
                binary_cands.append(m)

        # Binary fallback: top N by volume
        if not binary_cands:
            binary_pool.sort(key=lambda m: m.get('_db_volume', 0), reverse=True)
            binary_cands = binary_pool[:BINARY_TOP_N_BY_VOLUME]

        return {
            'negrisk': negrisk_cands,
            'binary': binary_cands,
            'negrisk_events_total': len(negrisk_all_events),
            'negrisk_events_passed': negrisk_events_passed,
            'negrisk_total': negrisk_total_conditions,
            'binary_total': len(binary_pool),
            'binary_mid_passed': binary_mid_passed,
        }

    async def analyze_negrisk_candidates(
        self, candidates: List[Dict], client: PolymarketClient
    ) -> List[ArbitrageOpportunity]:
        """
        Analyze NegRisk candidates grouped by event_id.
        For each event: fetch YES orderbook for each condition, run event-level detection.
        """
        opportunities = []

        # Group by event_id
        events = defaultdict(list)
        for m in candidates:
            event_id = m.get('_event_id', '')
            if event_id:
                events[event_id].append(m)

        logger.info(f"Processing {len(events)} NegRisk events "
                     f"({len(candidates)} total conditions)")

        for event_id, conditions in events.items():
            if len(conditions) < NEGRISK_MIN_CONDITIONS:
                continue

            try:
                event_title = conditions[0].get('_event_title', '')
                event_data = []  # [(condition_dict, yes_token_id, yes_orderbook)]

                for condition in conditions:
                    tokens = _parse_tokens(condition)
                    if not tokens:
                        continue

                    # Find YES token
                    yes_token_id = None
                    for t in tokens:
                        if isinstance(t, dict) and t.get('outcome') == 'Yes':
                            yes_token_id = t.get('token_id') or t.get('id')
                            break

                    if not yes_token_id:
                        continue

                    book = await client.get_orderbook(yes_token_id)
                    if book:
                        event_data.append((condition, yes_token_id, book))
                    await asyncio.sleep(DELAY_BETWEEN_ORDERBOOKS)
                    client.diagnostics['markets_analyzed'] += 1

                if len(event_data) < NEGRISK_MIN_CONDITIONS:
                    continue

                client.diagnostics['markets_with_tokens'] += len(event_data)

                opp = self.detector.detect_negrisk_arbitrage(
                    event_id=event_id,
                    event_title=event_title,
                    conditions_data=event_data,
                    slippage_estimator=self.slippage_estimator,
                )
                if opp:
                    opportunities.append(opp)
                    self.alert_manager.display_opportunity(opp)

            except Exception as e:
                logger.debug(f"Error analyzing NegRisk event {event_id}: {e}")
                continue

            await asyncio.sleep(DELAY_BETWEEN_MARKETS)

        return opportunities

    async def analyze_binary_batch(
        self, markets: List[Dict], client: PolymarketClient,
        batch_num: int = 1, total_batches: int = 1,
    ) -> List[ArbitrageOpportunity]:
        """
        Analyze a batch of binary market candidates.
        For each: fetch YES + NO orderbooks, run binary arb detection.
        """
        opportunities = []
        logger.info(f"Processing binary batch {batch_num}/{total_batches} "
                     f"({len(markets)} candidates)")

        for market in markets:
            try:
                tokens = _parse_tokens(market)
                if not tokens or len(tokens) < 2:
                    continue

                client.diagnostics['markets_with_tokens'] += 1

                # Find YES and NO tokens
                yes_tid = None
                no_tid = None
                for t in tokens:
                    if isinstance(t, dict):
                        tid = t.get('token_id') or t.get('id')
                        outcome = t.get('outcome', '')
                        if outcome == 'Yes':
                            yes_tid = tid
                        elif outcome == 'No':
                            no_tid = tid

                if not yes_tid or not no_tid:
                    continue

                # Fetch orderbooks for both sides
                yes_ob = await client.get_orderbook(yes_tid)
                await asyncio.sleep(DELAY_BETWEEN_ORDERBOOKS)
                no_ob = await client.get_orderbook(no_tid)
                await asyncio.sleep(DELAY_BETWEEN_ORDERBOOKS)

                opp = self.detector.detect_single_condition_arbitrage(
                    market, yes_ob, no_ob,
                    slippage_estimator=self.slippage_estimator,
                )
                if opp:
                    opportunities.append(opp)
                    self.alert_manager.display_opportunity(opp)

                await asyncio.sleep(DELAY_BETWEEN_MARKETS)
                client.diagnostics['markets_analyzed'] += 1

            except Exception as e:
                logger.debug(f"Error analyzing binary market: {e}")
                continue

        return opportunities

    async def run_single_scan(self):
        """Run one detection cycle: read DB -> mid-price screen -> fetch orderbooks for candidates."""
        self.scan_count += 1
        t0 = time.time()
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Starting scan #{self.scan_count} (DB read -> mid-price screen -> orderbook)")
        logger.info(f"{'=' * 80}\n")

        # Reset diagnostics
        self.detector.diagnostics = {k: 0 for k in self.detector.diagnostics}

        # Step 1: Read from DB (instant)
        all_markets = self._read_markets_from_db()
        if not all_markets:
            logger.warning("No markets in DB. Waiting for background scanner...")
            return []

        # Step 2: Mid-price pre-screening (separate binary vs NegRisk)
        screen = self._mid_price_screen(all_markets)
        negrisk_cands = screen['negrisk']
        binary_cands = screen['binary']
        total_candidates = len(negrisk_cands) + len(binary_cands)

        logger.info(
            f"DB markets: {len(all_markets)} "
            f"(binary: {screen['binary_total']}, "
            f"negrisk: {screen['negrisk_total']} in "
            f"{screen['negrisk_events_total']} events)"
        )

        # NegRisk logging
        if screen.get('negrisk_events_passed', 0) == 0 and len(negrisk_cands) > 0:
            logger.info(
                f"  NegRisk: 0 events passed mid-screen (API prices normalized), "
                f"using top {len(negrisk_cands)} conditions by volume"
            )
        else:
            logger.info(
                f"  NegRisk mid-screen: {screen.get('negrisk_events_passed', 0)} events passed "
                f"-> {len(negrisk_cands)} conditions "
                f"(event YES sum <{MID_SUM_THRESHOLD} or >{MID_SUM_UPPER})"
            )

        # Binary logging
        binary_mid_passed = screen.get('binary_mid_passed', 0)
        if binary_mid_passed > 0:
            logger.info(
                f"  Binary mid-screen: {binary_mid_passed}/{screen['binary_total']} passed "
                f"(YES+NO sum <{MID_SUM_THRESHOLD} or >{MID_SUM_UPPER})")
        else:
            logger.info(
                f"  Binary: {len(binary_cands)}/{screen['binary_total']} "
                f"(top by volume, mid-screen={binary_mid_passed})")

        logger.info(f"  Total candidates for orderbook: {total_candidates}")

        if total_candidates == 0:
            elapsed = time.time() - t0
            logger.info(f"Scan #{self.scan_count} complete in {elapsed:.1f}s. "
                         f"No candidates.")
            return []

        # Step 3: Fetch orderbooks and detect
        async with PolymarketClient() as client:
            all_opportunities = []

            # 3a. NegRisk events (process all at once to maintain event grouping)
            if negrisk_cands:
                negrisk_opps = await self.analyze_negrisk_candidates(
                    negrisk_cands, client)
                all_opportunities.extend(negrisk_opps)

            # 3b. Binary markets (process in batches)
            if binary_cands:
                total_batches = max(1, (len(binary_cands) + BATCH_SIZE - 1) // BATCH_SIZE)
                for i in range(0, len(binary_cands), BATCH_SIZE):
                    batch = binary_cands[i:i + BATCH_SIZE]
                    batch_num = (i // BATCH_SIZE) + 1
                    batch_opps = await self.analyze_binary_batch(
                        batch, client, batch_num, total_batches)
                    all_opportunities.extend(batch_opps)
                    await asyncio.sleep(DELAY_BETWEEN_BATCHES)

            # Build diagnostics for summary
            client_diag = {
                'markets_fetched': len(all_markets),
                'negrisk_events_fetched': screen['negrisk_events_total'],
                'negrisk_markets_extracted': len(negrisk_cands),
                'markets_with_tokens': client.diagnostics['markets_with_tokens'],
                'orderbooks_fetched': client.diagnostics['orderbooks_fetched'],
                'orderbooks_with_data': client.diagnostics['orderbooks_with_data'],
            }

            self.alert_manager.generate_summary(
                all_opportunities, client_diag, self.detector.diagnostics
            )

            elapsed = time.time() - t0
            logger.info(
                f"Scan #{self.scan_count} complete in {elapsed:.1f}s. "
                f"db={len(all_markets)}, negrisk_cands={len(negrisk_cands)}, "
                f"binary_cands={len(binary_cands)}, "
                f"orderbooks={client.diagnostics['orderbooks_fetched']}, "
                f"opportunities={len(all_opportunities)}"
            )
            logger.info(f"Next scan in {SCAN_INTERVAL} seconds...\n")

            return all_opportunities

    def _wait_for_db_data(self, max_wait: int = 120):
        """Block until DB has data (first run). Poll every 5s, max 2min."""
        import os
        if not os.path.exists(DB_FILE):
            logger.info(f"DB file {DB_FILE} not found, waiting for background scanner...")

        waited = 0
        while waited < max_wait:
            try:
                if os.path.exists(DB_FILE):
                    conn = sqlite3.connect(DB_FILE, timeout=5)
                    count = conn.execute(
                        "SELECT COUNT(*) FROM markets WHERE active=1"
                    ).fetchone()[0]
                    conn.close()
                    if count > 0:
                        logger.info(f"DB ready: {count} active markets (waited {waited}s)")
                        return True
            except Exception:
                pass  # DB not ready yet
            time.sleep(5)
            waited += 5
            if waited % 15 == 0:
                logger.info(f"Still waiting for DB data... ({waited}s elapsed)")

        logger.error(f"DB still empty after {max_wait}s. Starting anyway.")
        return False

    async def run_continuous(self):
        """Run continuous monitoring with background scanner + WebSocket."""
        await self.initialize()

        logger.info("Prediction Market Arbitrage Bot v2.0 Starting...")
        logger.info(f"Scan interval: {SCAN_INTERVAL}s (frontend detection)")
        logger.info(f"Background scan interval: {BACKGROUND_SCAN_INTERVAL}s")
        logger.info(f"Mid-price screen: buy < {MID_SUM_THRESHOLD}, sell > {MID_SUM_UPPER}")
        logger.info(f"Min profit: ${MIN_PROFIT_THRESHOLD}")
        logger.info(f"Min liquidity: ${MIN_LIQUIDITY}")
        logger.info(f"Slippage tolerance: {SLIPPAGE_TOLERANCE * 100}%")
        logger.info(f"Notifications: {NOTIFICATION_METHODS}")
        logger.info(f"WebSocket: {'enabled' if WS_ENABLED else 'disabled'}")
        logger.info(f"DB file: {DB_FILE}")
        logger.info("=" * 80 + "\n")

        # Start background scanner process
        try:
            from scanner_process import start_scanner_process
            self.scanner_process = start_scanner_process(DB_FILE)
            logger.info(f"Background scanner process started (PID: {self.scanner_process.pid})")
        except Exception as e:
            logger.error(f"Failed to start background scanner: {e}")
            logger.warning("Running without background scanner - DB may be stale")

        # Wait for DB to have data (first run boundary handling)
        self._wait_for_db_data(max_wait=120)

        # Start WebSocket
        ws_task = None
        if WS_ENABLED:
            try:
                from websocket_handler import WebSocketHandler
                self.ws_handler = WebSocketHandler(
                    detector=self.detector,
                    alert_manager=self.alert_manager,
                    slippage_estimator=self.slippage_estimator,
                )
                ws_task = asyncio.create_task(self.ws_handler.start())
                logger.info("WebSocket handler started")
            except ImportError:
                logger.info("websocket_handler.py not found, using polling only")
            except Exception as e:
                logger.warning(f"Failed to start WebSocket: {e}")

        # Main detection loop
        while True:
            try:
                # Check if scanner process is alive, restart if dead
                if self.scanner_process and not self.scanner_process.is_alive():
                    logger.warning("Background scanner process died, restarting...")
                    try:
                        from scanner_process import start_scanner_process
                        self.scanner_process = start_scanner_process(DB_FILE)
                        logger.info(f"Scanner restarted (PID: {self.scanner_process.pid})")
                    except Exception as e:
                        logger.error(f"Failed to restart scanner: {e}")

                await self.run_single_scan()
                await asyncio.sleep(SCAN_INTERVAL)

            except KeyboardInterrupt:
                logger.info("\nBot stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                logger.info(f"Retrying in {SCAN_INTERVAL} seconds...")
                await asyncio.sleep(SCAN_INTERVAL)

        # Graceful shutdown
        logger.info("Shutting down...")
        if self.ws_handler:
            await self.ws_handler.stop()
        if self.scanner_process and self.scanner_process.is_alive():
            logger.info("Terminating background scanner...")
            self.scanner_process.terminate()
            self.scanner_process.join(timeout=5)
            if self.scanner_process.is_alive():
                self.scanner_process.kill()
            logger.info("Background scanner terminated")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main entry point"""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║              PREDICTION MARKET ARBITRAGE BOT v2.0                       ║
║                   DB-BACKED MID-PRICE SCREENING                         ║
║                                                                         ║
║  Background:  Independent process scans all markets -> SQLite WAL DB   ║
║  Frontend:    Read DB -> mid-price screen -> orderbook only for cands  ║
║  Real-time:   WebSocket price updates + DB writes                      ║
║  Detection:   Best Ask + Fee Deduction + Slippage Estimation           ║
║  Alerts:      Console + Telegram + Discord                             ║
║                                                                         ║
║  ⚠️  DISCLAIMER: Detection only - NOT automatic execution               ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)

    bot = PredictionMarketBot()

    try:
        await bot.run_continuous()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        logger.info("Bot shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
