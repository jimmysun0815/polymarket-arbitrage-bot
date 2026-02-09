"""
Orderbook utilities for slippage estimation.

Provides:
- fetch_orderbook(): Fetch orderbook from CLOB API with TTL cache
- estimate_slippage(): Walk the ask side to compute avg fill price and slippage %
- SlippageEstimator: Class integrating fetch + estimate with caching
"""

import time
import logging
from typing import Dict, List, Optional, Tuple

import aiohttp

from config import (
    POLYMARKET_CLOB_URL,
    SLIPPAGE_TOLERANCE,
    ORDERBOOK_CACHE_TTL,
    DEFAULT_TRADE_SIZE,
    API_TIMEOUT,
)

logger = logging.getLogger(__name__)


def estimate_slippage(asks: List[Tuple[float, float]], target_amount: float) -> Dict:
    """
    Walk the ask side of the orderbook to estimate slippage for a given trade size.

    Args:
        asks: List of (price, size) tuples, sorted by price ascending.
        target_amount: Dollar amount to fill.

    Returns:
        {
            'avg_fill_price': float,
            'slippage_pct': float,     # (avg_price - best_ask) / best_ask
            'filled': float,            # total dollars filled
            'fully_filled': bool,       # whether we could fill the full amount
            'levels_used': int,         # number of price levels consumed
        }
    """
    if not asks or target_amount <= 0:
        return {
            'avg_fill_price': 0.0,
            'slippage_pct': 0.0,
            'filled': 0.0,
            'fully_filled': False,
            'levels_used': 0,
        }

    filled = 0.0
    total_cost = 0.0
    total_shares = 0.0
    levels_used = 0

    for price, size in asks:
        if price <= 0:
            continue

        # size is in shares; available_dollars = shares * price
        available_dollars = size * price
        needed = target_amount - filled
        fill_dollars = min(available_dollars, needed)
        fill_shares = fill_dollars / price

        total_cost += fill_shares * price
        total_shares += fill_shares
        filled += fill_dollars
        levels_used += 1

        if filled >= target_amount:
            break

    avg_fill_price = total_cost / total_shares if total_shares > 0 else 0.0
    best_ask = asks[0][0] if asks else 0.0
    slippage_pct = (avg_fill_price - best_ask) / best_ask if best_ask > 0 else 0.0

    return {
        'avg_fill_price': avg_fill_price,
        'slippage_pct': max(0.0, slippage_pct),
        'filled': filled,
        'fully_filled': filled >= target_amount,
        'levels_used': levels_used,
    }


def parse_orderbook_levels(raw_levels: list) -> List[Tuple[float, float]]:
    """
    Parse orderbook levels from API response into (price, size) tuples.
    Handles both dict format {'price': '0.5', 'size': '100'} and list format [0.5, 100].
    """
    parsed = []
    for level in raw_levels:
        try:
            if isinstance(level, dict):
                price = float(level.get('price', 0))
                size = float(level.get('size', 0))
            elif isinstance(level, (list, tuple)) and len(level) >= 2:
                price = float(level[0])
                size = float(level[1])
            else:
                continue

            if price > 0 and size > 0:
                parsed.append((price, size))
        except (ValueError, TypeError):
            continue

    return sorted(parsed, key=lambda x: x[0])  # Sort by price ascending


class SlippageEstimator:
    """
    Orderbook-based slippage estimator with TTL caching.

    Usage:
        estimator = SlippageEstimator()
        result = estimator.estimate_slippage_from_book(asks_raw, target_amount=100)
        # or async:
        result = await estimator.fetch_and_estimate(session, token_id, target_amount=100)
    """

    def __init__(self):
        # Cache: {token_id: {'data': orderbook_dict, 'expires': float}}
        self._cache: Dict[str, Dict] = {}

    def _get_cached(self, token_id: str) -> Optional[Dict]:
        """Return cached orderbook if not expired"""
        entry = self._cache.get(token_id)
        if entry and time.time() < entry['expires']:
            return entry['data']
        return None

    def _set_cache(self, token_id: str, data: Dict):
        """Cache orderbook with TTL"""
        self._cache[token_id] = {
            'data': data,
            'expires': time.time() + ORDERBOOK_CACHE_TTL,
        }

    def estimate_slippage_from_book(self, asks_raw: list,
                                      target_amount: float = None) -> Dict:
        """
        Estimate slippage from raw asks list (as returned by API).

        Args:
            asks_raw: Raw asks from orderbook API response
            target_amount: Dollar amount (default: DEFAULT_TRADE_SIZE)

        Returns:
            Slippage estimation dict
        """
        if target_amount is None:
            target_amount = DEFAULT_TRADE_SIZE

        parsed = parse_orderbook_levels(asks_raw)
        return estimate_slippage(parsed, target_amount)

    async def fetch_orderbook(self, session: aiohttp.ClientSession,
                                token_id: str) -> Optional[Dict]:
        """
        Fetch orderbook from CLOB API, with TTL cache.
        """
        # Check cache first
        cached = self._get_cached(token_id)
        if cached is not None:
            return cached

        try:
            url = f"{POLYMARKET_CLOB_URL}/book"
            params = {'token_id': token_id}
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; ArbitrageBot/2.0)'}

            async with session.get(url, params=params, headers=headers) as resp:
                if resp.status == 200:
                    book = await resp.json()
                    self._set_cache(token_id, book)
                    return book
                elif resp.status == 429:
                    # Rate limited - return cached if available (even expired)
                    entry = self._cache.get(token_id)
                    if entry:
                        logger.debug(f"429 on orderbook, using stale cache for {token_id}")
                        return entry['data']
                    return None
                else:
                    return None
        except Exception as e:
            logger.debug(f"Orderbook fetch error for {token_id}: {e}")
            # Fallback to stale cache
            entry = self._cache.get(token_id)
            if entry:
                return entry['data']
            return None

    async def fetch_and_estimate(self, session: aiohttp.ClientSession,
                                   token_id: str,
                                   target_amount: float = None) -> Dict:
        """
        Fetch orderbook and estimate slippage in one call.

        Returns:
            Slippage estimation dict, or empty/zero dict on failure.
        """
        if target_amount is None:
            target_amount = DEFAULT_TRADE_SIZE

        book = await self.fetch_orderbook(session, token_id)
        if not book:
            return {
                'avg_fill_price': 0.0,
                'slippage_pct': 0.0,
                'filled': 0.0,
                'fully_filled': False,
                'levels_used': 0,
                'shallow_book': True,
            }

        asks_raw = book.get('asks', [])
        result = self.estimate_slippage_from_book(asks_raw, target_amount)

        if not result['fully_filled']:
            result['shallow_book'] = True
            logger.debug(f"Shallow book warning for {token_id}: "
                          f"filled ${result['filled']:.2f} of ${target_amount:.2f}")
        else:
            result['shallow_book'] = False

        return result
