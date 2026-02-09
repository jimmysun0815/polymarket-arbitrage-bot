"""
WebSocket handler for real-time Polymarket price updates (v2).

Architecture:
1. Watchlist built from SQLite DB (NegRisk + high volume, MIN_WATCHLIST_VOLUME filter)
2. WebSocket subscribes to price_change / book delta events
3. On update: update DB mid_price_sum -> mid-price screen -> trigger detection only if passes
4. Auto-reconnect with fallback to REST polling
"""

import asyncio
import json
import time
import sqlite3
import logging
from typing import Dict, List, Optional

import websockets

from config import (
    POLYMARKET_WS_URL,
    WS_SUBS_LIMIT,
    WS_RECONNECT_DELAY,
    WS_MAX_RECONNECT_ATTEMPTS,
    WS_FALLBACK_INTERVAL,
    WS_MID_SCREEN_COOLDOWN_SECONDS,
    MIN_WATCHLIST_VOLUME,
    MID_SUM_THRESHOLD,
    MID_SUM_UPPER,
    NEGRISK_MIN_CONDITIONS,
    DB_FILE,
)

logger = logging.getLogger(__name__)


class WebSocketHandler:
    """
    Real-time WebSocket price/book handler for Polymarket CLOB (v2).

    Key v2 improvements:
    - Watchlist built from SQLite DB (volume > MIN_WATCHLIST_VOLUME, NegRisk priority)
    - Price updates write back to DB (mid_price_sum)
    - Mid-price screening before triggering detection (avoid unnecessary orderbook fetches)
    """

    def __init__(self, detector=None, alert_manager=None, slippage_estimator=None,
                 db_file: str = DB_FILE):
        self.detector = detector
        self.alert_manager = alert_manager
        self.slippage_estimator = slippage_estimator
        self.db_file = db_file

        self.ws = None
        self.running = False
        self.fallback_mode = False
        self.reconnect_count = 0

        # In-memory price state: {token_id: {'price': float, 'asks': [...], 'bids': [...], 'updated': float}}
        self.price_state: Dict[str, Dict] = {}

        # Watchlist of token_ids to subscribe
        self._watchlist: List[str] = []

        # Market metadata: {token_id: market_data} for detection callbacks
        self._token_to_market: Dict[str, Dict] = {}

        # Condition_id -> list of token_ids (for mid_price_sum update)
        self._condition_tokens: Dict[str, List[str]] = {}

        # Token_id -> outcome type ("Yes" or "No")
        self._token_outcome: Dict[str, str] = {}

        # Rate limit: last time we logged + triggered detection per condition_id or event_id
        self._last_ws_trigger: Dict[str, float] = {}

    def _build_watchlist_from_db(self):
        """
        Build watchlist from SQLite DB.
        Priority: NegRisk markets first, then by volume desc.
        Filter: volume > MIN_WATCHLIST_VOLUME.
        Limit: WS_SUBS_LIMIT tokens.
        Also reads is_negrisk/event_id/event_title from DB columns for reliable metadata.
        """
        try:
            conn = sqlite3.connect(self.db_file, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.row_factory = sqlite3.Row

            rows = conn.execute("""
                SELECT condition_id, data, is_negrisk, event_id, event_title
                FROM markets
                WHERE active=1 AND volume > ?
                ORDER BY is_negrisk DESC, volume DESC
                LIMIT 500
            """, (MIN_WATCHLIST_VOLUME,)).fetchall()
            conn.close()

            token_ids = []
            token_market_map = {}
            condition_tokens = {}
            token_outcome = {}

            for row in rows:
                try:
                    market = json.loads(row['data'])
                    cid = row['condition_id']

                    # Ensure NegRisk metadata from DB columns (more reliable than JSON)
                    if row['is_negrisk']:
                        market['_is_negrisk'] = True
                        market['_event_id'] = row['event_id']
                        market['_event_title'] = row['event_title']

                    tokens = market.get('tokens') or []
                    if isinstance(tokens, str):
                        tokens = json.loads(tokens) if tokens else []

                    cid_tids = []
                    for t in tokens:
                        if isinstance(t, dict):
                            tid = t.get('token_id') or t.get('id')
                            if tid and tid not in token_market_map:
                                token_ids.append(tid)
                                token_market_map[tid] = market
                                cid_tids.append(tid)
                                # Track outcome type (Yes/No)
                                token_outcome[tid] = t.get('outcome', '')

                                if len(token_ids) >= WS_SUBS_LIMIT:
                                    break

                    if cid_tids:
                        condition_tokens[cid] = cid_tids

                except (json.JSONDecodeError, Exception):
                    continue

                if len(token_ids) >= WS_SUBS_LIMIT:
                    break

            self._watchlist = token_ids[:WS_SUBS_LIMIT]
            self._token_to_market = token_market_map
            self._condition_tokens = condition_tokens
            self._token_outcome = token_outcome

            logger.info(f"WS watchlist built from DB: {len(self._watchlist)} tokens "
                         f"(min_volume: ${MIN_WATCHLIST_VOLUME}, limit: {WS_SUBS_LIMIT})")

        except Exception as e:
            logger.error(f"Failed to build watchlist from DB: {e}")

    def update_watchlist(self, token_ids: List[str],
                          token_market_map: Dict[str, Dict] = None):
        """Manual watchlist update (kept for backward compatibility)."""
        self._watchlist = token_ids[:WS_SUBS_LIMIT]
        if token_market_map:
            self._token_to_market.update(token_market_map)
        logger.info(f"WebSocket watchlist manually updated: {len(self._watchlist)} tokens")

    async def start(self):
        """Start the WebSocket listener loop with auto-reconnect."""
        self.running = True
        self.reconnect_count = 0

        # Build watchlist from DB
        self._build_watchlist_from_db()

        while self.running:
            try:
                await self._connect_and_listen()
            except Exception as e:
                self.reconnect_count += 1
                logger.warning(f"WebSocket disconnected: {e} "
                               f"(reconnect {self.reconnect_count}/{WS_MAX_RECONNECT_ATTEMPTS})")

                if self.reconnect_count >= WS_MAX_RECONNECT_ATTEMPTS:
                    logger.warning("Max reconnect attempts reached, switching to fallback polling mode")
                    self.fallback_mode = True
                    return

                await asyncio.sleep(WS_RECONNECT_DELAY)

    async def stop(self):
        """Stop the WebSocket handler."""
        self.running = False
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
        logger.info("WebSocket handler stopped")

    async def _connect_and_listen(self):
        """Connect to WebSocket and listen for messages."""
        logger.info(f"Connecting to WebSocket: {POLYMARKET_WS_URL}")

        async with websockets.connect(POLYMARKET_WS_URL, ping_interval=30,
                                       ping_timeout=10) as ws:
            self.ws = ws
            self.reconnect_count = 0
            logger.info("WebSocket connected")

            # Subscribe to watchlist
            if self._watchlist:
                await self._subscribe(self._watchlist)
            else:
                logger.warning("WS watchlist empty — no subscriptions sent")

            # Listen loop
            async for raw_msg in ws:
                try:
                    msg = json.loads(raw_msg)
                    await self._on_message(msg)
                except json.JSONDecodeError:
                    logger.debug(f"Non-JSON WS message: {raw_msg[:200]}")
                except Exception as e:
                    logger.debug(f"WS message processing error: {e}")

    async def _subscribe(self, asset_ids: List[str]):
        """Send subscription message for asset_ids. Per Polymarket docs: first message
        uses type + assets_ids; further chunks use operation 'subscribe'."""
        if not self.ws or not asset_ids:
            return

        chunk_size = 50
        for i in range(0, len(asset_ids), chunk_size):
            chunk = asset_ids[i:i + chunk_size]
            if i == 0:
                # Initial subscription (required format on connect)
                # custom_feature_enabled: receive best_bid_ask + new_market + market_resolved
                sub_msg = {"type": "market", "assets_ids": chunk,
                           "custom_feature_enabled": True}
            else:
                sub_msg = {"operation": "subscribe", "assets_ids": chunk}
            await self.ws.send(json.dumps(sub_msg))
            logger.debug(f"Subscribed to {len(chunk)} assets (chunk {i // chunk_size + 1})")
            await asyncio.sleep(0.1)

        logger.info(f"Subscribed to {len(asset_ids)} asset_ids via WebSocket")

    async def _on_message(self, msg: dict):
        """Process incoming WebSocket message."""
        msg_type = msg.get('type') or msg.get('event_type') or msg.get('channel', '')

        if msg_type in ('price_change', 'tick'):
            await self._handle_price_change(msg)
        elif msg_type in ('book', 'book_delta', 'book_snapshot'):
            await self._handle_book_update(msg)
        elif msg_type == 'last_trade_price':
            await self._handle_last_trade_price(msg)
        elif msg_type == 'best_bid_ask':
            await self._handle_best_bid_ask(msg)
        elif msg_type in ('subscribed', 'connected', 'pong',
                          'tick_size_change', 'new_market', 'market_resolved'):
            logger.debug(f"WS control/info message: {msg_type}")
        else:
            logger.debug(f"Unknown WS message type: {msg_type}")

    async def _handle_price_change(self, msg: dict):
        """Handle a price_change event.

        Polymarket price_change format (2025+):
        {
            "event_type": "price_change",
            "market": "<condition_id>",
            "price_changes": [
                {"asset_id": "<token_id>", "price": "0.5", "size": "200",
                 "side": "BUY", "best_bid": "0.5", "best_ask": "0.6"},
                ...
            ]
        }
        We extract best_bid & best_ask per token and compute midpoint.
        """
        price_changes = msg.get('price_changes', [])

        if price_changes and isinstance(price_changes, list):
            # New format: parse each entry in price_changes array
            for entry in price_changes:
                asset_id = entry.get('asset_id', '')
                if not asset_id:
                    continue

                try:
                    best_bid = float(entry.get('best_bid', 0))
                    best_ask = float(entry.get('best_ask', 0))
                except (ValueError, TypeError):
                    continue

                # Compute midpoint for screening
                if best_bid > 0 and best_ask > 0:
                    mid = (best_bid + best_ask) / 2.0
                elif best_ask > 0:
                    mid = best_ask
                elif best_bid > 0:
                    mid = best_bid
                else:
                    continue

                if asset_id not in self.price_state:
                    self.price_state[asset_id] = {}

                self.price_state[asset_id]['price'] = mid
                self.price_state[asset_id]['updated'] = time.time()

                logger.debug(f"Price change: {asset_id[:16]}.. mid={mid:.4f} "
                             f"(bid={best_bid:.4f} ask={best_ask:.4f})")

                await self._update_db_and_screen(asset_id)
        else:
            # Legacy format fallback: top-level asset_id + price
            asset_id = msg.get('asset_id') or msg.get('market') or msg.get('token_id', '')
            price = msg.get('price') or msg.get('last_price')

            if asset_id and price is not None:
                try:
                    price = float(price)
                except (ValueError, TypeError):
                    return

                if asset_id not in self.price_state:
                    self.price_state[asset_id] = {}

                self.price_state[asset_id]['price'] = price
                self.price_state[asset_id]['updated'] = time.time()
                await self._update_db_and_screen(asset_id)

    async def _handle_book_update(self, msg: dict):
        """Handle an orderbook (book snapshot) update.

        Book format: asks sorted ascending, bids can be ascending or descending.
        We store the full asks/bids for detection, but compute MID price for screening.
        """
        asset_id = msg.get('asset_id') or msg.get('market') or msg.get('token_id', '')
        asks = msg.get('asks', [])
        bids = msg.get('bids', [])

        if asset_id and (asks or bids):
            if asset_id not in self.price_state:
                self.price_state[asset_id] = {}

            if asks:
                self.price_state[asset_id]['asks'] = asks
            if bids:
                self.price_state[asset_id]['bids'] = bids
            self.price_state[asset_id]['updated'] = time.time()

            # Compute midpoint (avg of best_bid and best_ask) for screening
            best_ask = 0.0
            best_bid = 0.0

            if asks:
                try:
                    if isinstance(asks[0], dict):
                        best_ask = float(asks[0].get('price', 0))
                    elif isinstance(asks[0], (list, tuple)):
                        best_ask = float(asks[0][0])
                    else:
                        best_ask = float(asks[0])
                except (ValueError, TypeError, IndexError):
                    pass

            if bids:
                try:
                    # bids may be ascending or descending; take max as best bid
                    bid_prices = []
                    for b in bids:
                        if isinstance(b, dict):
                            bid_prices.append(float(b.get('price', 0)))
                        elif isinstance(b, (list, tuple)):
                            bid_prices.append(float(b[0]))
                        else:
                            bid_prices.append(float(b))
                    if bid_prices:
                        best_bid = max(bid_prices)
                except (ValueError, TypeError, IndexError):
                    pass

            # Use midpoint for screening price; fall back to best_ask if no bids
            if best_bid > 0 and best_ask > 0:
                mid = (best_bid + best_ask) / 2.0
            elif best_ask > 0:
                mid = best_ask
            elif best_bid > 0:
                mid = best_bid
            else:
                mid = 0.0

            if mid > 0:
                self.price_state[asset_id]['price'] = mid

            logger.debug(f"Book update: {asset_id[:16]}.. mid={mid:.4f} "
                         f"(bid={best_bid:.4f} ask={best_ask:.4f}, "
                         f"{len(asks)} asks, {len(bids)} bids)")

            await self._update_db_and_screen(asset_id)

    async def _handle_last_trade_price(self, msg: dict):
        """Handle last_trade_price: {asset_id, price, side, size, ...}
        Use as a supplementary price signal (last traded price ≈ mid).
        """
        asset_id = msg.get('asset_id', '')
        price = msg.get('price')
        if not asset_id or price is None:
            return
        try:
            price = float(price)
        except (ValueError, TypeError):
            return
        if price <= 0:
            return

        if asset_id not in self.price_state:
            self.price_state[asset_id] = {}

        # Only update if we don't already have a mid from book/price_change
        if 'price' not in self.price_state[asset_id]:
            self.price_state[asset_id]['price'] = price
            self.price_state[asset_id]['updated'] = time.time()
            await self._update_db_and_screen(asset_id)

    async def _handle_best_bid_ask(self, msg: dict):
        """Handle best_bid_ask: {asset_id, best_bid, best_ask, spread, ...}
        Best source for midpoint computation.
        """
        asset_id = msg.get('asset_id', '')
        if not asset_id:
            return
        try:
            best_bid = float(msg.get('best_bid', 0))
            best_ask = float(msg.get('best_ask', 0))
        except (ValueError, TypeError):
            return

        if best_bid > 0 and best_ask > 0:
            mid = (best_bid + best_ask) / 2.0
        elif best_ask > 0:
            mid = best_ask
        elif best_bid > 0:
            mid = best_bid
        else:
            return

        if asset_id not in self.price_state:
            self.price_state[asset_id] = {}

        self.price_state[asset_id]['price'] = mid
        self.price_state[asset_id]['updated'] = time.time()

        logger.debug(f"Best bid/ask: {asset_id[:16]}.. mid={mid:.4f} "
                     f"(bid={best_bid:.4f} ask={best_ask:.4f})")

        await self._update_db_and_screen(asset_id)

    def _compute_mid_sum_for_condition(self, condition_id: str) -> float:
        """Compute mid_price_sum from in-memory price_state for a condition's tokens.

        - NegRisk conditions: returns only the YES token's mid price
          (event-level aggregation is done in _update_db_and_screen)
        - Binary conditions: returns YES mid + NO mid
          (for direct binary arb screening)

        Returns 0.0 if required price data is missing.
        """
        tids = self._condition_tokens.get(condition_id, [])
        if not tids:
            return 0.0

        # Check if this is a NegRisk condition
        is_negrisk = False
        for tid in tids:
            market = self._token_to_market.get(tid)
            if market and market.get('_is_negrisk'):
                is_negrisk = True
                break

        prices = []
        for tid in tids:
            outcome = self._token_outcome.get(tid, '')
            # NegRisk: only YES tokens; Binary: all tokens
            if is_negrisk and outcome != 'Yes':
                continue

            state = self.price_state.get(tid, {})
            p = state.get('price', 0)
            if p > 0:
                prices.append(p)

        if is_negrisk:
            # Need at least 1 YES token with price
            if not prices:
                return 0.0
        else:
            # Binary: need ALL tokens with price data
            if len(prices) < len(tids):
                return 0.0

        return sum(prices)

    async def _update_db_and_screen(self, asset_id: str):
        """
        After price update:
        1. Recompute per-condition mid_price_sum (YES-only for NegRisk, YES+NO for binary)
        2. Write updated mid_price_sum to DB
        3. Screen:
           - NegRisk: aggregate all conditions in same event, check event-level sum
           - Binary: check per-condition sum directly, trigger detection if passes
        """
        market = self._token_to_market.get(asset_id)
        if not market:
            return

        condition_id = market.get('condition_id') or market.get('conditionId') or ''
        if not condition_id:
            return

        is_negrisk = bool(market.get('_is_negrisk'))

        # Recompute per-condition mid_price_sum
        new_mid_sum = self._compute_mid_sum_for_condition(condition_id)
        if new_mid_sum <= 0:
            return

        # Write to DB
        try:
            conn = sqlite3.connect(self.db_file, timeout=5)
            conn.execute("PRAGMA journal_mode=WAL")
            with conn:
                conn.execute(
                    "UPDATE markets SET mid_price_sum=?, last_updated=? WHERE condition_id=?",
                    (new_mid_sum, time.time(), condition_id)
                )
            conn.close()
        except Exception as e:
            logger.debug(f"DB update error for {condition_id}: {e}")

        if is_negrisk:
            # --- NegRisk: event-level aggregation ---
            try:
                event_id = market.get('_event_id', '')
                if not event_id:
                    return

                conn = sqlite3.connect(self.db_file, timeout=5)
                conn.execute("PRAGMA journal_mode=WAL")
                rows = conn.execute(
                    "SELECT condition_id, mid_price_sum FROM markets "
                    "WHERE event_id=? AND is_negrisk=1 AND active=1",
                    (event_id,)
                ).fetchall()
                conn.close()

                if len(rows) < NEGRISK_MIN_CONDITIONS:
                    return

                event_sum = sum(row[1] for row in rows if row[1] > 0)
                n_conditions = len(rows)
                n_with_price = sum(1 for row in rows if row[1] > 0)

                logger.debug(
                    f"NegRisk event {event_id}: {n_with_price}/{n_conditions} conditions, "
                    f"YES sum={event_sum:.4f}")

                if event_sum < MID_SUM_THRESHOLD or event_sum > MID_SUM_UPPER:
                    now = time.time()
                    last = self._last_ws_trigger.get(event_id, 0)
                    if now - last >= WS_MID_SCREEN_COOLDOWN_SECONDS:
                        self._last_ws_trigger[event_id] = now
                        direction = 'buy' if event_sum < MID_SUM_THRESHOLD else 'sell'
                        event_title = market.get('_event_title', '')
                        logger.info(
                            f"WS NegRisk mid-screen PASS: \"{event_title}\" "
                            f"event_sum={event_sum:.4f} "
                            f"({direction}, {n_with_price}/{n_conditions} conditions) "
                            f"event_id={event_id}")
                        # Note: event-level detection requires all conditions' orderbooks,
                        # handled by the main scan loop, not WS trigger
            except Exception as e:
                logger.warning(f"WS NegRisk event aggregation error: {e}")
        else:
            # --- Binary: per-condition screening ---
            market_name = (market.get('question') or market.get('title') or
                           market.get('description') or '')
            if len(market_name) > 60:
                market_name = market_name[:57] + '...'

            if new_mid_sum < MID_SUM_THRESHOLD or new_mid_sum > MID_SUM_UPPER:
                now = time.time()
                last = self._last_ws_trigger.get(condition_id, 0)
                if now - last >= WS_MID_SCREEN_COOLDOWN_SECONDS:
                    self._last_ws_trigger[condition_id] = now
                    direction = 'buy' if new_mid_sum < MID_SUM_THRESHOLD else 'sell'

                    slug = market.get('market_slug') or market.get('slug', '')
                    market_url = f"https://polymarket.com/market/{slug}" if slug else ''

                    tids = self._condition_tokens.get(condition_id, [])
                    logger.info(
                        f"WS binary mid-screen PASS: \"{market_name}\" "
                        f"mid_sum={new_mid_sum:.4f} "
                        f"({direction}, {len(tids)} tokens) "
                        f"{market_url}")

                    try:
                        await self._trigger_detection(asset_id)
                    except Exception as e:
                        logger.warning(f"WS binary detection trigger error: {e}")
            else:
                logger.debug(
                    f"WS mid-screen skip: \"{market_name}\" "
                    f"mid_sum={new_mid_sum:.4f}")

    async def _trigger_detection(self, asset_id: str):
        """Trigger binary arbitrage detection using cached price_state orderbooks.
        NegRisk event-level detection is handled by the main scan loop, not WS trigger.
        """
        if not self.detector or not self.alert_manager:
            return

        market = self._token_to_market.get(asset_id)
        if not market:
            return

        # Only handle binary markets via WS trigger
        if market.get('_is_negrisk'):
            return

        tokens = market.get('tokens', [])
        if isinstance(tokens, str):
            try:
                tokens = json.loads(tokens)
            except Exception:
                tokens = []

        if len(tokens) != 2:
            return

        # Find YES and NO tokens
        yes_tid = None
        no_tid = None
        for token in tokens:
            if isinstance(token, dict):
                tid = token.get('token_id') or token.get('id')
                outcome = token.get('outcome', '')
                if outcome == 'Yes':
                    yes_tid = tid
                elif outcome == 'No':
                    no_tid = tid

        if not yes_tid or not no_tid:
            return

        # Build orderbooks from price_state cache
        yes_ob = None
        no_ob = None
        if yes_tid in self.price_state:
            state = self.price_state[yes_tid]
            if state.get('asks') or state.get('bids'):
                yes_ob = {'asks': state.get('asks', []), 'bids': state.get('bids', [])}
        if no_tid in self.price_state:
            state = self.price_state[no_tid]
            if state.get('asks') or state.get('bids'):
                no_ob = {'asks': state.get('asks', []), 'bids': state.get('bids', [])}

        if not yes_ob or not no_ob:
            return

        opp = self.detector.detect_single_condition_arbitrage(
            market, yes_ob, no_ob,
            slippage_estimator=self.slippage_estimator,
        )
        if opp:
            self.alert_manager.display_opportunity(opp)
