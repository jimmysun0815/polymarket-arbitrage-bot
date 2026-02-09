"""
Background market scanner process for Polymarket Arbitrage Bot v2.

Runs as a multiprocessing.Process, continuously scanning all markets via REST API
and maintaining a SQLite WAL database with enriched market data + pre-computed
mid_price_sum for fast frontend screening.

Architecture:
  - Independent process (main loop crash does not affect scanner)
  - SQLite WAL mode for concurrent read/write without locks
  - 3-step scan: NegRisk events (volume desc) -> all events (id desc, newest first) -> merge & dedup
  - Uses /events endpoint (official recommended) instead of /markets — fewer objects, faster response
  - CLOB enrichment for token_ids
  - Incremental DB updates (INSERT OR REPLACE)
  - Periodic cleanup (delete stale inactive markets) + VACUUM
"""

import asyncio
import aiohttp
import json
import time
import sqlite3
import logging
import multiprocessing
from typing import Dict, List, Optional

from config import (
    POLYMARKET_CLOB_URL, POLYMARKET_GAMMA_URL,
    API_TIMEOUT, RATE_LIMIT_RETRY_SLEEP, RATE_LIMIT_MAX_RETRIES,
    EVENTS_ORDER_FIELD, EVENTS_FETCH_LIMIT, NEGRISK_LIMIT,
    BACKGROUND_SCAN_INTERVAL, DB_FILE, DB_CLEANUP_AGE,
    DELAY_BETWEEN_MARKETS,
    LOG_LEVEL, LOG_FILE,
)

# Set up logging for this module (each process needs its own)
def _setup_logger():
    logger = logging.getLogger('scanner_process')
    if not logger.handlers:
        logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
        fh = logging.FileHandler(LOG_FILE)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [BG] %(message)s'))
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [BG] %(message)s'))
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


# ============================================================================
# DATABASE HELPERS
# ============================================================================

def init_db(db_file: str) -> sqlite3.Connection:
    """Initialize SQLite database with WAL mode and schema."""
    conn = sqlite3.connect(db_file, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row

    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS markets (
                condition_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                is_negrisk INTEGER DEFAULT 0,
                event_id TEXT DEFAULT '',
                event_title TEXT DEFAULT '',
                volume REAL DEFAULT 0,
                mid_price_sum REAL DEFAULT 0,
                num_tokens INTEGER DEFAULT 0,
                active INTEGER DEFAULT 1,
                last_updated REAL NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_active ON markets(active)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_negrisk ON markets(is_negrisk)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mid_sum ON markets(mid_price_sum)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_volume ON markets(volume)")

    return conn


def compute_mid_price_sum(market: dict) -> float:
    """
    Compute mid price sum for pre-screening.
    - NegRisk conditions: returns only the YES token's mid price
      (event-level aggregation happens in the screener)
    - Binary conditions: returns YES mid + NO mid

    Price source priority: (best_bid+best_ask)/2 > token['price'] > outcomePrices.
    Fallback: 0.0 (will be skipped by screener)
    """
    is_negrisk = bool(market.get('_is_negrisk'))

    # Method 1: from tokens list
    tokens = market.get('tokens') or []
    if isinstance(tokens, str):
        try:
            tokens = json.loads(tokens)
        except Exception:
            tokens = []

    if tokens and isinstance(tokens, list):
        prices = []
        for t in tokens:
            if isinstance(t, dict):
                outcome = t.get('outcome', '')
                # NegRisk: only YES tokens; Binary: all tokens
                if is_negrisk and outcome != 'Yes':
                    continue
                try:
                    # Prefer mid = (best_bid + best_ask) / 2
                    best_bid = float(t.get('best_bid', 0) or 0)
                    best_ask = float(t.get('best_ask', 0) or 0)
                    if best_bid > 0 and best_ask > 0:
                        p = (best_bid + best_ask) / 2.0
                    else:
                        p = float(t.get('price', 0))
                    if p > 0:
                        prices.append(p)
                except (ValueError, TypeError):
                    pass
        if prices:
            logging.getLogger('scanner_process').debug(
                f"compute_mid_price_sum: negrisk={is_negrisk} "
                f"prices={[f'{p:.4f}' for p in prices]} sum={sum(prices):.4f}")
            return sum(prices)

    # Method 2: from outcomePrices paired with outcomes
    outcome_prices = market.get('outcomePrices')
    outcomes = market.get('outcomes')
    if outcome_prices:
        if isinstance(outcome_prices, str):
            try:
                outcome_prices = json.loads(outcome_prices)
            except Exception:
                outcome_prices = []
        if isinstance(outcomes, str):
            try:
                outcomes = json.loads(outcomes)
            except Exception:
                outcomes = []
        if outcome_prices:
            try:
                if is_negrisk and outcomes and len(outcome_prices) == len(outcomes):
                    # Only YES outcomes for NegRisk
                    s = sum(float(p) for p, o in zip(outcome_prices, outcomes)
                            if str(o).strip() == 'Yes' and float(p) > 0)
                else:
                    # All outcomes for binary (YES + NO)
                    s = sum(float(p) for p in outcome_prices if float(p) > 0)
                if s > 0:
                    return s
            except (ValueError, TypeError):
                pass

    return 0.0


def count_tokens(market: dict) -> int:
    """Count number of tokens/outcomes in a market."""
    tokens = market.get('tokens') or []
    if isinstance(tokens, str):
        try:
            tokens = json.loads(tokens)
        except Exception:
            tokens = []
    if isinstance(tokens, list):
        return len(tokens)

    # Try clob_token_ids
    clob_ids = market.get('clob_token_ids')
    if clob_ids:
        if isinstance(clob_ids, str):
            try:
                clob_ids = json.loads(clob_ids)
            except Exception:
                clob_ids = []
        if isinstance(clob_ids, list):
            return len(clob_ids)

    return 0


# ============================================================================
# BACKGROUND SCANNER
# ============================================================================

class PolymarketScanner:
    """
    Background market scanner that maintains a SQLite database.
    Designed to run inside a multiprocessing.Process.
    """

    def __init__(self, db_file: str = DB_FILE):
        self.db_file = db_file
        self.logger = _setup_logger()
        self.scan_count = 0

    async def _request_with_retry(self, session: aiohttp.ClientSession,
                                    url: str, params: dict = None) -> Optional[any]:
        """HTTP GET with 429 retry."""
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; ArbitrageBot/2.0)'}
        for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
            try:
                async with session.get(url, params=params, headers=headers) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:
                        self.logger.warning(
                            f"429 rate limited, sleeping {RATE_LIMIT_RETRY_SLEEP}s "
                            f"(attempt {attempt + 1}/{RATE_LIMIT_MAX_RETRIES})")
                        await asyncio.sleep(RATE_LIMIT_RETRY_SLEEP)
                    else:
                        return None
            except Exception as e:
                self.logger.debug(f"Request error: {e}")
                return None
        return None

    # ------------------------------------------------------------------
    # Step A: NegRisk events
    # ------------------------------------------------------------------
    async def fetch_negrisk_events(self, session: aiohttp.ClientSession) -> List[Dict]:
        """Fetch NegRisk events sorted by volume desc."""
        url = f"{POLYMARKET_GAMMA_URL}/events"
        params = {
            'negRisk': 'true',
            'order': 'volume',
            'ascending': 'false',
            'limit': NEGRISK_LIMIT,
            'active': 'true',
            'closed': 'false',
        }
        data = await self._request_with_retry(session, url, params)
        if not data:
            self.logger.warning("Step A: Failed to fetch NegRisk events (empty response)")
            return []

        events = data if isinstance(data, list) else data.get('data', [])
        markets = []
        for event in events:
            event_id = event.get('id') or event.get('slug', '')
            for mkt in event.get('markets', []):
                mkt['_is_negrisk'] = True
                mkt['_event_id'] = event_id
                mkt['_event_title'] = event.get('title', '')
                mkt['_event_slug'] = event.get('slug', '')
                markets.append(mkt)

        self.logger.info(f"Step A: {len(events)} NegRisk events -> {len(markets)} markets")
        return markets

    # ------------------------------------------------------------------
    # Step B: All events (newest first) -> expand to markets
    # ------------------------------------------------------------------
    async def fetch_all_events_markets(self, session: aiohttp.ClientSession) -> List[Dict]:
        """
        Fetch all active events via /events?order=id&ascending=false (newest first),
        then expand each event's nested markets array.
        This is the official recommended approach — events are fewer than markets,
        responses are faster, and id desc gives natural time ordering.
        """
        all_markets = []
        total_events = 0
        offset = 0
        page_size = 100  # /events supports smaller pages, iterate more

        while total_events < EVENTS_FETCH_LIMIT:
            url = f"{POLYMARKET_GAMMA_URL}/events"
            params = {
                'limit': min(page_size, EVENTS_FETCH_LIMIT - total_events),
                'offset': offset,
                'order': EVENTS_ORDER_FIELD,
                'ascending': 'false',
                'closed': 'false',
            }
            data = await self._request_with_retry(session, url, params)
            if not data:
                break

            events = data if isinstance(data, list) else data.get('data', [])
            if not events:
                break

            for event in events:
                event_id = event.get('id') or event.get('slug', '')
                event_title = event.get('title', '')
                is_neg = bool(event.get('negRisk'))
                event_markets = event.get('markets', [])

                for mkt in event_markets:
                    # Skip closed/inactive markets within events
                    if mkt.get('closed') or mkt.get('archived'):
                        continue
                    if is_neg:
                        mkt['_is_negrisk'] = True
                    mkt['_event_id'] = event_id
                    mkt['_event_title'] = event_title
                    mkt['_event_slug'] = event.get('slug', '')
                    all_markets.append(mkt)

            total_events += len(events)

            if len(events) < page_size:
                break  # last page
            offset += page_size
            await asyncio.sleep(0.2)

        self.logger.info(
            f"Step B: {total_events} events (order={EVENTS_ORDER_FIELD} desc) "
            f"-> {len(all_markets)} markets expanded"
        )
        return all_markets

    # ------------------------------------------------------------------
    # Step C: Merge & dedup
    # ------------------------------------------------------------------
    @staticmethod
    def merge_and_dedup(negrisk_markets: List[Dict],
                         hot_markets: List[Dict]) -> List[Dict]:
        """Merge NegRisk first, then hot markets, dedup by condition_id."""
        seen = set()
        merged = []
        for mkt in negrisk_markets + hot_markets:
            cid = mkt.get('condition_id') or mkt.get('conditionId') or mkt.get('id', '')
            if cid and cid not in seen:
                seen.add(cid)
                merged.append(mkt)
        return merged

    # ------------------------------------------------------------------
    # CLOB enrichment
    # ------------------------------------------------------------------
    async def enrich_market(self, session: aiohttp.ClientSession,
                              condition_id: str) -> Optional[Dict]:
        """Fetch full market details from CLOB API."""
        url = f"{POLYMARKET_CLOB_URL}/markets/{condition_id}"
        return await self._request_with_retry(session, url)

    # ------------------------------------------------------------------
    # Main scan cycle
    # ------------------------------------------------------------------
    async def run_scan_cycle(self):
        """Execute one full scan cycle: fetch -> enrich -> write DB."""
        self.scan_count += 1
        t0 = time.time()
        self.logger.info(f"=== Background scan #{self.scan_count} starting ===")

        conn = init_db(self.db_file)
        timeout = aiohttp.ClientTimeout(total=API_TIMEOUT)

        enriched_count = 0
        total_scanned = 0

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # 3-step scan:
                # A) NegRisk events (by volume desc) — priority
                # B) All events (by id desc = newest first) — expand to markets
                # C) Merge & dedup (NegRisk first)
                negrisk = await self.fetch_negrisk_events(session)
                events_markets = await self.fetch_all_events_markets(session)
                all_markets = self.merge_and_dedup(negrisk, events_markets)
                total_scanned = len(all_markets)

                self.logger.info(f"Step C: {total_scanned} unique markets to process")

                # Enrich and write to DB
                for i, market in enumerate(all_markets):
                    cid = market.get('condition_id') or market.get('conditionId') or ''
                    if not cid:
                        continue

                    # CLOB enrichment (get token_ids)
                    enriched = await self.enrich_market(session, cid)
                    if enriched:
                        # Preserve NegRisk metadata
                        if market.get('_is_negrisk'):
                            enriched['_is_negrisk'] = True
                            enriched['_event_id'] = market.get('_event_id', '')
                            enriched['_event_title'] = market.get('_event_title', '')
                            enriched['_event_slug'] = market.get('_event_slug', '')
                        final = enriched
                        enriched_count += 1
                    else:
                        final = market

                    # Compute mid_price_sum
                    mid_sum = compute_mid_price_sum(final)
                    n_tokens = count_tokens(final)
                    vol = 0.0
                    try:
                        vol = float(market.get('volume') or market.get('volumeNum') or 0)
                    except (ValueError, TypeError):
                        pass

                    is_neg = 1 if market.get('_is_negrisk') else 0
                    evt_id = market.get('_event_id', '')
                    evt_title = market.get('_event_title', '')

                    is_active = 1
                    if final.get('closed') or final.get('archived'):
                        is_active = 0
                    if final.get('active') is False:
                        is_active = 0

                    with conn:
                        conn.execute("""
                            INSERT OR REPLACE INTO markets
                            (condition_id, data, is_negrisk, event_id, event_title,
                             volume, mid_price_sum, num_tokens, active, last_updated)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            cid, json.dumps(final, default=str),
                            is_neg, evt_id, evt_title,
                            vol, mid_sum, n_tokens, is_active, time.time()
                        ))

                    if (i + 1) % 200 == 0:
                        self.logger.info(f"  Processed {i + 1}/{total_scanned}...")

                    await asyncio.sleep(DELAY_BETWEEN_MARKETS)

                # Mark closed/inactive markets
                with conn:
                    # Markets that API says are closed
                    scanned_cids = set()
                    for m in all_markets:
                        cid = m.get('condition_id') or m.get('conditionId') or ''
                        if cid:
                            scanned_cids.add(cid)

                # Cleanup: delete old inactive markets
                cutoff = time.time() - DB_CLEANUP_AGE
                with conn:
                    deleted = conn.execute(
                        "DELETE FROM markets WHERE active=0 AND last_updated < ?",
                        (cutoff,)
                    ).rowcount

                if deleted:
                    self.logger.info(f"Cleaned up {deleted} stale inactive markets")

                # VACUUM every 10 scans
                if self.scan_count % 10 == 0:
                    conn.execute("VACUUM")
                    self.logger.info("DB VACUUM completed")

        except Exception as e:
            self.logger.error(f"Scan cycle error: {e}", exc_info=True)
        finally:
            db_count = conn.execute("SELECT COUNT(*) FROM markets WHERE active=1").fetchone()[0]
            conn.close()

        elapsed = time.time() - t0
        self.logger.info(
            f"BG scan #{self.scan_count}: {elapsed:.1f}s, "
            f"scanned: {total_scanned}, enriched: {enriched_count}, "
            f"db_active: {db_count}"
        )

    # ------------------------------------------------------------------
    # Main loop (runs inside the process)
    # ------------------------------------------------------------------
    async def run_forever(self):
        """Continuously scan markets."""
        self.logger.info("Background scanner starting...")
        while True:
            try:
                await self.run_scan_cycle()
            except Exception as e:
                self.logger.error(f"Scanner error: {e}", exc_info=True)
            self.logger.info(
                f"Next background scan in {BACKGROUND_SCAN_INTERVAL}s...")
            await asyncio.sleep(BACKGROUND_SCAN_INTERVAL)


# ============================================================================
# MULTIPROCESSING ENTRY POINT
# ============================================================================

def _scanner_process_entry(db_file: str):
    """Entry point for the background scanner process."""
    scanner = PolymarketScanner(db_file=db_file)
    asyncio.run(scanner.run_forever())


def start_scanner_process(db_file: str = DB_FILE) -> multiprocessing.Process:
    """Start the background scanner as an independent process."""
    proc = multiprocessing.Process(
        target=_scanner_process_entry,
        args=(db_file,),
        daemon=True,
        name="PolymarketScanner",
    )
    proc.start()
    return proc
