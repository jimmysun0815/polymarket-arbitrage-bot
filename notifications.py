"""
Notification module for Polymarket Arbitrage Bot
Supports: Console, Telegram, Discord Webhook
"""

import asyncio
import time
import logging
import aiohttp

from config import (
    TELEGRAM_TOKEN,
    TELEGRAM_CHAT_ID,
    DISCORD_WEBHOOK_URL,
    NOTIFICATION_METHODS,
    ALERT_RATE_LIMIT_SECONDS,
)

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Send alerts via Telegram Bot API"""

    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.enabled = bool(token and chat_id)
        self.api_url = f"https://api.telegram.org/bot{token}"

    async def send(self, message: str) -> bool:
        if not self.enabled:
            return False
        try:
            url = f"{self.api_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True,
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        logger.debug("Telegram alert sent")
                        return True
                    else:
                        text = await resp.text()
                        logger.warning(f"Telegram send failed ({resp.status}): {text[:200]}")
                        return False
        except Exception as e:
            logger.warning(f"Telegram send error: {e}")
            return False


class DiscordNotifier:
    """Send alerts via Discord Webhook"""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.enabled = bool(webhook_url)

    async def send(self, message: str) -> bool:
        if not self.enabled:
            return False
        try:
            payload = {'content': message}
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload,
                                         timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status in (200, 204):
                        logger.debug("Discord alert sent")
                        return True
                    else:
                        text = await resp.text()
                        logger.warning(f"Discord send failed ({resp.status}): {text[:200]}")
                        return False
        except Exception as e:
            logger.warning(f"Discord send error: {e}")
            return False


class NotificationManager:
    """
    Unified notification dispatcher.
    - Routes alerts to configured channels (console, telegram, discord).
    - Rate-limits: same opportunity key at most once per ALERT_RATE_LIMIT_SECONDS.
    - Falls back to console on channel failure.
    """

    def __init__(self):
        self.telegram = TelegramNotifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
        self.discord = DiscordNotifier(DISCORD_WEBHOOK_URL)
        self._rate_cache: dict = {}  # {opp_key: last_sent_timestamp}
        self._disabled_channels: set = set()

    def _is_rate_limited(self, opp_key: str) -> bool:
        now = time.time()
        if opp_key in self._rate_cache:
            if now - self._rate_cache[opp_key] < ALERT_RATE_LIMIT_SECONDS:
                return True
        self._rate_cache[opp_key] = now
        return False

    @staticmethod
    def format_opportunity(opp) -> str:
        """Format an ArbitrageOpportunity into a notification message"""
        opp_type_label = "NegRisk" if opp.opportunity_type == 'negrisk' else "Single-Condition"
        urgency_emoji = "🔴" if opp.urgency == 'high' else "🟡" if opp.urgency == 'medium' else "🟢"

        lines = [
            f"🚨 *{opp_type_label} Arbitrage Opportunity*",
            f"{urgency_emoji} Urgency: *{opp.urgency.upper()}*",
            f"📊 Market: {opp.market_name}",
            f"💰 Profit: ${opp.expected_profit:.4f} (ROI: {opp.roi * 100:.2f}%)",
            f"💵 Capital Required: ${opp.capital_required:.2f}",
        ]

        if opp.net_profit_after_fees > 0:
            lines.append(f"💲 Net Profit After Fees: ${opp.net_profit_after_fees:.4f}")

        if opp.estimated_slippage > 0:
            lines.append(f"📉 Est. Slippage: {opp.estimated_slippage * 100:.2f}%")
            lines.append(f"📈 Net Profit After Slippage: ${opp.net_profit_after_slippage:.4f}")

        lines.append(f"⚠️ Risk Score: {opp.risk_score:.2f}/1.00")

        if opp.merge_advice:
            lines.append(f"🎯 {opp.merge_advice}")

        # Add key details
        details = opp.details
        if details.get('event_title'):
            lines.append(f"📋 Event: {details['event_title']}")
        if details.get('num_conditions'):
            lines.append(f"🔢 Outcomes: {details['num_conditions']}")
        if details.get('action'):
            lines.append(f"⚡ Action: {details['action']}")

        lines.append(f"🕐 {opp.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        return "\n".join(lines)

    async def send(self, opp) -> None:
        """Send notification for an opportunity across all configured channels"""
        opp_key = f"{opp.market_id}_{opp.opportunity_type}"

        if self._is_rate_limited(opp_key):
            logger.debug(f"Rate limited notification for {opp_key}")
            return

        message = self.format_opportunity(opp)

        for method in NOTIFICATION_METHODS:
            if method in self._disabled_channels:
                continue

            if method == 'console':
                # Console output is handled by AlertManager already
                pass

            elif method == 'telegram':
                if self.telegram.enabled:
                    ok = await self.telegram.send(message)
                    if not ok:
                        logger.warning("Telegram failed, falling back to console only")
                        self._disabled_channels.add('telegram')

            elif method == 'discord':
                if self.discord.enabled:
                    ok = await self.discord.send(message)
                    if not ok:
                        logger.warning("Discord failed, falling back to console only")
                        self._disabled_channels.add('discord')
