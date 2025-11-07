#!/usr/bin/env python3
"""
Quick test to verify Polymarket API connectivity
"""

import asyncio
import aiohttp
import sys


async def test_connection():
    """Test if we can reach Polymarket API"""

    print("🔍 Testing Polymarket API connection...")
    print("-" * 60)

    url = "https://clob.polymarket.com/markets"
    params = {'limit': 5, 'active': 'true'}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    markets = data if isinstance(data, list) else []

                    print(f"✅ SUCCESS! Connected to Polymarket API")
                    print(f"✅ Fetched {len(markets)} active markets")
                    print("-" * 60)

                    if markets:
                        print("\n📊 Sample markets found:")
                        for i, market in enumerate(markets[:3], 1):
                            question = market.get('question', 'Unknown')[:60]
                            tokens = len(market.get('tokens', []))
                            print(f"  {i}. {question}")
                            print(f"     Outcomes: {tokens}")

                    print("\n" + "="*60)
                    print("🎉 API is working! Ready to run the bot.")
                    print("="*60)
                    print("\nRun the bot with:")
                    print("  python prediction_market_arbitrage.py")
                    print("  or")
                    print("  ./start_bot.sh")

                    return True
                else:
                    print(f"❌ API returned status {resp.status}")
                    return False

    except asyncio.TimeoutError:
        print("❌ Connection timeout - check your internet connection")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_connection())
    sys.exit(0 if success else 1)
