# 🚀 Quick Start Guide - Prediction Market Arbitrage Bot

Get up and running in 5 minutes!

---

## ✅ Step 1: Check Requirements

You need:
- ✅ Python 3.8 or higher
- ✅ Internet connection
- ✅ 5 minutes

Check Python version:
```bash
python3 --version
```

If you don't have Python, download from: https://www.python.org/downloads/

---

## 📦 Step 2: Install Dependencies

Run this command in the project folder:

```bash
pip install -r requirements.txt
```

This installs:
- `aiohttp` - For API requests
- `websockets` - For real-time data
- `pandas` - For data analysis
- `numpy` - For calculations

**All are free and open-source.**

---

## 🧪 Step 3: Test Connection

Verify the bot can connect to Polymarket:

```bash
python test_connection.py
```

You should see:
```
✅ SUCCESS! Connected to Polymarket API
✅ Fetched 5 active markets
🎉 API is working! Ready to run the bot.
```

If you see errors:
- Check your internet connection
- Check if your firewall blocks Python
- Try running as administrator/sudo

---

## 🎯 Step 4: Run the Bot

### Option A: Simple Start
```bash
python prediction_market_arbitrage.py
```

### Option B: Use Launcher Script
```bash
./start_bot.sh
```

### Option C: Background Mode (Linux/Mac)
```bash
nohup python prediction_market_arbitrage.py > output.log 2>&1 &
```

---

## 📊 Step 5: Understand the Output

### Normal Scan Output
```
[1/50] Analyzing: Will Trump win 2024 election?
[2/50] Analyzing: Bitcoin $100K by end of 2024?
```

This means: Bot is scanning markets, no opportunities yet.

### 🔴 HIGH PRIORITY Alert
```
================================================================================
🔴 ARBITRAGE OPPORTUNITY DETECTED - NEGRISK
================================================================================
Market: Democratic VP Nominee 2024?
Expected Profit: $127.50
ROI: 4.73%
Capital Required: $2,695.00
Risk Score: 0.15/1.00
Urgency: HIGH

Details:
  num_conditions: 5
  prob_sum: 0.9527
  deviation: 0.0473
  action: buy_all
```

**What this means:**
- 💰 Potential profit: $127.50
- 📈 Return: 4.73% on $2,695
- ⚠️ Risk: Low (0.15/1.00)
- 🎯 Action: Buy all 5 outcomes

### 🟡 MEDIUM PRIORITY Alert
Lower profit/ROI but still worth checking.

### 🟢 LOW PRIORITY Alert
Small profit - only for high-frequency traders.

---

## ⚙️ Step 6: Customize Settings

Edit `config.py` to change bot behavior:

### Scan Less Frequently
```python
SCAN_INTERVAL = 300  # 5 minutes instead of 1 minute
```

### Monitor More Markets
```python
TOP_MARKETS = 100  # Up from 50
```

### Higher Profit Threshold
```python
MIN_ALERT_PROFIT = 50.0  # Only alert if profit > $50
```

### Only High Urgency Alerts
```python
ALERT_ONLY_HIGH_URGENCY = True
```

---

## 📱 What to Do When You Get an Alert

### ✅ DO:
1. **Verify manually** - Go to Polymarket.com and check the market
2. **Check the math** - Verify prices sum correctly
3. **Check liquidity** - Ensure enough volume to execute
4. **Start small** - Test with $100-500 first
5. **Set limits** - Know your max loss tolerance

### ❌ DON'T:
1. ❌ Blindly follow every alert
2. ❌ Risk more than you can afford to lose
3. ❌ Ignore risk scores
4. ❌ Trade on markets resolving in <24 hours (oracle risk)
5. ❌ Forget about gas fees (Polygon)

---

## 🎓 Understanding the Strategies

### Strategy 1: Single-Condition (Simplest)
```
Market: "Will it rain tomorrow?"
YES: $0.53
NO: $0.42
Sum: $0.95 ← Should be $1.00!

Action: Buy YES ($0.53) + NO ($0.42) = Spend $0.95
Result: One will pay $1.00
Profit: $0.05 (5.3% return)
```

### Strategy 2: NegRisk (Most Profitable - 29× More Efficient!)
```
Market: "Senate Majority?"
Democrat: 47%
Republican: 46%
Independent: 2%
Tie: 3%
Sum: 98% ← Should be 100%!

Action: Buy all 4 outcomes for $0.98
Result: One will pay $1.00
Profit: $0.02 per dollar

Why 29× better?
- More complexity = less competition
- Higher liquidity fragmentation
- Retail ignores tail outcomes
```

### Strategy 3: Whale Tracking
```
Detect: Large trader buys $15,000 of "YES"
Signal: Strong buying pressure
Action: Consider following (price likely to rise)
Research: 61-68% prediction accuracy
```

---

## 🔧 Troubleshooting

### "No opportunities found"
**Normal!** Arbitrage opportunities are rare. The bot scans continuously.

**What to expect:**
- 1-5 opportunities per hour (busy periods)
- 0-2 opportunities per hour (quiet periods)
- More during major events (elections, debates)

### "Connection failed"
1. Check internet connection
2. Try: `ping clob.polymarket.com`
3. Check firewall settings
4. Increase timeout in config.py

### "Too many low-quality alerts"
Edit `config.py`:
```python
MIN_ALERT_PROFIT = 50.0  # Increase threshold
ALERT_ONLY_HIGH_URGENCY = True
```

### Bot runs but shows no markets
Polymarket API might be down. Check: https://polymarket.com

---

## 📈 Expected Results (Realistic)

Based on the research ($39.59M extracted over 12 months):

### Conservative Approach
- **Capital:** $1,000
- **Time:** 1-2 hours/day
- **Trades:** 2-5 per day
- **Expected monthly return:** 5-10% ($50-100)
- **Risk:** Low-Medium

### Aggressive Approach
- **Capital:** $10,000
- **Time:** 4-6 hours/day or automation
- **Trades:** 10-15 per day
- **Expected monthly return:** 12-20% ($1,200-2,000)
- **Risk:** Medium-High

### Top Performer (Research)
- **Capital:** $100,000+
- **Time:** Full-time + automation
- **Trades:** 11+ per day systematically
- **Achieved:** $2.01M in 12 months
- **Risk:** Professional risk management

**Your results will vary!** Start small, learn the patterns, scale gradually.

---

## 🛡️ Risk Management Rules

### Rule 1: Position Sizing
Never risk more than 5% of capital per opportunity.

Example: $10,000 capital = max $500 per trade

### Rule 2: Time to Resolution
- **>7 days:** Safe
- **3-7 days:** Moderate risk
- **<3 days:** High risk (oracle manipulation)
- **<24 hours:** AVOID

### Rule 3: Market Type
- **Objective criteria:** Safer
  - Example: "Official vote count shows..."
- **Subjective criteria:** Riskier
  - Example: "Who won the debate?"

### Rule 4: Diversification
Spread capital across:
- 5-10 different markets
- 2-3 different strategies
- Different resolution dates

### Rule 5: Stop Loss
If you lose 10% of capital, pause and review strategy.

---

## 📊 Tracking Your Performance

Create a simple spreadsheet:

| Date | Market | Strategy | Capital | Profit | ROI | Notes |
|------|--------|----------|---------|--------|-----|-------|
| 2025-01-11 | Election | NegRisk | $500 | $23 | 4.6% | ✅ Good |
| 2025-01-11 | Sports | Single | $200 | $8 | 4.0% | ✅ Good |
| 2025-01-12 | Politics | Whale | $1000 | -$15 | -1.5% | ❌ Wrong call |

Track:
- Win rate (target: >60%)
- Average ROI per trade
- Total profit/loss
- Best performing strategies

---

## 🎯 7-Day Action Plan

### Day 1-2: Learning Mode
- ✅ Run bot, observe alerts
- ✅ Don't trade yet
- ✅ Read the research articles
- ✅ Verify 5-10 opportunities manually on Polymarket

### Day 3-4: Paper Trading
- ✅ Record trades in spreadsheet
- ✅ Track hypothetical performance
- ✅ Learn which alerts are most reliable

### Day 5-7: Small Capital
- ✅ Start with $100-500
- ✅ Execute 3-5 trades
- ✅ Focus on single-condition (simplest)
- ✅ Learn the execution process

### Week 2+: Scale Gradually
- ✅ Increase capital if profitable
- ✅ Add NegRisk strategy (29× efficiency)
- ✅ Target 10+ trades/day
- ✅ Implement systematic approach

---

## 🚨 Critical Warnings

### ⚠️ WARNING 1: Oracle Risk
**March 2025 Case:** $7M market manipulated via governance vote.
**Your protection:** Exit 24-48 hours before resolution.

### ⚠️ WARNING 2: Market Compression
**ICE $2B investment (Oct 2025)** = Institutional competition coming.
**Timeline:** 12-18 months until spreads compress.
**Action:** Extract value NOW, not later.

### ⚠️ WARNING 3: Regulatory
**Massachusetts sued Kalshi (Sept 2025)** for sports betting.
**Your protection:** Understand laws in your jurisdiction.

### ⚠️ WARNING 4: Smart Contracts
**Polygon bridge risks** - Keep only necessary funds on-chain.
**Your protection:** Daily withdrawals to hardware wallet.

---

## ✅ Checklist Before First Trade

- [ ] Bot running successfully for 24+ hours
- [ ] Verified 5+ opportunities manually
- [ ] Read risk management section
- [ ] Started with capital you can afford to lose
- [ ] Set up hardware wallet (Ledger/Trezor)
- [ ] Tested with small amount ($100-500)
- [ ] Spreadsheet ready for tracking
- [ ] Understand Polygon gas fees
- [ ] Know how to exit positions
- [ ] Have stop-loss plan

---

## 🔗 Resources

### Official APIs
- **Polymarket CLOB:** https://docs.polymarket.com/
- **Market Data:** https://clob.polymarket.com/markets

### Analytics
- **Dune Dashboard:** https://dune.com/genejp999/polymarket-leaderboard
- **Polywhaler:** https://www.polywhaler.com

### Research
- **IMDEA Paper:** Unravelling the Probabilistic Forest
- **Your 15 articles:** Comprehensive strategy guide

---

## 🆘 Getting Help

### Check Logs First
```bash
tail -f arbitrage_bot.log
```

### Common Issues
1. **No opportunities:** Normal - keep scanning
2. **Connection errors:** Check internet/firewall
3. **Low quality alerts:** Adjust config.py thresholds
4. **False positives:** Verify manually before trading

---

## 🎉 Success Tips from Research

### From the $2M Top Performer:
1. **Frequency over size:** 11 trades/day at $496 average
2. **Systematic approach:** Bot-like consistency
3. **NegRisk focus:** 29× capital efficiency
4. **Risk management:** Hold to resolution (confident in objective criteria)

### From the $40M Total Extraction:
1. **NegRisk dominated:** 73% of profits ($29M)
2. **Simple wins:** Single-condition still valuable ($10.6M)
3. **Avoid complexity:** Combinatorial strategies failed (62% failure rate)
4. **Time matters:** Peak opportunities during events

---

## 🚀 Ready to Start?

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test connection
python test_connection.py

# 3. Run bot
python prediction_market_arbitrage.py

# 4. Watch for alerts and learn!
```

---

## 📞 Next Steps

1. ✅ Run bot for 24 hours
2. ✅ Observe 10+ opportunities
3. ✅ Verify 3-5 manually
4. ✅ Execute first small trade
5. ✅ Scale based on results

**Remember:** This bot DETECTS opportunities. YOU make trading decisions.

**Start small. Learn fast. Scale gradually.**

---

*Good luck! The window is compressing - deploy capital aggressively or accept opportunity closure.*

*- Based on $39.59M arbitrage research (IMDEA Networks, 2025)*
