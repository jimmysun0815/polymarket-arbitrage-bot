#!/bin/bash
# Quick launcher for Prediction Market Arbitrage Bot

echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║           PREDICTION MARKET ARBITRAGE BOT - LAUNCHER                      ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "🔍 Checking dependencies..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+<br/>    exit 1
fi

# Check pip packages
pip3 install -q -r requirements.txt

echo "✅ Dependencies OK"
echo ""
echo "🚀 Starting bot..."
echo "   Press Ctrl+C to stop"
echo ""

python3 prediction_market_arbitrage.py
