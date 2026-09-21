import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Default Market Data Parameters
DEFAULT_SYMBOL   = os.getenv("DEFAULT_SYMBOL", "^NSEI")  # Nifty 50 Index
DEFAULT_PERIOD   = os.getenv("DEFAULT_PERIOD", "1y")      # 1 year history
DEFAULT_INTERVAL = os.getenv("DEFAULT_INTERVAL", "1d")    # Daily candles

# Realistic Cost Model Parameters for Indian Equity / F&O
COST_MODEL = {
    "brokerage_per_trade": 20.0,   # Flat ₹20 per trade (Zerodha/Groww standard)
    "slippage_pct":        0.0005, # 0.05% slippage estimate
    "stt_tax_pct":         0.00025,# STT tax on trade value
}

# Risk & Position Sizing Defaults
RISK_FREE_RATE        = 0.065  # 6.5% annualized (Indian repo rate)
DEFAULT_POSITION_SIZE = 1.0    # 1.0 = 100% of cash deployed per trade (all-in)

# yfinance maximum supported periods per interval
# Used to clamp user requests to valid ranges and avoid empty downloads
INTERVAL_MAX_PERIOD = {
    "1m":  "7d",
    "2m":  "60d",
    "5m":  "60d",
    "15m": "60d",
    "30m": "60d",
    "60m": "730d",
    "1h":  "730d",
    "90m": "60d",
    "1d":  None,   # No restriction
    "1wk": None,
    "1mo": None,
}

# Approved Python modules for AST strategy validation
# Single source of truth — imported by validator.py
ALLOWED_MODULES = {"pandas", "numpy", "math", "datetime", "ta"}
