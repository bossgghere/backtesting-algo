import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Default Market Data Parameters
DEFAULT_SYMBOL = os.getenv("DEFAULT_SYMBOL", "^NSEI")  # Nifty 50 Index
DEFAULT_PERIOD = os.getenv("DEFAULT_PERIOD", "1y")      # 1 year history
DEFAULT_INTERVAL = os.getenv("DEFAULT_INTERVAL", "1d")  # Daily candles

# Realistic Cost Model Parameters for Indian Equity F&O / Equities
COST_MODEL = {
    "brokerage_per_trade": 20.0,   # Flat ₹20 per trade (Zerodha/Groww standard)
    "slippage_pct": 0.0005,        # 0.05% slippage estimate
    "stt_tax_pct": 0.00025,        # STT tax estimate
}

# Approved Python modules for AST validation
ALLOWED_MODULES = {
    "pandas", "numpy", "math", "datetime", "ta"
}
