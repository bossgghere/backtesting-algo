import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Default Market Data Parameters
DEFAULT_SYMBOL = os.getenv("DEFAULT_SYMBOL", "^NSEI")  # Nifty 50 Index
DEFAULT_PERIOD = os.getenv("DEFAULT_PERIOD", "1y")      # 1 year history
DEFAULT_INTERVAL = os.getenv("DEFAULT_INTERVAL", "1d")  # Daily candles

# Illustrative costs only; configure for the actual instrument and dates.
COST_MODEL = {
    "brokerage_per_trade": 20.0,   # Flat fee per filled order
    "slippage_pct": 0.0005,        # 0.05% slippage estimate
    "stt_tax_pct": 0.00025,        # Simplified sell-turnover charge; not a complete tax model
}

# Approved Python modules for AST validation
ALLOWED_MODULES = {
    "pandas", "numpy", "math", "datetime", "ta"
}
