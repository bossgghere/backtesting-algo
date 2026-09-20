PRESET_STRATEGIES = [
    {
        "id": "1",
        "name": "RSI Oversold Mean Reversion (Nifty 50)",
        "prompt": "Buy Nifty when RSI(14) crosses below 30 (oversold) and exit when RSI(14) crosses above 70 (overbought).",
        "code": """import pandas as pd
import numpy as np

def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    # 1. Parameter extraction with defaults
    rsi_period = params.get('rsi_period', 14) # RSI lookback length
    oversold = params.get('rsi_oversold', 30) # Oversold buy threshold
    overbought = params.get('rsi_overbought', 70) # Overbought sell threshold

    # 2. RSI technical indicator calculation
    delta = df['close'].diff() # Daily price changes
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean() # Positive price movement
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean() # Negative price movement
    rs = gain / (loss + 1e-10) # Relative Strength calculation
    rsi = 100 - (100 / (1 + rs)) # RSI momentum formula

    # 3. Signal logic generation
    signals = pd.Series(0, index=df.index) # Initialize zero-signal series
    signals[rsi < oversold] = 1 # BUY signal on RSI dip
    signals[rsi > overbought] = -1 # SELL signal on RSI spike

    return signals
"""
    },
    {
        "id": "2",
        "name": "EMA Dual Crossover (20 EMA & 50 EMA)",
        "prompt": "Buy when the 20-day Exponential Moving Average crosses above the 50-day EMA. Exit when 20 EMA crosses below 50 EMA.",
        "code": """import pandas as pd
import numpy as np

def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    # 1. Extract fast and slow EMA periods
    fast_p = params.get('fast_period', 20) # Fast trend period
    slow_p = params.get('slow_period', 50) # Slow trend period

    # 2. Compute Exponential Moving Averages
    fast_ema = df['close'].ewm(span=fast_p, adjust=False).mean() # 20-day EMA
    slow_ema = df['close'].ewm(span=slow_p, adjust=False).mean() # 50-day EMA

    # 3. Detect trend crossover signals
    signals = pd.Series(0, index=df.index) # Default neutral position
    signals[fast_ema > slow_ema] = 1 # Golden cross - LONG position
    signals[fast_ema < slow_ema] = -1 # Death cross - EXIT existing long position

    return signals
"""
    },
    {
        "id": "3",
        "name": "Bollinger Bands Mean Reversion",
        "prompt": "Buy when price touches or drops below the lower Bollinger Band (20 period, 2 std dev). Exit when price reaches upper band.",
        "code": """import pandas as pd
import numpy as np

def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    # 1. Extract band parameters
    period = params.get('period', 20) # SMA lookback
    num_std = params.get('num_std', 2.0) # Standard deviation multiplier

    # 2. Calculate Bollinger Bands
    sma = df['close'].rolling(window=period).mean() # Middle band
    std = df['close'].rolling(window=period).std() # Standard deviation
    upper_band = sma + (std * num_std) # Upper volatility envelope
    lower_band = sma - (std * num_std) # Lower volatility envelope

    # 3. Generate mean reversion signals
    signals = pd.Series(0, index=df.index) # Signal series initialization
    signals[df['close'] <= lower_band] = 1 # BUY signal at lower band expansion
    signals[df['close'] >= upper_band] = -1 # EXIT signal at upper band

    return signals
"""
    }
]
