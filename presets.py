PRESET_STRATEGIES = [
    {
        "id": "1",
        "name": "RSI Oversold Mean Reversion (Nifty 50)",
        "prompt": "Buy Nifty when RSI(14) crosses below 30 (oversold) and exit when RSI(14) crosses above 70 (overbought).",
        "code": """import pandas as pd
import numpy as np

def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    if len(df) < 15:                                      # Need at least 15 bars for RSI(14)
        return pd.Series(0, index=df.index)

    rsi_period = params.get('rsi_period', 14)             # RSI lookback length
    oversold   = params.get('rsi_oversold', 30)           # Oversold buy threshold
    overbought = params.get('rsi_overbought', 70)         # Overbought sell threshold

    delta = df['close'].diff()                                           # Daily price changes
    gain  = delta.where(delta > 0, 0).rolling(window=rsi_period).mean() # Average gains
    loss  = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean() # Average losses
    rs    = gain / (loss + 1e-10)                                        # Relative Strength
    rsi   = 100 - (100 / (1 + rs))                                       # RSI 0–100

    signals = pd.Series(0, index=df.index)                # Initialize neutral signals
    valid   = rsi.notna()                                  # Only apply where RSI is valid
    signals[valid & (rsi < oversold)]   = 1               # BUY when oversold
    signals[valid & (rsi > overbought)] = -1              # SELL when overbought

    return signals
"""
    },
    {
        "id": "2",
        "name": "EMA Dual Crossover (20 EMA & 50 EMA)",
        "prompt": "Buy when the 20-day EMA crosses above the 50-day EMA. Exit when 20 EMA crosses below 50 EMA.",
        "code": """import pandas as pd
import numpy as np

def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    if len(df) < 51:                                  # Need 51 bars for EMA(50) to stabilize
        return pd.Series(0, index=df.index)

    fast_p = params.get('fast_period', 20)            # Fast EMA period
    slow_p = params.get('slow_period', 50)            # Slow EMA period

    fast_ema = df['close'].ewm(span=fast_p, adjust=False).mean()  # 20-day EMA
    slow_ema = df['close'].ewm(span=slow_p, adjust=False).mean()  # 50-day EMA

    signals = pd.Series(0, index=df.index)            # Default neutral
    valid   = fast_ema.notna() & slow_ema.notna()     # Both EMAs must exist
    signals[valid & (fast_ema > slow_ema)] = 1        # Golden cross → LONG
    signals[valid & (fast_ema < slow_ema)] = -1       # Death cross → EXIT/SHORT

    return signals
"""
    },
    {
        "id": "3",
        "name": "Bollinger Bands Mean Reversion",
        "prompt": "Buy when price touches the lower Bollinger Band (20 period, 2 std dev). Exit when price reaches the upper band.",
        "code": """import pandas as pd
import numpy as np

def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    if len(df) < 21:                                  # Need 21 bars for 20-period SMA
        return pd.Series(0, index=df.index)

    period  = params.get('period', 20)                # SMA lookback
    num_std = params.get('num_std', 2.0)              # Band width multiplier

    sma        = df['close'].rolling(window=period).mean()   # Middle band
    std        = df['close'].rolling(window=period).std()    # Rolling standard deviation
    upper_band = sma + (std * num_std)                       # Upper volatility envelope
    lower_band = sma - (std * num_std)                       # Lower volatility envelope

    signals = pd.Series(0, index=df.index)                   # Signal series initialization
    valid   = sma.notna() & upper_band.notna()               # Bands must be computed
    signals[valid & (df['close'] <= lower_band)] = 1         # BUY at lower band squeeze
    signals[valid & (df['close'] >= upper_band)] = -1        # EXIT at upper band

    return signals
"""
    },
    {
        "id": "4",
        "name": "MACD Signal Line Crossover",
        "prompt": "Buy when MACD line crosses above signal line. Sell when MACD crosses below signal line.",
        "code": """import pandas as pd
import numpy as np

def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    if len(df) < 35:                                          # Minimum bars for MACD(12,26,9)
        return pd.Series(0, index=df.index)

    fast_p   = params.get('macd_fast', 12)                   # Fast EMA period
    slow_p   = params.get('macd_slow', 26)                   # Slow EMA period
    signal_p = params.get('macd_signal', 9)                  # Signal line smoothing period

    ema_fast    = df['close'].ewm(span=fast_p,   adjust=False).mean()  # Fast EMA
    ema_slow    = df['close'].ewm(span=slow_p,   adjust=False).mean()  # Slow EMA
    macd_line   = ema_fast - ema_slow                                   # MACD histogram
    signal_line = macd_line.ewm(span=signal_p, adjust=False).mean()    # Signal smoothing

    prev_macd   = macd_line.shift(1)                          # Previous bar MACD
    prev_signal = signal_line.shift(1)                        # Previous bar signal

    signals = pd.Series(0, index=df.index)
    valid   = macd_line.notna() & signal_line.notna() & prev_macd.notna()
    signals[valid & (macd_line > signal_line) & (prev_macd <= prev_signal)] = 1   # Bullish crossover
    signals[valid & (macd_line < signal_line) & (prev_macd >= prev_signal)] = -1  # Bearish crossover

    return signals
"""
    },
    {
        "id": "5",
        "name": "ATR Supertrend-like Trend Follower",
        "prompt": "Buy when price breaks above the ATR-based upper band. Sell when price breaks below the lower band.",
        "code": """import pandas as pd
import numpy as np

def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    atr_period = params.get('atr_period', 14)                 # ATR lookback window
    multiplier = params.get('atr_mult', 3.0)                  # Band width multiplier

    if len(df) < atr_period + 1:
        return pd.Series(0, index=df.index)

    high_low    = df['high'] - df['low']                                            # TR component 1
    high_close  = (df['high'] - df['close'].shift(1)).abs()                         # TR component 2
    low_close   = (df['low']  - df['close'].shift(1)).abs()                         # TR component 3
    true_range  = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1) # True Range
    atr         = true_range.rolling(window=atr_period).mean()                      # Average True Range

    hl2         = (df['high'] + df['low']) / 2               # Median price
    upper_band  = hl2 + (multiplier * atr)                   # Resistance ceiling
    lower_band  = hl2 - (multiplier * atr)                   # Support floor

    signals = pd.Series(0, index=df.index)
    valid   = atr.notna()
    signals[valid & (df['close'] > upper_band.shift(1))] = 1    # Price breaks resistance → BUY
    signals[valid & (df['close'] < lower_band.shift(1))] = -1   # Price breaks support → SELL

    return signals
"""
    },
    {
        "id": "6",
        "name": "VWAP Mean Reversion",
        "prompt": "Buy when price dips below VWAP with RSI confirmation. Exit when price recovers above VWAP.",
        "code": """import pandas as pd
import numpy as np

def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    rsi_period = params.get('rsi_period', 14)                 # RSI confirmation window
    vwap_band  = params.get('vwap_band', 0.005)               # 0.5% below VWAP for entry

    if len(df) < rsi_period + 1:
        return pd.Series(0, index=df.index)

    # Running VWAP: cumulative(typical_price × volume) / cumulative(volume)
    typical_price = (df['high'] + df['low'] + df['close']) / 3           # Typical price per bar
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum() # Running VWAP

    # RSI for momentum confirmation
    delta = df['close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rsi   = 100 - (100 / (1 + gain / (loss + 1e-10)))                    # RSI 0–100

    signals    = pd.Series(0, index=df.index)
    valid      = vwap.notna() & rsi.notna()
    below_vwap = df['close'] < vwap * (1 - vwap_band)                    # Price dipped below VWAP
    above_vwap = df['close'] > vwap                                       # Price recovered above VWAP
    oversold   = rsi < 40                                                  # RSI confirms weakness

    signals[valid & below_vwap & oversold] = 1   # BUY dip with RSI support
    signals[valid & above_vwap]            = -1  # EXIT when price recovers

    return signals
"""
    },
]
