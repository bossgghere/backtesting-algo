import os
import pandas as pd
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".data_cache")

def get_market_data(symbol: str = "^NSEI", period: str = "1y", interval: str = "1d", force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetches real historical market data via yfinance and returns a standardized pandas DataFrame.
    Standardized columns: ['open', 'high', 'low', 'close', 'volume']
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    clean_symbol = symbol.replace("^", "").replace(".", "_")
    cache_file = os.path.join(CACHE_DIR, f"{clean_symbol}_{period}_{interval}.parquet")

    if not force_refresh and os.path.exists(cache_file):
        try:
            df = pd.read_parquet(cache_file)
            if not df.empty:
                return df
        except Exception:
            pass

    print(f"Fetching real market data for '{symbol}' ({period}, {interval})...")
    df = yf.download(symbol, period=period, interval=interval, progress=False)

    if df.empty:
        raise ValueError(f"No market data found for symbol '{symbol}'. Please check symbol name or connectivity.")

    # Flatten multi-index columns if present (common in yfinance output)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [col.lower() for col in df.columns]

    # Ensure required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing expected column '{col}' in market data for {symbol}.")

    df = df[required_cols].copy()
    df.dropna(subset=['close'], inplace=True)
    df.sort_index(inplace=True)

    # Save to local cache
    try:
        df.to_parquet(cache_file)
    except Exception:
        pass

    return df
