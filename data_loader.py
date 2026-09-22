import os
import pandas as pd
import yfinance as yf

from config import INTERVAL_MAX_PERIOD

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".data_cache")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _period_to_days(period_str: str) -> int:
    """Converts a yfinance period string (e.g. '60d', '1y', '2y') to integer days."""
    period_str = period_str.strip().lower()
    if period_str.endswith("d"):
        return int(period_str[:-1])
    if period_str.endswith("mo"):
        return int(period_str[:-2]) * 30
    if period_str.endswith("y"):
        return int(period_str[:-1]) * 365
    return 365  # fallback


def _clamp_period_for_interval(interval: str, requested_period: str) -> str:
    """
    Returns the effective period to use for a given interval.
    Clamps to yfinance's maximum supported window and warns the user.
    """
    max_period_str = INTERVAL_MAX_PERIOD.get(interval)
    if max_period_str is None:
        return requested_period  # Daily / weekly / monthly have no restriction

    max_days = _period_to_days(max_period_str)
    req_days = _period_to_days(requested_period)

    if req_days > max_days:
        print(
            f"[Warning] Interval '{interval}' supports max {max_period_str}. "
            f"Clamping period from '{requested_period}' → '{max_period_str}'."
        )
        return max_period_str
    return requested_period


# ---------------------------------------------------------------------------
# Symbol Validation
# ---------------------------------------------------------------------------

def validate_symbol(symbol: str) -> tuple:
    """
    Quickly checks whether a symbol resolves on Yahoo Finance.
    Returns (is_valid: bool, message: str).
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info
        # fast_info.last_price is None / 0.0 for unknown tickers
        price = getattr(info, "last_price", None)
        if not price:
            return False, (
                f"Symbol '{symbol}' could not be resolved on Yahoo Finance. "
                "Check the ticker name (e.g. '^NSEI', 'RELIANCE.NS')."
            )
        return True, "OK"
    except Exception as e:
        return False, f"Symbol validation failed for '{symbol}': {e}"


# ---------------------------------------------------------------------------
# Main Data Fetcher
# ---------------------------------------------------------------------------

def get_market_data(
    symbol: str = "^NSEI",
    period: str = "1y",
    interval: str = "1d",
    start_date: str = None,
    end_date: str = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetches real historical OHLCV market data via yfinance.
    Caches results locally as Parquet for fast re-runs.

    Args:
        symbol        : Ticker symbol (e.g. '^NSEI', 'RELIANCE.NS')
        period        : History window ('1y', '6mo', '60d', etc.) — ignored when start_date/end_date set
        interval      : Candle size ('1d', '1h', '15m', '5m', etc.)
        start_date    : ISO date string 'YYYY-MM-DD' (overrides period)
        end_date      : ISO date string 'YYYY-MM-DD' (overrides period)
        force_refresh : Bypass local cache and re-download

    Returns:
        DataFrame with lowercase columns: ['open', 'high', 'low', 'close', 'volume']
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    use_dates = bool(start_date and end_date)
    clean_symbol = symbol.replace("^", "").replace(".", "_")

    if use_dates:
        cache_key = f"{clean_symbol}_{start_date}_{end_date}_{interval}"
    else:
        effective_period = _clamp_period_for_interval(interval, period)
        cache_key = f"{clean_symbol}_{effective_period}_{interval}"

    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.parquet")

    # Serve from cache if available and not forced refresh
    if not force_refresh and os.path.exists(cache_file):
        try:
            df = pd.read_parquet(cache_file)
            if not df.empty:
                return df
        except Exception:
            pass  # Cache corrupted — fall through to re-download

    if use_dates:
        print(f"Fetching market data for '{symbol}' ({start_date} → {end_date}, {interval})...")
    else:
        print(f"Fetching market data for '{symbol}' ({effective_period}, {interval})...")

    # Download with error handling
    try:
        if use_dates:
            df = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
        else:
            df = yf.download(symbol, period=effective_period, interval=interval, progress=False)
    except Exception as e:
        err = str(e).lower()
        if any(k in err for k in ("connection", "timeout", "network", "urlopen")):
            raise ConnectionError(
                f"Network error fetching data for '{symbol}'. "
                f"Check your internet connection and try again.\nDetails: {e}"
            )
        raise RuntimeError(f"Failed to download market data for '{symbol}': {e}")

    if df.empty:
        raise ValueError(
            f"No market data returned for '{symbol}' (period={effective_period}, "
            f"interval={interval}). The symbol may be delisted or the period too short."
        )

    # Flatten multi-index columns (common in newer yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [col.lower() for col in df.columns]

    # Validate required columns exist
    required_cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing expected columns {missing} in data for '{symbol}'. "
            f"Got: {list(df.columns)}"
        )

    df = df[required_cols].copy()
    df.dropna(subset=["close"], inplace=True)
    df.sort_index(inplace=True)

    # Persist to local cache
    try:
        df.to_parquet(cache_file)
    except Exception:
        pass  # Cache write failure is non-fatal

    return df
