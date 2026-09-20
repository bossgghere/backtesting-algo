"""Replaceable daily OHLCV sources for the prototype (no credentials in code)."""
from pathlib import Path
from typing import Protocol
import hashlib
import json
import os
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
COLUMNS = ['open', 'high', 'low', 'close', 'volume']

class HistoricalProvider(Protocol):
    def fetch(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Return one instrument's OHLCV indexed by timestamp."""
        ...

class YahooProvider:
    def fetch(self, symbol, period, interval):
        import yfinance as yf
        # Explicit adjustment policy; do not depend on changing library defaults.
        data = yf.download(symbol, period=period, interval=interval,
                           auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            if len(data.columns.get_level_values(1).unique()) != 1:
                raise ValueError('Expected one Yahoo instrument.')
            data.columns = data.columns.get_level_values(0)
        return data

class CsvProvider:
    def fetch(self, symbol, period, interval):
        filename = os.getenv('HISTORICAL_CSV_PATH', '').strip()
        expected = os.getenv('HISTORICAL_CSV_SYMBOL', '').strip()
        if not filename or not expected:
            raise ValueError('Set HISTORICAL_CSV_PATH and HISTORICAL_CSV_SYMBOL in .env.')
        if symbol != expected:
            raise ValueError(f'CSV is configured for {expected}, not {symbol}. Change the instrument or CSV settings.')
        path = Path(filename)
        if not path.is_absolute():
            path = ROOT / path
        data = pd.read_csv(path)
        data.columns = data.columns.str.strip().str.lower()
        if 'timestamp' not in data:
            raise ValueError('CSV must include a timestamp column.')
        if 'symbol' in data and not data['symbol'].eq(symbol).all():
            raise ValueError('CSV contains a different instrument; provide one instrument per file.')
        data.index = pd.to_datetime(data.pop('timestamp'), errors='raise')
        # CSV is an explicit historical snapshot: use its whole date range.
        return data

PROVIDERS = {'yahoo': YahooProvider, 'csv': CsvProvider}

def validate_market_data(frame):
    data = frame.copy()
    if not isinstance(data.columns, pd.MultiIndex):
        data.columns = [str(c).strip().lower() for c in data.columns]
    if data.empty or not set(COLUMNS).issubset(data.columns) or data.columns.has_duplicates:
        raise ValueError('Expected nonempty data with unique open, high, low, close, volume columns.')
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError('Provider must return a DatetimeIndex.')
    if data.index.has_duplicates or data.index.hasnans:
        raise ValueError('Duplicate or missing timestamps in historical data.')
    # This engine/report currently assumes daily bars. Reject intraday CSVs.
    if data.index.normalize().has_duplicates:
        raise ValueError('Prototype supports one daily bar per date; intraday support is not implemented.')
    data = data.sort_index()[COLUMNS].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError('Missing or non-finite values in historical data; inspect the source.')
    if (data[['open','high','low','close']] <= 0).any().any() or (data.volume < 0).any():
        raise ValueError('Prices must be positive and volume nonnegative.')
    if ((data.high < data[['open','close','low']].max(axis=1)) |
        (data.low > data[['open','close','high']].min(axis=1))).any():
        raise ValueError('Invalid OHLC range: low <= open/close <= high is required.')
    return data

def get_market_data(symbol='^NSEI', period='1y', interval='1d', force_refresh=False, *, provider=None):
    """Normalize a selected source; engine remains independent of API format.

    Always fetch/read anew: old rolling-period caches are not silently reused.
    force_refresh is retained for backward compatibility. Returned attrs identify
    source, request and exact normalized snapshot. Missing trading sessions require
    exchange-calendar checks beyond this basic validation.
    """
    if interval != '1d':
        raise ValueError('This prototype currently supports DEFAULT_INTERVAL=1d only.')
    name = (provider or os.getenv('DATA_PROVIDER', 'yahoo')).strip().lower()
    if name not in PROVIDERS:
        raise ValueError(f'Unknown DATA_PROVIDER {name!r}; available: {", ".join(PROVIDERS)}. A new API requires an adapter.')
    print(f'Loading {name} data for {symbol} (daily; requested period {period})...')
    data = validate_market_data(PROVIDERS[name]().fetch(symbol, period, interval))
    metadata = {'provider': name, 'symbol': symbol, 'interval': interval,
                'period': period if name != 'csv' else 'full CSV',
                'first_timestamp': str(data.index[0]), 'last_timestamp': str(data.index[-1]),
                'rows': len(data), 'adjustment': 'Yahoo auto_adjust=True' if name == 'yahoo' else 'as supplied in CSV'}
    serialized = data.to_csv(index_label='timestamp')
    digest = hashlib.sha256((json.dumps(metadata, sort_keys=True)+serialized).encode()).hexdigest()
    metadata['snapshot_id'] = digest
    folder = ROOT / '.data_cache' / 'snapshots'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f'{digest}.csv').write_text(serialized, encoding='utf-8')
    (folder / f'{digest}.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    data.attrs.update(metadata)
    print(f'Data: {metadata["first_timestamp"]} to {metadata["last_timestamp"]}; {len(data)} rows; snapshot {digest[:12]}')
    if name == 'yahoo':
        print('Yahoo prototype data: not historical futures/options contracts. Prices are adjusted.')
    return data
