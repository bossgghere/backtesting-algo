# studio.trade — Universal Algo Trading Platform CLI (1-Day Prototype)

An algorithmic trading platform prototype for Indian F&O / Equities with Natural Language Strategy Authoring, AST Safety Validation, and Strategy-Agnostic Backtesting Engine using **real market data**.

---

## 🚀 Quick Start Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Gemini API Key
Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey) (takes 10 seconds).

Open the `.env` file and set your key:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
DEFAULT_SYMBOL=^NSEI
```

### 3. Run the Terminal Prototype
```bash
python main.py
```
*(or `python cli.py`)*

---

## 🎯 Features Implemented (Matching Client PRD)

1. **Natural Language Strategy Authoring (NL → Code):**
   * Describe any trading idea in plain English (e.g. *"Buy Nifty when RSI 14 drops below 30 and volume > 20-day MA, exit at RSI > 65"*).
   * Gemini LLM generates Python strategy code matching the contract `generate_signals(df, params) -> pd.Series`.

2. **AST Safety & Validation Engine (`validator.py`):**
   * Inspects code before execution.
   * Enforces module allowlists (`pandas`, `numpy`, `math`, `ta`).
   * Blocks dangerous calls (`eval`, `exec`, `open`, `subprocess`, `os`, `sys`).
   * Generates deterministic SHA-256 **Strategy Fingerprints** for deduplication (PRD §7.7).

3. **Real Market Data Ingestion (`data_loader.py`):**
   * Fetches real historical market data for **Nifty 50 (`^NSEI`)**, **Bank Nifty (`^NSEBANK`)**, **Reliance (`RELIANCE.NS`)**, etc. via `yfinance`.
   * Caches data locally for instant re-runs.

4. **Strategy-Agnostic Backtest Simulation (`backtester.py`):**
   * Realistic cost models: brokerage charges (₹20/trade), slippage (0.05%), STT taxes.
   * Quantitative Metrics: Win Rate %, Total Return %, Buy & Hold Return %, Profit Factor, Max Drawdown %, Sharpe Ratio.
   * Detailed Trade Log (Entry/Exit dates, prices, PnL %, Net Return ₹).

5. **Preset Strategies (`presets.py`):**
   * Built-in preset strategies (RSI Mean Reversion, EMA Dual Crossover, Bollinger Bands) to demonstrate immediate real-data backtesting even without API keys!

---

## 📁 File Structure
```
AlgoTrade/
├── .env                # API Keys & Configuration
├── config.py           # Application settings & cost model parameters
├── data_loader.py      # Real market data fetcher & local parquet cache
├── validator.py        # AST safety checker & SHA-256 fingerprinting
├── generator.py        # Gemini API prompt to Python code converter
├── backtester.py       # Strategy-agnostic backtesting simulation engine
├── presets.py          # Pre-built trading strategies
├── cli.py              # Interactive Terminal UI with tables & panels
├── main.py             # Entrypoint script
└── requirements.txt    # Python dependencies
```

## Corrected backtesting behavior

The current engine is a long-only, single-instrument simulator, not a complete
Indian F&O execution model. Index data is an illustrative price series, not a
tradable futures/options contract. The earlier feature list describes the prototype's
intent; it does not establish production accuracy or execution isolation.

- Signals are **actions**: `1` buys when flat, `0` holds the current position,
  and `-1` sells an existing long. Repeated buys do not add to a position;
  selling while flat does not open a short. This matches the existing presets.
- A signal computed after a bar closes executes at the **next bar's open**,
  with adverse slippage. The last bar's signal cannot execute in this dataset.
  Strategy code must itself avoid future data; shifting signals is not enough.
- `quantity` is an optional fixed number of units (not number of lots).
  It must be a multiple of `lot_size`. With no quantity, the engine buys the
  largest affordable whole-lot quantity after reserving the entry fee.
  Unaffordable orders are recorded in `rejected_orders` without charging fees.
- Cash pays for actual units. Closed-trade net PnL includes entry and exit
  brokerage plus a configurable tax on actual sell turnover. These are
  illustrative costs, not an instrument/date-specific statutory tax schedule.
- Final portfolio value includes open holdings at the last close. There is no
  forced liquidation or assumed exit fee. `open_position` reports this exposure,
  and `cash_balance` is shown separately in the returned summary.
- Trade statistics cover closed trades only. Breakeven trades are counted
  separately. Profit factor is infinity for wins without losses, and `None`
  when there are neither wins nor losses. Undefined Sharpe is also `None`.
- Sharpe uses zero risk-free rate and `periods_per_year=252` for daily data.
  Pass an appropriate annualization factor for other bar frequencies.
  Buy-and-hold is a gross close-to-close reference, not a cost-matched benchmark.
- Internal values retain precision; presentation may round them. Results are
  reproducible for identical inputs. Data must be nonempty and chronologically
  ordered, with unique indices, positive finite opens/closes, and aligned signals.

### Offline example (no AI key or network needed)

```python
import pandas as pd
from backtester import run_backtest_simulation

prices = pd.DataFrame({
    'open': [100, 100, 110],
    'close': [100, 105, 110],
}, index=pd.date_range('2024-01-01', periods=3))

result = run_backtest_simulation(
    prices,
    lambda data, params: pd.Series([1, -1, 0], index=data.index),
    initial_capital=1000, quantity=10,
    brokerage_per_trade=0, slippage_pct=0, stt_tax_pct=0,
)
print(result['summary']['final_capital'])  # 1100: buy 10 at 100, sell at 110
```

Run deterministic accounting regression tests:

```bash
python -m unittest -v
```

Still outside this correction: short selling, derivative margin/expiry/settlement,
option-contract datasets, order-book liquidity, realistic partial fills,
corporate-action cash flows, and sandboxed execution of AI-generated code.
Historical data quality and live-market accuracy have not been verified by these tests.

## Swappable historical data for the prototype

The default `DATA_PROVIDER=yahoo` uses yfinance without a data API key. It is
for local exploration; yfinance documents Yahoo data as intended for personal
use. This is not a licensed production feed or expired Indian options archive.
Gemini credentials are separate and are only used for English-to-code generation.

Restart the CLI after changing `.env`. Existing `.env` files need no changes
for Yahoo; DATA_PROVIDER defaults to yahoo. DEFAULT_PERIOD now reaches the loader.
Only DEFAULT_INTERVAL=1d is supported until intraday execution/reporting is added.

To use a supplied daily dataset instead, set:

```env
DATA_PROVIDER=csv
HISTORICAL_CSV_PATH=data/my_prices.csv
HISTORICAL_CSV_SYMBOL=RELIANCE.NS
DEFAULT_SYMBOL=RELIANCE.NS
DEFAULT_INTERVAL=1d
```

Provide one instrument per file with these columns:

```csv
timestamp,open,high,low,close,volume
2024-01-01,100,102,99,101,1000
2024-01-02,101,104,100,103,1200
```

Those rows are format examples, not actual historical prices. CSV mode uses the
whole file, not the rolling DEFAULT_PERIOD. Timestamps must be consistently
formatted, daily and unique. Document the timezone/adjustment policy of supplied
files. Optional symbol columns are checked against the configured symbol.

The loader checks required fields, numeric values, duplicate timestamps, negative
volume, positive prices and OHLC consistency. It does not certify completeness,
exchange calendars, corporate actions, or source accuracy. Zero volume is allowed
because index feeds may report it; avoid volume strategies on such feeds.

Each normalized dataset and its source metadata are saved under
`.data_cache/snapshots/`, identified by a content hash. Yahoo requests fetch anew
rather than silently reuse an old one-year rolling cache. Snapshot saving uses
CSV and does not require parquet dependencies. Select a saved snapshot with the
CSV provider to replay the exact data. Protect any licensed datasets according to
your provider's terms; the cache is already ignored by Git.

When the manager chooses a provider, implement its `fetch(symbol, period, interval)`
method and register it in `data_loader.PROVIDERS`. It must map the provider's
symbols/timestamps/fields into the same daily OHLCV DataFrame. Keep its credentials
in local environment settings. The strategy and engine then remain unchanged for
compatible instruments. A key alone cannot adapt another API. Futures/options
still require engine work for contracts, margin, expiry and settlement.

Run `python -m unittest -v` for engine and data-adapter tests. Tests use synthetic
fixtures/mocks and do not validate a live provider's accuracy or availability.
