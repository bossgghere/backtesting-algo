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
