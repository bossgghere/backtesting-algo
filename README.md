# studio.trade

An AI-powered algo trading backtesting platform. Describe a trading strategy in plain English — the platform converts it to executable Python code using Gemini, runs a full backtest on real Nifty 50 data, and shows you detailed results with interactive parameter sliders.

![status](https://img.shields.io/badge/status-active-brightgreen) ![Python](https://img.shields.io/badge/python-3.10+-blue) ![React](https://img.shields.io/badge/react-18-61dafb)

---

## How it works

1. **Describe** your strategy in plain English (e.g. *"Buy when RSI drops below 30, sell when it crosses above 70"*)
2. **Gemini AI** converts your idea into a safe, executable Python strategy
3. **Tune** the extracted parameters via sliders before running
4. **Backtest** runs on 1 year of real Nifty 50 OHLCV data via yfinance
5. **Results** show equity curve, trade log, Sharpe ratio, drawdown, win rate, and more

---

## Features

- Natural language → Python strategy via Google Gemini
- Real market data from Yahoo Finance (Nifty 50, daily OHLCV)
- Interactive parameter sliders — tune RSI period, EMA window, etc. without re-generating
- Full backtest engine with realistic costs (brokerage, slippage, STT)
- Metrics: Total Return, Sharpe Ratio, Max Drawdown, Win Rate, Profit Factor, Avg Hold
- Strategy vs Buy & Hold comparison
- Trade-by-trade log with entry/exit dates, PnL, and exit reason
- Equity curve chart
- Generated code viewer
- 6 built-in preset strategies (RSI, EMA crossover, Bollinger Bands, MACD, ATR, VWAP)
- AST-based code safety validation before exec()

---

## Tech stack

| Layer | Tech |
|---|---|
| Frontend | React 18, Vite, Recharts |
| Backend | FastAPI, Python 3.10+ |
| AI | Google Gemini API |
| Market Data | yfinance |
| Indicators | `ta` library |

---

## Project structure

```
├── api.py              # FastAPI backend (generate, run, presets endpoints)
├── backtester.py       # Bar-by-bar backtest engine
├── generator.py        # Gemini NL → Python code generation
├── data_loader.py      # yfinance data fetching + validation
├── validator.py        # AST safety check before exec()
├── presets.py          # 6 built-in strategies
├── config.py           # Constants (costs, risk-free rate, allowed modules)
├── tests.py            # 62 unit + integration tests
└── frontend/
    └── src/
        ├── App.jsx     # 3-screen UI (Entry → Params → Results)
        ├── App.css     # Dark theme styles
        └── components/
            └── EquityChart.jsx
```

---

## Getting started

### Prerequisites

- Python 3.10+
- Node.js 18+
- A free [Google Gemini API key](https://aistudio.google.com/app/apikey)

### 1. Clone the repo

```bash
git clone https://github.com/bossgghere/backtesting-algo.git
cd backtesting-algo
```

### 2. Set up Python environment

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
DEFAULT_SYMBOL=^NSEI
DEFAULT_PERIOD=1y
DEFAULT_INTERVAL=1d
```

### 4. Start the backend

```bash
uvicorn api:app --reload --port 8000
```

### 5. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/generate` | NL prompt → Python code + param sliders |
| `POST` | `/api/run` | Run backtest with code + params |
| `GET` | `/api/presets` | List built-in strategies |

---

## Example strategies to try

```
Buy when RSI drops below 30, sell when RSI goes above 70
```
```
Buy when the 10-day EMA crosses above the 50-day EMA, sell when it crosses below
```
```
Buy when MACD line crosses above the signal line, sell when it crosses below
```
```
Buy when price closes below the lower Bollinger Band, sell when it closes above the upper band
```

---

## Backtest assumptions

- Starting capital: ₹1,00,000
- Brokerage: ₹20 per trade
- Slippage: 0.05%
- STT: 0.1% on sell side
- Risk-free rate: 6.5% (for Sharpe ratio)
- Universe: Nifty 50 index (^NSEI), 1 year daily data

---

## Running tests

```bash
python -m pytest tests.py -v
```

62 tests covering config, data loading, validation, backtester, presets, generator, and API endpoints.

---

## Security

All AI-generated code is validated through an AST safety check before execution. Imports are restricted to `pandas`, `numpy`, `math`, and `ta`. System calls, file access, and network requests are blocked.

---

## License

MIT
