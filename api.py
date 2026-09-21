"""
studio.trade — FastAPI backend
"""
import re
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

from config import COST_MODEL, RISK_FREE_RATE, DEFAULT_POSITION_SIZE
from data_loader import get_market_data
from validator import validate_strategy_code, get_strategy_fingerprint
from generator import generate_strategy_from_nl
from backtester import run_backtest_simulation
from presets import PRESET_STRATEGIES

app = FastAPI(title="studio.trade API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Parameter config: known params → slider metadata ──────────────────────
PARAM_CONFIG = {
    # RSI
    "rsi_period":       {"label": "RSI Period",       "min": 5,   "max": 30,  "step": 1,   "type": "int"},
    "rsi_window":       {"label": "RSI Period",       "min": 5,   "max": 30,  "step": 1,   "type": "int"},
    "period":           {"label": "Period",           "min": 5,   "max": 50,  "step": 1,   "type": "int"},
    "window":           {"label": "Window",           "min": 5,   "max": 50,  "step": 1,   "type": "int"},
    "rsi_oversold":     {"label": "Oversold Level",   "min": 20,  "max": 45,  "step": 1,   "type": "int"},
    "oversold":         {"label": "Oversold Level",   "min": 20,  "max": 45,  "step": 1,   "type": "int"},
    "oversold_level":   {"label": "Oversold Level",   "min": 20,  "max": 45,  "step": 1,   "type": "int"},
    "rsi_overbought":   {"label": "Overbought Level", "min": 55,  "max": 85,  "step": 1,   "type": "int"},
    "overbought":       {"label": "Overbought Level", "min": 55,  "max": 85,  "step": 1,   "type": "int"},
    "overbought_level": {"label": "Overbought Level", "min": 55,  "max": 85,  "step": 1,   "type": "int"},
    # EMA / MA
    "fast_period":      {"label": "Fast EMA",         "min": 5,   "max": 50,  "step": 1,   "type": "int"},
    "fast_window":      {"label": "Fast EMA",         "min": 5,   "max": 50,  "step": 1,   "type": "int"},
    "slow_period":      {"label": "Slow EMA",         "min": 10,  "max": 200, "step": 5,   "type": "int"},
    "slow_window":      {"label": "Slow EMA",         "min": 10,  "max": 200, "step": 5,   "type": "int"},
    "short_window":     {"label": "Short Window",     "min": 5,   "max": 50,  "step": 1,   "type": "int"},
    "long_window":      {"label": "Long Window",      "min": 10,  "max": 200, "step": 5,   "type": "int"},
    "short_period":     {"label": "Short Period",     "min": 5,   "max": 50,  "step": 1,   "type": "int"},
    "long_period":      {"label": "Long Period",      "min": 10,  "max": 200, "step": 5,   "type": "int"},
    # Bollinger Bands
    "num_std":          {"label": "Band Width (σ)",   "min": 1.0, "max": 3.0, "step": 0.1, "type": "float"},
    "std_dev":          {"label": "Band Width (σ)",   "min": 1.0, "max": 3.0, "step": 0.1, "type": "float"},
    "num_std_dev":      {"label": "Band Width (σ)",   "min": 1.0, "max": 3.0, "step": 0.1, "type": "float"},
    "bb_window":        {"label": "BB Period",        "min": 5,   "max": 50,  "step": 1,   "type": "int"},
    "bb_period":        {"label": "BB Period",        "min": 5,   "max": 50,  "step": 1,   "type": "int"},
    # MACD
    "macd_fast":        {"label": "MACD Fast",        "min": 5,   "max": 20,  "step": 1,   "type": "int"},
    "macd_slow":        {"label": "MACD Slow",        "min": 15,  "max": 50,  "step": 1,   "type": "int"},
    "macd_signal":      {"label": "Signal Line",      "min": 3,   "max": 15,  "step": 1,   "type": "int"},
    "fast_period_macd": {"label": "MACD Fast",        "min": 5,   "max": 20,  "step": 1,   "type": "int"},
    "slow_period_macd": {"label": "MACD Slow",        "min": 15,  "max": 50,  "step": 1,   "type": "int"},
    "signal_period":    {"label": "Signal Line",      "min": 3,   "max": 15,  "step": 1,   "type": "int"},
    "signal_window":    {"label": "Signal Line",      "min": 3,   "max": 15,  "step": 1,   "type": "int"},
    # ATR / Volatility
    "atr_period":       {"label": "ATR Period",       "min": 5,   "max": 30,  "step": 1,   "type": "int"},
    "atr_window":       {"label": "ATR Period",       "min": 5,   "max": 30,  "step": 1,   "type": "int"},
    "atr_mult":         {"label": "ATR Multiplier",   "min": 1.0, "max": 6.0, "step": 0.5, "type": "float"},
    "atr_multiplier":   {"label": "ATR Multiplier",   "min": 1.0, "max": 6.0, "step": 0.5, "type": "float"},
    # VWAP
    "vwap_band":        {"label": "VWAP Band %",      "min": 0.1, "max": 2.0, "step": 0.1, "type": "float"},
}


def extract_params(code_str: str) -> list:
    """
    Scans generated code for params.get('name', default) calls.
    Returns up to 2 known params with slider metadata.
    """
    pattern = r"params\.get\(['\"](\w+)['\"],\s*([0-9.]+)\)"
    matches = re.findall(pattern, code_str)

    result = []
    seen = set()
    for name, default_str in matches:
        if name not in PARAM_CONFIG or name in seen:
            continue
        seen.add(name)
        try:
            raw = float(default_str)
            default = int(raw) if raw == int(raw) and PARAM_CONFIG[name]["type"] == "int" else raw
        except ValueError:
            continue

        entry = PARAM_CONFIG[name].copy()
        entry["name"] = name
        entry["default"] = default
        result.append(entry)

    return result[:2]  # max 2 sliders


def _run_backtest_from_code(code_str: str, params: dict, symbol: str, period: str, interval: str) -> dict:
    """Shared helper: compile code, load data, run backtest, serialize output."""
    is_safe, msg = validate_strategy_code(code_str)
    if not is_safe:
        raise HTTPException(status_code=400, detail=f"Strategy failed safety check: {msg}")

    fingerprint = get_strategy_fingerprint(code_str)

    exec_scope = {}
    try:
        exec(code_str, exec_scope)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Code compilation error: {e}")

    fn = exec_scope.get("generate_signals")
    if not fn:
        raise HTTPException(status_code=400, detail="Code must define generate_signals(df, params).")

    try:
        df = get_market_data(symbol=symbol, period=period, interval=interval)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    results = run_backtest_simulation(
        df=df,
        generate_signals_fn=fn,
        params=params,
        initial_capital=100000.0,
        brokerage_per_trade=COST_MODEL["brokerage_per_trade"],
        slippage_pct=COST_MODEL["slippage_pct"],
        stt_tax_pct=COST_MODEL["stt_tax_pct"],
        risk_free_rate=RISK_FREE_RATE,
    )

    equity_curve = [
        {"date": str(idx)[:10], "equity": round(float(r["equity"]), 2), "close": round(float(r["close"]), 2)}
        for idx, r in results["equity_curve"].iterrows()
    ]

    return {
        "summary":      results["summary"],
        "trades":       results["trades"],
        "equity_curve": equity_curve,
        "code":         code_str,
        "fingerprint":  fingerprint,
        "symbol":       symbol,
    }


# ── Request models ─────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str

class RunRequest(BaseModel):
    code: str
    params: Dict[str, Any] = {}
    symbol: str = "^NSEI"
    period: str = "1y"
    interval: str = "1d"


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/api/generate")
def generate_strategy(req: GenerateRequest):
    """
    Step 1: Convert NL prompt → Python code + extracted param sliders.
    Retries up to 3 times if Gemini returns invalid/unsafe code.
    """
    if len(req.prompt.strip()) < 10:
        raise HTTPException(status_code=400, detail="Prompt must be at least 10 characters.")

    last_error = None
    for attempt in range(3):
        try:
            code_str = generate_strategy_from_nl(req.prompt.strip())
        except ConnectionError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        is_safe, msg = validate_strategy_code(code_str)
        if is_safe:
            return {
                "code":   code_str,
                "params": extract_params(code_str),
            }

        last_error = msg
        print(f"  [Generate retry {attempt + 1}/3] Validation failed: {msg[:120]}")

    raise HTTPException(status_code=500, detail="Strategy generation failed after 3 attempts. Please try rephrasing your strategy.")


@app.post("/api/run")
def run_strategy(req: RunRequest):
    """
    Step 2: Run backtest with provided code + user-adjusted params.
    Can be called multiple times with different param values (no Gemini call).
    """
    try:
        return _run_backtest_from_code(req.code, req.params, req.symbol, req.period, req.interval)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/presets")
def get_presets():
    return [{"id": p["id"], "name": p["name"], "prompt": p["prompt"]} for p in PRESET_STRATEGIES]


@app.post("/api/backtest")
def run_backtest_legacy(req: dict):
    """Legacy endpoint kept for backward compat."""
    try:
        strategy_type = req.get("strategy_type", "nl")
        symbol = req.get("symbol", "^NSEI")
        period = req.get("period", "1y")
        interval = req.get("interval", "1d")

        if strategy_type == "nl":
            prompt = req.get("prompt", "")
            if len(prompt.strip()) < 10:
                raise HTTPException(status_code=400, detail="Prompt must be at least 10 characters.")
            code_str = generate_strategy_from_nl(prompt.strip())
        elif strategy_type == "preset":
            preset_id = req.get("preset_id")
            preset = next((p for p in PRESET_STRATEGIES if p["id"] == preset_id), None)
            if not preset:
                raise HTTPException(status_code=404, detail=f"Preset '{preset_id}' not found.")
            code_str = preset["code"]
        else:
            raise HTTPException(status_code=400, detail="Invalid strategy_type.")

        return _run_backtest_from_code(code_str, {}, symbol, period, interval)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
