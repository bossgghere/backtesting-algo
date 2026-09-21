import re
import json
import time
import urllib.request
import urllib.error

from config import GEMINI_API_KEY

MAX_PROMPT_CHARS = 2000
MIN_PROMPT_CHARS = 10

SYSTEM_PROMPT = """
You are an expert quantitative trading strategist and Python developer for 'studio.trade'.
Your job is to convert a user's natural language trading idea into safe, executable Python strategy code.

STRICT CONTRACT & REQUIREMENTS:
1. You MUST define a top-level function named `generate_signals`:
   def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
   - `df` contains lower-case OHLCV columns: ['open', 'high', 'low', 'close', 'volume'].
   - `params` is a dictionary containing tunable indicator parameters (periods, thresholds, stop-loss %).
   - Return a `pd.Series` with identical index to `df.index` containing signal values:
      +1 : Long Entry / Buy
      -1 : Short Entry or Exit / Sell
       0 : Hold / Neutral

2. Tunable numbers MUST NOT be hardcoded inside logic. Extract default parameters into `params` at top of function:
   period = params.get('rsi_period', 14)
   oversold = params.get('rsi_oversold', 30)

3. Imports are STRICTLY restricted to: pandas as pd, numpy as np, math, ta (technical analysis library).
   DO NOT import os, sys, subprocess, eval, open, or any other module.

4. EVERY single logic line MUST have an inline comment `#` explaining the trading rationale (PRD Requirement §7.1).

5. NaN SAFETY IS MANDATORY: All indicator series must use boolean masks to avoid NaN signal assignments.
   Always initialize signals as zero and apply signals only on valid (non-NaN) rows:
       signals = pd.Series(0, index=df.index)
       valid = indicator.notna()
       signals[valid & (indicator < threshold)] = 1

6. MINIMUM DATA GUARD: At the very top of generate_signals, check that df has enough rows for the
   longest lookback window. If not, return a zero Series immediately:
       if len(df) < max_lookback:
           return pd.Series(0, index=df.index)

7. NEVER use pandas chained assignment. Always use boolean Series masks on the signals Series:
       signals[condition] = 1   # CORRECT
       df['col'][mask] = val    # FORBIDDEN

8. Output ONLY valid executable Python code wrapped in ```python ... ``` markdown block. No conversational filler, no markdown outside the code block, no explanations.

9. SYNTAX MUST BE PERFECT: The code will be parsed by Python's AST before execution. Any syntax error will cause complete failure. Double-check all parentheses are balanced, all strings are closed, and all indentation is correct before outputting.

EXAMPLE OUTPUT:
```python
import pandas as pd
import numpy as np

def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    if len(df) < 15:                              # Minimum bars needed for RSI(14)
        return pd.Series(0, index=df.index)

    period   = params.get('rsi_period', 14)       # RSI calculation lookback window
    oversold = params.get('rsi_oversold', 30)     # Oversold threshold to trigger buy
    overbought = params.get('rsi_overbought', 70) # Overbought threshold to exit

    delta = df['close'].diff()                                      # Bar-to-bar close change
    gain  = delta.where(delta > 0, 0).rolling(window=period).mean() # Average gains
    loss  = (-delta.where(delta < 0, 0)).rolling(window=period).mean() # Average losses
    rs    = gain / (loss + 1e-10)                                   # Relative Strength (avoid div/0)
    rsi   = 100 - (100 / (1 + rs))                                  # RSI formula 0–100

    signals = pd.Series(0, index=df.index)        # Initialize neutral signals
    valid   = rsi.notna()                          # Only assign where RSI is computed
    signals[valid & (rsi < oversold)]   = 1       # BUY when RSI dips below oversold
    signals[valid & (rsi > overbought)] = -1      # SELL when RSI spikes above overbought

    return signals
```
"""


def _call_gemini_with_retry(url: str, payload: dict, max_retries: int = 3) -> dict:
    """
    Calls a Gemini API endpoint with exponential backoff on transient errors (429/500/503).
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))

        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and attempt < max_retries - 1:
                wait = 2.0 ** attempt  # 1s, 2s, 4s
                print(f"  [Retry {attempt + 1}/{max_retries}] HTTP {e.code} — retrying in {wait:.0f}s...")
                time.sleep(wait)
                last_err = e
            else:
                raise

        except urllib.error.URLError as e:
            if attempt < max_retries - 1:
                wait = 2.0 ** attempt
                print(f"  [Retry {attempt + 1}/{max_retries}] Network error — retrying in {wait:.0f}s...")
                time.sleep(wait)
                last_err = e
            else:
                raise

    raise last_err


def generate_strategy_from_nl(user_prompt: str, api_key: str = None) -> str:
    """
    Calls Gemini API to convert a natural language strategy description into executable Python code.

    Args:
        user_prompt : Plain-English strategy description (10–2000 characters)
        api_key     : Override API key (uses config if omitted)

    Returns:
        Python code string (already extracted from markdown block)
    """
    # Input validation
    if len(user_prompt) < MIN_PROMPT_CHARS:
        raise ValueError(
            f"Strategy prompt is too short ({len(user_prompt)} chars). "
            f"Please describe your strategy in at least {MIN_PROMPT_CHARS} characters."
        )
    if len(user_prompt) > MAX_PROMPT_CHARS:
        raise ValueError(
            f"Strategy prompt is too long ({len(user_prompt)} chars). "
            f"Maximum is {MAX_PROMPT_CHARS} characters."
        )

    key = api_key or GEMINI_API_KEY
    if not key:
        raise ValueError(
            "Gemini API key is missing. Set GEMINI_API_KEY in your .env file "
            "or get a free key at https://aistudio.google.com/app/apikey"
        )

    models = ["gemini-3.6-flash", "gemini-3.1-pro-preview"]
    last_errs = []

    for model in models:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={key}"
        )
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": SYSTEM_PROMPT},
                        {"text": f"User Strategy Idea: {user_prompt}"},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 2048,
            },
        }

        try:
            res_data = _call_gemini_with_retry(url, payload)
            text_out = res_data["candidates"][0]["content"]["parts"][0]["text"]
            return _extract_code_block(text_out)

        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                pass
            last_errs.append(f"Model {model} → HTTP {e.code}: {err_body[:200]}")

        except urllib.error.URLError as e:
            last_errs.append(f"Model {model} → Network error: {e.reason}")

        except Exception as e:
            last_errs.append(f"Model {model} → Error: {e}")

    # Classify the failure type for a useful error message
    is_network = any(
        kw in err.lower()
        for err in last_errs
        for kw in ("network", "urlopen", "connection", "timeout", "reason")
    )
    if is_network:
        raise ConnectionError(
            "Could not reach Gemini API. Check your internet connection.\n"
            + "\n".join(last_errs)
        )
    raise RuntimeError(
        "Gemini strategy generation failed after trying all models.\n"
        "Possible causes: invalid API key, quota exceeded, or service unavailable.\n"
        + "\n".join(last_errs)
    )


def _extract_code_block(response_text: str) -> str:
    """Extracts Python code from a Markdown code block robustly."""
    pattern = r"```(?:python)?\s*\n?(.*?)\s*```"
    match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback: strip any stray fence markers and return raw text
    lines = [
        line for line in response_text.splitlines()
        if not line.strip().startswith("```")
    ]
    return "\n".join(lines).strip()
