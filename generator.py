import re
import json
import urllib.request
import urllib.error
from config import GEMINI_API_KEY, ALLOWED_MODULES

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
3. Imports are STRICTLY restricted to: pandas as pd, numpy as np, math. DO NOT import os, sys, subprocess, eval, or open.
4. EVERY single logic line MUST have an inline comment `#` explaining the trading rationale (PRD Requirement §7.1).
5. Output ONLY valid executable Python code wrapped in ```python ... ``` markdown block. No conversational filler.

EXAMPLE OUTPUT:
```python
import pandas as pd
import numpy as np

def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    # 1. Extract tunable strategy parameters with sensible defaults
    period = params.get('rsi_period', 14) # RSI calculation lookback window
    oversold = params.get('rsi_oversold', 30) # Oversold threshold to trigger buy
    overbought = params.get('rsi_overbought', 70) # Overbought threshold to exit

    # 2. Calculate price changes and RSI indicator
    delta = df['close'].diff() # Calculate bar-to-bar close price difference
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean() # Average gains
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean() # Average losses
    rs = gain / (loss + 1e-10) # Relative Strength calculation (avoid division by zero)
    rsi = 100 - (100 / (1 + rs)) # RSI formula 0 to 100

    # 3. Generate trading signals
    signals = pd.Series(0, index=df.index) # Initialize neutral signals series
    signals[rsi < oversold] = 1 # Generate BUY signal when RSI dips below oversold
    signals[rsi > overbought] = -1 # Generate SELL signal when RSI crosses above overbought

    return signals
```
"""

def generate_strategy_from_nl(user_prompt: str, api_key: str = None) -> str:
    """
    Calls Gemini API to convert natural language strategy description to executable Python code.
    """
    key = api_key or GEMINI_API_KEY
    if not key:
        raise ValueError(
            "Gemini API key is missing! Please set GEMINI_API_KEY in your .env file or get a free key at https://aistudio.google.com/app/apikey"
        )

    # Gemini 3.6 Flash models
    models = ["gemini-3.6-flash", "gemini-2.0-flash-exp", "gemini-2.5-flash-001"]
    last_errs = []

    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": SYSTEM_PROMPT},
                        {"text": f"User Strategy Idea: {user_prompt}"}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 2048
            }
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )

        try:
            with urllib.request.urlopen(req) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                text_out = res_data['candidates'][0]['content']['parts'][0]['text']
                code = _extract_code_block(text_out)
                return code
        except urllib.error.HTTPError as e:
            err_body = e.read().decode('utf-8')
            last_errs.append(f"Model {model} -> HTTP {e.code}: {err_body}")
        except Exception as e:
            last_errs.append(f"Model {model} -> Error: {e}")

    raise RuntimeError(f"Failed to generate strategy code:\n" + "\n".join(last_errs))


def _extract_code_block(response_text: str) -> str:
    """Extracts python code block from Markdown output robustly."""
    pattern = r"```(?:python)?\s*\n?(.*?)\s*```"
    match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Remove any stray ``` fences
    lines = [line for line in response_text.splitlines() if not line.strip().startswith("```")]
    return "\n".join(lines).strip()
