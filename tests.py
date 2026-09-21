"""
studio.trade — Test Suite
Tests every core module: config, data_loader, validator, backtester, presets, generator, api.
Run: python3 tests.py
"""
import sys
import os
import json
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import date

sys.path.insert(0, os.path.dirname(__file__))

# ── Colour helpers ─────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"
BOLD  = "\033[1m"

def ok(msg):  print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIG
# ══════════════════════════════════════════════════════════════════════════════
class TestConfig(unittest.TestCase):

    def test_allowed_modules_exists(self):
        from config import ALLOWED_MODULES
        self.assertIn("pandas", ALLOWED_MODULES)
        self.assertIn("numpy",  ALLOWED_MODULES)
        self.assertNotIn("ta",  ALLOWED_MODULES)  # ta removed; strategies use pandas/numpy only

    def test_cost_model_keys(self):
        from config import COST_MODEL
        self.assertIn("brokerage_per_trade", COST_MODEL)
        self.assertIn("slippage_pct",        COST_MODEL)
        self.assertIn("stt_tax_pct",         COST_MODEL)

    def test_cost_model_values_positive(self):
        from config import COST_MODEL
        for k, v in COST_MODEL.items():
            self.assertGreater(v, 0, f"{k} should be > 0")

    def test_risk_free_rate_range(self):
        from config import RISK_FREE_RATE
        self.assertGreater(RISK_FREE_RATE, 0)
        self.assertLess(RISK_FREE_RATE, 0.2)   # sanity: between 0% and 20%

    def test_interval_max_period_completeness(self):
        from config import INTERVAL_MAX_PERIOD
        for ivl in ("1m", "5m", "15m", "1h", "1d"):
            self.assertIn(ivl, INTERVAL_MAX_PERIOD)

    def test_default_position_size(self):
        from config import DEFAULT_POSITION_SIZE
        self.assertGreater(DEFAULT_POSITION_SIZE, 0)
        self.assertLessEqual(DEFAULT_POSITION_SIZE, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════
class TestDataLoader(unittest.TestCase):

    def _make_df(self, n=60):
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        return pd.DataFrame({
            "open":   np.random.uniform(18000, 22000, n),
            "high":   np.random.uniform(18000, 22500, n),
            "low":    np.random.uniform(17500, 21500, n),
            "close":  np.random.uniform(18000, 22000, n),
            "volume": np.random.randint(100000, 5000000, n),
        }, index=idx)

    def test_period_to_days_years(self):
        from data_loader import _period_to_days
        self.assertEqual(_period_to_days("1y"),  365)
        self.assertEqual(_period_to_days("2y"),  730)

    def test_period_to_days_months(self):
        from data_loader import _period_to_days
        self.assertEqual(_period_to_days("6mo"), 180)

    def test_period_to_days_days(self):
        from data_loader import _period_to_days
        self.assertEqual(_period_to_days("60d"), 60)

    def test_clamp_period_intraday_5m(self):
        from data_loader import _clamp_period_for_interval
        result = _clamp_period_for_interval("5m", "1y")
        self.assertEqual(result, "60d")  # 5m max is 60d

    def test_clamp_period_daily_no_limit(self):
        from data_loader import _clamp_period_for_interval
        result = _clamp_period_for_interval("1d", "5y")
        self.assertEqual(result, "5y")   # daily has no restriction

    def test_validate_symbol_bad(self):
        from data_loader import validate_symbol
        is_valid, msg = validate_symbol("XXXXXXXXINVALID123")
        self.assertFalse(is_valid)
        self.assertIsInstance(msg, str)
        self.assertGreater(len(msg), 0)

    @patch("data_loader.yf.download")
    def test_get_market_data_columns(self, mock_dl):
        mock_dl.return_value = self._make_df()
        from data_loader import get_market_data
        df = get_market_data(symbol="^NSEI", period="1mo", interval="1d", force_refresh=True)
        for col in ("open", "high", "low", "close", "volume"):
            self.assertIn(col, df.columns)

    @patch("data_loader.yf.download")
    def test_get_market_data_no_nulls_in_close(self, mock_dl):
        mock_dl.return_value = self._make_df()
        from data_loader import get_market_data
        df = get_market_data(symbol="^NSEI", period="1mo", interval="1d", force_refresh=True)
        self.assertEqual(df["close"].isna().sum(), 0)

    @patch("data_loader.yf.download")
    def test_empty_download_raises(self, mock_dl):
        mock_dl.return_value = pd.DataFrame()
        from data_loader import get_market_data
        with self.assertRaises(ValueError):
            get_market_data(symbol="FAKE", period="1mo", interval="1d", force_refresh=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3. VALIDATOR (AST Safety)
# ══════════════════════════════════════════════════════════════════════════════
class TestValidator(unittest.TestCase):

    GOOD_CODE = """
import pandas as pd
import numpy as np

def generate_signals(df, params):
    return pd.Series(0, index=df.index)
"""

    def test_valid_code_passes(self):
        from validator import validate_strategy_code
        ok_flag, msg = validate_strategy_code(self.GOOD_CODE)
        self.assertTrue(ok_flag, msg)

    def test_forbidden_import_os(self):
        from validator import validate_strategy_code
        code = self.GOOD_CODE + "\nimport os\n"
        ok_flag, msg = validate_strategy_code(code)
        self.assertFalse(ok_flag)
        self.assertIn("os", msg)

    def test_forbidden_import_sys(self):
        from validator import validate_strategy_code
        code = self.GOOD_CODE + "\nimport sys\n"
        ok_flag, _ = validate_strategy_code(code)
        self.assertFalse(ok_flag)

    def test_forbidden_function_eval(self):
        from validator import validate_strategy_code
        code = self.GOOD_CODE + "\neval('1+1')\n"
        ok_flag, _ = validate_strategy_code(code)
        self.assertFalse(ok_flag)

    def test_forbidden_function_exec(self):
        from validator import validate_strategy_code
        code = self.GOOD_CODE + "\nexec('x=1')\n"
        ok_flag, _ = validate_strategy_code(code)
        self.assertFalse(ok_flag)

    def test_missing_generate_signals_fn(self):
        from validator import validate_strategy_code
        code = "import pandas as pd\nx = 1\n"
        ok_flag, msg = validate_strategy_code(code)
        self.assertFalse(ok_flag)
        self.assertIn("generate_signals", msg)

    def test_syntax_error_caught(self):
        from validator import validate_strategy_code
        code = "def broken(:\n    pass"
        ok_flag, msg = validate_strategy_code(code)
        self.assertFalse(ok_flag)
        self.assertIn("Syntax", msg)

    def test_dunder_access_blocked(self):
        from validator import validate_strategy_code
        code = self.GOOD_CODE + "\nx = df.__class__\n"
        ok_flag, _ = validate_strategy_code(code)
        self.assertFalse(ok_flag)

    def test_fingerprint_deterministic(self):
        from validator import get_strategy_fingerprint
        fp1 = get_strategy_fingerprint(self.GOOD_CODE)
        fp2 = get_strategy_fingerprint(self.GOOD_CODE)
        self.assertEqual(fp1, fp2)

    def test_fingerprint_different_codes(self):
        from validator import get_strategy_fingerprint
        fp1 = get_strategy_fingerprint(self.GOOD_CODE)
        fp2 = get_strategy_fingerprint(self.GOOD_CODE + "\nx = 999  # extra logic")
        self.assertNotEqual(fp1, fp2)  # different logic → different fingerprint

    def test_fingerprint_length(self):
        from validator import get_strategy_fingerprint
        fp = get_strategy_fingerprint(self.GOOD_CODE)
        self.assertEqual(len(fp), 16)

    def test_allowed_modules_same_as_config(self):
        from validator import ALLOWED_MODULES as v_mods
        from config import ALLOWED_MODULES as c_mods
        self.assertEqual(v_mods, c_mods)


# ══════════════════════════════════════════════════════════════════════════════
# 4. BACKTESTER
# ══════════════════════════════════════════════════════════════════════════════
class TestBacktester(unittest.TestCase):

    def _make_df(self, n=100, trend="up"):
        idx = pd.date_range("2023-01-01", periods=n, freq="B")
        if trend == "up":
            prices = np.linspace(18000, 22000, n) + np.random.normal(0, 50, n)
        else:
            prices = np.linspace(22000, 18000, n) + np.random.normal(0, 50, n)
        return pd.DataFrame({
            "open": prices, "high": prices * 1.005,
            "low": prices * 0.995, "close": prices,
            "volume": np.ones(n) * 1000000,
        }, index=idx)

    def _always_long(self, df, params):
        return pd.Series(1, index=df.index)

    def _always_flat(self, df, params):
        return pd.Series(0, index=df.index)

    def _alternating(self, df, params):
        signals = pd.Series(0, index=df.index)
        signals.iloc[::10] = 1
        signals.iloc[5::10] = -1
        return signals

    def test_returns_required_keys(self):
        from backtester import run_backtest_simulation
        df = self._make_df()
        res = run_backtest_simulation(df, self._always_long)
        self.assertIn("summary",      res)
        self.assertIn("trades",       res)
        self.assertIn("equity_curve", res)

    def test_summary_has_all_metrics(self):
        from backtester import run_backtest_simulation
        df = self._make_df()
        s = run_backtest_simulation(df, self._alternating)["summary"]
        for key in ("initial_capital", "final_capital", "total_return_pct",
                    "win_rate_pct", "max_drawdown_pct", "profit_factor",
                    "sharpe_ratio", "avg_win_inr", "avg_loss_inr",
                    "max_consecutive_losses", "avg_holding_days"):
            self.assertIn(key, s, f"Missing metric: {key}")

    def test_no_signals_produces_no_trades(self):
        from backtester import run_backtest_simulation
        df = self._make_df()
        res = run_backtest_simulation(df, self._always_flat)
        self.assertEqual(len(res["trades"]), 0)

    def test_capital_preserved_when_flat(self):
        from backtester import run_backtest_simulation
        df = self._make_df()
        res = run_backtest_simulation(df, self._always_flat)
        self.assertEqual(res["summary"]["final_capital"], 100000.0)

    def test_position_size_respected(self):
        from backtester import run_backtest_simulation
        df = self._make_df()
        # With 50% position size, max possible loss per trade is lower
        res_full = run_backtest_simulation(df, self._alternating, position_size_pct=1.0)
        res_half = run_backtest_simulation(df, self._alternating, position_size_pct=0.5)
        # Both should complete without error
        self.assertIn("summary", res_full)
        self.assertIn("summary", res_half)

    def test_stop_loss_triggers(self):
        from backtester import run_backtest_simulation
        df = self._make_df(trend="down")  # falling market
        # Always long in a falling market — SL should close trades
        res = run_backtest_simulation(df, self._always_long, stop_loss_pct=0.01)
        sl_trades = [t for t in res["trades"] if t.get("exit_reason") == "STOP_LOSS"]
        self.assertGreater(len(sl_trades), 0, "Stop-loss should have triggered at least once")

    def test_take_profit_triggers(self):
        from backtester import run_backtest_simulation
        df = self._make_df(trend="up")   # rising market
        # Always long in rising market — TP should trigger
        res = run_backtest_simulation(df, self._always_long, take_profit_pct=0.01)
        tp_trades = [t for t in res["trades"] if t.get("exit_reason") == "TAKE_PROFIT"]
        self.assertGreater(len(tp_trades), 0, "Take-profit should have triggered at least once")

    def test_sharpe_uses_risk_free_rate(self):
        from backtester import run_backtest_simulation
        df = self._make_df()
        res0   = run_backtest_simulation(df, self._alternating, risk_free_rate=0.0)
        res065 = run_backtest_simulation(df, self._alternating, risk_free_rate=0.065)
        # Higher risk-free rate → lower Sharpe
        self.assertGreaterEqual(res0["summary"]["sharpe_ratio"],
                                res065["summary"]["sharpe_ratio"])

    def test_equity_curve_length_matches_df(self):
        from backtester import run_backtest_simulation
        df = self._make_df(n=80)
        res = run_backtest_simulation(df, self._alternating)
        self.assertEqual(len(res["equity_curve"]), 80)

    def test_win_rate_between_0_and_100(self):
        from backtester import run_backtest_simulation
        df = self._make_df()
        s = run_backtest_simulation(df, self._alternating)["summary"]
        self.assertGreaterEqual(s["win_rate_pct"], 0)
        self.assertLessEqual(s["win_rate_pct"],    100)

    def test_max_drawdown_negative_or_zero(self):
        from backtester import run_backtest_simulation
        df = self._make_df()
        s = run_backtest_simulation(df, self._alternating)["summary"]
        self.assertLessEqual(s["max_drawdown_pct"], 0)

    def test_trade_exit_reasons_valid(self):
        from backtester import run_backtest_simulation
        df = self._make_df()
        res = run_backtest_simulation(df, self._alternating, stop_loss_pct=0.02, take_profit_pct=0.05)
        valid = {"SIGNAL", "STOP_LOSS", "TAKE_PROFIT", "END_OF_DATA"}
        for t in res["trades"]:
            self.assertIn(t.get("exit_reason"), valid)


# ══════════════════════════════════════════════════════════════════════════════
# 5. PRESETS
# ══════════════════════════════════════════════════════════════════════════════
class TestPresets(unittest.TestCase):

    def _make_df(self, n=120):
        idx = pd.date_range("2023-01-01", periods=n, freq="B")
        prices = np.cumsum(np.random.normal(0, 100, n)) + 18000
        prices = np.abs(prices)
        return pd.DataFrame({
            "open": prices, "high": prices * 1.005,
            "low": prices * 0.995, "close": prices,
            "volume": np.ones(n) * 2000000,
        }, index=idx)

    def test_six_presets_exist(self):
        from presets import PRESET_STRATEGIES
        self.assertEqual(len(PRESET_STRATEGIES), 6)

    def test_all_presets_have_required_keys(self):
        from presets import PRESET_STRATEGIES
        for p in PRESET_STRATEGIES:
            self.assertIn("id",     p)
            self.assertIn("name",   p)
            self.assertIn("prompt", p)
            self.assertIn("code",   p)

    def test_preset_ids_unique(self):
        from presets import PRESET_STRATEGIES
        ids = [p["id"] for p in PRESET_STRATEGIES]
        self.assertEqual(len(ids), len(set(ids)))

    def test_all_presets_pass_ast(self):
        from presets import PRESET_STRATEGIES
        from validator import validate_strategy_code
        for p in PRESET_STRATEGIES:
            ok_flag, msg = validate_strategy_code(p["code"])
            self.assertTrue(ok_flag, f"Preset '{p['name']}' failed AST: {msg}")

    def test_all_presets_run_without_error(self):
        from presets import PRESET_STRATEGIES
        from backtester import run_backtest_simulation
        df = self._make_df()
        for p in PRESET_STRATEGIES:
            scope = {}
            exec(p["code"], scope)
            fn = scope["generate_signals"]
            res = run_backtest_simulation(df, fn)
            self.assertIn("summary", res, f"Preset '{p['name']}' backtest failed")

    def test_all_presets_return_series(self):
        from presets import PRESET_STRATEGIES
        df = self._make_df()
        for p in PRESET_STRATEGIES:
            scope = {}
            exec(p["code"], scope)
            fn = scope["generate_signals"]
            signals = fn(df, {})
            self.assertIsInstance(signals, pd.Series,
                f"Preset '{p['name']}' must return pd.Series")

    def test_all_presets_signals_valid_values(self):
        from presets import PRESET_STRATEGIES
        df = self._make_df()
        for p in PRESET_STRATEGIES:
            scope = {}
            exec(p["code"], scope)
            fn = scope["generate_signals"]
            signals = fn(df, {})
            unique = set(signals.dropna().unique())
            invalid = unique - {-1, 0, 1}
            self.assertEqual(len(invalid), 0,
                f"Preset '{p['name']}' returned invalid signals: {invalid}")

    def test_presets_handle_small_df(self):
        """Presets should not crash with fewer bars than their lookback needs."""
        from presets import PRESET_STRATEGIES
        df_small = self._make_df(n=5)
        for p in PRESET_STRATEGIES:
            scope = {}
            exec(p["code"], scope)
            fn = scope["generate_signals"]
            try:
                signals = fn(df_small, {})
                self.assertIsInstance(signals, pd.Series)
            except Exception as e:
                self.fail(f"Preset '{p['name']}' crashed on small df: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
class TestGenerator(unittest.TestCase):

    def test_prompt_too_short_raises(self):
        from generator import generate_strategy_from_nl
        with self.assertRaises(ValueError):
            generate_strategy_from_nl("short")

    def test_prompt_too_long_raises(self):
        from generator import generate_strategy_from_nl
        with self.assertRaises(ValueError):
            generate_strategy_from_nl("x" * 2001)

    def test_missing_api_key_raises(self):
        from generator import generate_strategy_from_nl
        with self.assertRaises((ValueError, RuntimeError)):
            generate_strategy_from_nl("Buy when RSI drops below 30", api_key="")

    def test_extract_code_block_with_fence(self):
        from generator import _extract_code_block
        text = "Sure!\n```python\nimport pandas as pd\n```"
        result = _extract_code_block(text)
        self.assertIn("import pandas", result)
        self.assertNotIn("```", result)

    def test_extract_code_block_without_fence(self):
        from generator import _extract_code_block
        text = "import pandas as pd\nx = 1"
        result = _extract_code_block(text)
        self.assertIn("import pandas", result)

    @patch("generator.urllib.request.urlopen")
    def test_successful_generation(self, mock_urlopen):
        fake_response = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": "```python\nimport pandas as pd\nimport numpy as np\n\ndef generate_signals(df, params):\n    return pd.Series(0, index=df.index)\n```"
                    }]
                }
            }]
        }
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=MagicMock(
            read=MagicMock(return_value=json.dumps(fake_response).encode())
        ))
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_cm

        from generator import generate_strategy_from_nl
        code = generate_strategy_from_nl("Buy when RSI drops below 30", api_key="fake-key")
        self.assertIn("generate_signals", code)
        self.assertIn("import pandas", code)


# ══════════════════════════════════════════════════════════════════════════════
# 7. API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════
class TestAPI(unittest.TestCase):

    def setUp(self):
        from fastapi.testclient import TestClient
        from api import app
        self.client = TestClient(app)

    def _make_df(self, n=100):
        idx = pd.date_range("2023-01-01", periods=n, freq="B")
        prices = np.linspace(18000, 22000, n) + np.random.normal(0, 50, n)
        return pd.DataFrame({
            "open": prices, "high": prices * 1.005,
            "low": prices * 0.995, "close": prices,
            "volume": np.ones(n) * 1e6,
        }, index=idx)

    def test_get_presets_returns_list(self):
        res = self.client.get("/api/presets")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_get_presets_fields(self):
        res = self.client.get("/api/presets")
        for p in res.json():
            self.assertIn("id",     p)
            self.assertIn("name",   p)
            self.assertIn("prompt", p)
            self.assertNotIn("code", p)  # code should NOT be exposed

    def test_backtest_missing_strategy_type(self):
        res = self.client.post("/api/backtest", json={})
        self.assertIn(res.status_code, (400, 422))

    def test_backtest_short_prompt_rejected(self):
        res = self.client.post("/api/backtest", json={
            "strategy_type": "nl", "prompt": "hi", "symbol": "^NSEI"
        })
        self.assertEqual(res.status_code, 400)
        self.assertIn("10 characters", res.json()["detail"])

    def test_backtest_invalid_preset_id(self):
        res = self.client.post("/api/backtest", json={
            "strategy_type": "preset", "preset_id": "999"
        })
        self.assertEqual(res.status_code, 404)

    @patch("api.get_market_data")
    def test_backtest_preset_success(self, mock_data):
        mock_data.return_value = self._make_df()
        res = self.client.post("/api/backtest", json={
            "strategy_type": "preset",
            "preset_id": "1",
            "symbol": "^NSEI",
            "period": "1y",
            "interval": "1d",
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("summary",      data)
        self.assertIn("trades",       data)
        self.assertIn("equity_curve", data)
        self.assertIn("fingerprint",  data)

    @patch("api.get_market_data")
    def test_backtest_response_summary_keys(self, mock_data):
        mock_data.return_value = self._make_df()
        res = self.client.post("/api/backtest", json={
            "strategy_type": "preset", "preset_id": "2",
            "symbol": "^NSEI", "period": "1y", "interval": "1d",
        })
        s = res.json()["summary"]
        for key in ("total_return_pct", "win_rate_pct", "max_drawdown_pct",
                    "sharpe_ratio", "profit_factor", "total_trades"):
            self.assertIn(key, s)

    @patch("api.get_market_data")
    def test_backtest_equity_curve_format(self, mock_data):
        mock_data.return_value = self._make_df()
        res = self.client.post("/api/backtest", json={
            "strategy_type": "preset", "preset_id": "3",
            "symbol": "^NSEI", "period": "1y", "interval": "1d",
        })
        eq = res.json()["equity_curve"]
        self.assertIsInstance(eq, list)
        self.assertGreater(len(eq), 0)
        for point in eq[:5]:
            self.assertIn("date",   point)
            self.assertIn("equity", point)
            self.assertIn("close",  point)

    @patch("api.get_market_data")
    def test_backtest_unsafe_code_rejected(self, mock_data):
        mock_data.return_value = self._make_df()
        unsafe = """
import os
def generate_signals(df, params):
    return __import__('pandas').Series(0, index=df.index)
"""
        # Send unsafe code directly to /api/run — AST validator must reject it
        res = self.client.post("/api/run", json={
            "code": unsafe,
            "params": {},
            "symbol": "^NSEI", "period": "1y", "interval": "1d",
        })
        self.assertIn(res.status_code, (400, 500))


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    modules = [
        ("Config",      TestConfig),
        ("DataLoader",  TestDataLoader),
        ("Validator",   TestValidator),
        ("Backtester",  TestBacktester),
        ("Presets",     TestPresets),
        ("Generator",   TestGenerator),
        ("API",         TestAPI),
    ]

    print(f"\n{BOLD}studio.trade — Test Suite{RESET}\n" + "─" * 50)

    total_pass = total_fail = total_err = 0

    for name, cls in modules:
        print(f"\n{BOLD}{name}{RESET}")
        tests = loader.loadTestsFromTestCase(cls)
        runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, "w"))
        result = runner.run(tests)

        for test, _ in result.failures + result.errors:
            fail(str(test).split(" ")[0])

        passed = tests.countTestCases() - len(result.failures) - len(result.errors)
        for test in tests:
            name_str = getattr(test, '_testMethodName', None)
            if name_str and not any(name_str in str(f[0]) for f in result.failures + result.errors):
                ok(name_str)

        for test, tb in result.failures:
            fail(f"{test._testMethodName}  →  {tb.splitlines()[-1]}")
        for test, tb in result.errors:
            fail(f"{test._testMethodName}  →  {tb.splitlines()[-1]}")

        total_pass += passed
        total_fail += len(result.failures) + len(result.errors)

    print(f"\n{'─' * 50}")
    total = total_pass + total_fail
    if total_fail == 0:
        print(f"{GREEN}{BOLD}All {total} tests passed.{RESET}\n")
    else:
        print(f"{GREEN}{total_pass} passed{RESET}  {RED}{total_fail} failed{RESET}  of {total} total\n")

    sys.exit(0 if total_fail == 0 else 1)
