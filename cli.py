import sys
import os
import csv
import threading
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from config import (
    GEMINI_API_KEY, DEFAULT_SYMBOL, DEFAULT_PERIOD,
    DEFAULT_INTERVAL, COST_MODEL, RISK_FREE_RATE, DEFAULT_POSITION_SIZE,
)
from data_loader import get_market_data, validate_symbol
from validator import validate_strategy_code, get_strategy_fingerprint
from generator import generate_strategy_from_nl
from backtester import run_backtest_simulation
from presets import PRESET_STRATEGIES

# Try importing rich for enhanced terminal output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.prompt import Prompt
    console = Console()
    USE_RICH = True
except ImportError:
    USE_RICH = False

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "backtest_results")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _show_error(msg: str):
    """Uniform error display across Rich and plain modes."""
    if USE_RICH:
        console.print(f"[bold red]ERROR:[/bold red] {msg}")
    else:
        print(f"\nERROR: {msg}\n")


def _show_info(msg: str):
    if USE_RICH:
        console.print(f"[bold cyan]{msg}[/bold cyan]")
    else:
        print(msg)


def _show_success(msg: str):
    if USE_RICH:
        console.print(f"[bold green]{msg}[/bold green]")
    else:
        print(msg)


def _spinner(stop_event: threading.Event, message: str):
    """Simple spinner for non-Rich mode."""
    spin = itertools.cycle(["|", "/", "-", "\\"])
    while not stop_event.is_set():
        sys.stdout.write(f"\r{message} {next(spin)}")
        sys.stdout.flush()
        __import__("time").sleep(0.12)
    sys.stdout.write("\r" + " " * (len(message) + 4) + "\r")
    sys.stdout.flush()


def _run_with_spinner(message: str, fn, *args, **kwargs):
    """Runs fn() with a progress indicator. Returns fn's result."""
    if USE_RICH:
        with console.status(f"[bold green]{message}[/bold green]", spinner="dots"):
            return fn(*args, **kwargs)
    else:
        stop = threading.Event()
        t = threading.Thread(target=_spinner, args=(stop, message), daemon=True)
        t.start()
        try:
            result = fn(*args, **kwargs)
        finally:
            stop.set()
            t.join()
        return result


# ---------------------------------------------------------------------------
# Display Functions
# ---------------------------------------------------------------------------

def print_header():
    title    = "studio.trade — Universal Algo Trading Platform CLI"
    subtitle = "Natural Language Strategy Authoring · AST Validation · Real Market Backtesting"
    if USE_RICH:
        console.print(Panel(
            f"[bold green]{title}[/bold green]\n[dim]{subtitle}[/dim]",
            border_style="cyan",
        ))
    else:
        print("=" * 70)
        print(f"  {title}")
        print(f"  {subtitle}")
        print("=" * 70)


def display_code(code_str: str):
    if USE_RICH:
        syntax = Syntax(code_str, "python", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title="Generated Strategy Code (AST Verified)", border_style="green"))
    else:
        print("\n--- GENERATED STRATEGY CODE ---")
        print(code_str)
        print("-------------------------------\n")


def _display_ascii_equity_curve(equity_series: pd.Series, width: int = 60, height: int = 10):
    """Renders a simple ASCII bar chart of the equity curve."""
    vals = equity_series.dropna().values
    if len(vals) < 2:
        return

    min_v, max_v = vals.min(), vals.max()
    if max_v == min_v:
        return

    # Downsample to `width` columns
    indices = np.linspace(0, len(vals) - 1, width).astype(int)
    sampled = vals[indices]
    normalized = ((sampled - min_v) / (max_v - min_v) * (height - 1)).astype(int)

    print("\n  Equity Curve")
    print(f"  ₹{max_v:,.0f} ┐")

    grid = [[" "] * width for _ in range(height)]
    for col, row_val in enumerate(normalized):
        grid[height - 1 - row_val][col] = "█"

    for row in grid:
        print("          │" + "".join(row))

    print("          └" + "─" * width)
    print(f"  ₹{min_v:,.0f}    Start {'─' * 20} End  ₹{vals[-1]:,.0f}\n")


def display_backtest_results(results: dict, symbol: str, fingerprint: str):
    summary = results["summary"]
    trades  = results["trades"]

    if USE_RICH:
        # Performance Summary Table
        table = Table(
            title=f"Backtest Results [{symbol}] | Fingerprint: {fingerprint}",
            border_style="blue",
        )
        table.add_column("Metric",          style="cyan",        no_wrap=True)
        table.add_column("Value",           style="bold yellow")
        table.add_column("Notes",           style="dim")

        ret_color = "green" if summary["total_return_pct"] >= 0 else "red"
        dd_color  = "red"   if summary["max_drawdown_pct"] < -10 else "yellow"

        table.add_row("Initial Capital",       f"₹{summary['initial_capital']:,.2f}", "Starting balance")
        table.add_row("Final Capital",         f"₹{summary['final_capital']:,.2f}",   f"Net PnL: ₹{summary['final_capital'] - summary['initial_capital']:,.2f}")
        table.add_row("Strategy Return",       f"[{ret_color}]{summary['total_return_pct']}%[/{ret_color}]", f"Buy & Hold: {summary['buy_and_hold_pct']}%")
        table.add_row("Win Rate",              f"{summary['win_rate_pct']}%",          f"{summary['winning_trades']}W / {summary['losing_trades']}L of {summary['total_trades']} trades")
        table.add_row("Avg Win",               f"₹{summary['avg_win_inr']:,.2f}",      "Avg profit per winning trade")
        table.add_row("Avg Loss",              f"₹{summary['avg_loss_inr']:,.2f}",     "Avg loss per losing trade")
        table.add_row("Max Drawdown",          f"[{dd_color}]{summary['max_drawdown_pct']}%[/{dd_color}]", "Peak-to-trough decline")
        table.add_row("Profit Factor",         f"{summary['profit_factor']}",          "Gross profit / Gross loss")
        table.add_row("Sharpe Ratio",          f"{summary['sharpe_ratio']}",           "Risk-adjusted return (rf subtracted)")
        table.add_row("Max Consec. Losses",    f"{summary['max_consecutive_losses']}", "Worst losing streak")
        table.add_row("Avg Holding Period",    f"{summary['avg_holding_days']} days",  "Average trade duration")
        console.print(table)

        # Trade Log Table
        if trades:
            t_table = Table(
                title=f"Trade Log (Last {min(5, len(trades))} of {len(trades)} trades)",
                border_style="magenta",
            )
            t_table.add_column("#",           style="dim")
            t_table.add_column("Type",        style="bold")
            t_table.add_column("Entry Date")
            t_table.add_column("Exit Date")
            t_table.add_column("Entry ₹")
            t_table.add_column("Exit ₹")
            t_table.add_column("PnL %")
            t_table.add_column("Net PnL ₹")
            t_table.add_column("Exit Reason",  style="dim")

            for t in trades[-5:]:
                c = "green" if t["net_pnl"] >= 0 else "red"
                t_table.add_row(
                    str(t["trade_num"]),
                    t["type"],
                    t["entry_date"][:10],
                    t["exit_date"][:10],
                    f"₹{t['entry_price']}",
                    f"₹{t['exit_price']}",
                    f"[{c}]{t['pnl_pct']}%[/{c}]",
                    f"[{c}]₹{t['net_pnl']:.2f}[/{c}]",
                    t.get("exit_reason", "SIGNAL"),
                )
            console.print(t_table)
        else:
            console.print("[bold yellow]No trades triggered during this period.[/bold yellow]")

    else:
        print("\n=== BACKTEST RESULTS ===")
        print(f"Symbol: {symbol} | Fingerprint: {fingerprint}")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        if trades:
            print(f"\nLast {min(5, len(trades))} trades:")
            for t in trades[-5:]:
                print(f"  #{t['trade_num']} {t['type']} {t['entry_date']} → {t['exit_date']} | "
                      f"PnL: {t['pnl_pct']}% | ₹{t['net_pnl']:.2f} | {t.get('exit_reason','')}")
        else:
            print("No trades triggered.")

    # ASCII equity curve (always shown)
    try:
        eq = results["equity_curve"]["equity"]
        _display_ascii_equity_curve(eq)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------

def _save_results_to_csv(results: dict, symbol: str, fingerprint: str):
    """Saves summary metrics and full trade log to timestamped CSV files."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_sym = symbol.replace("^", "").replace(".", "_")

    # Summary CSV
    summary_path = os.path.join(RESULTS_DIR, f"{clean_sym}_{fingerprint}_{ts}_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in results["summary"].items():
            writer.writerow([k, v])

    # Trades CSV
    trades = results["trades"]
    if trades:
        trades_path = os.path.join(RESULTS_DIR, f"{clean_sym}_{fingerprint}_{ts}_trades.csv")
        with open(trades_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=trades[0].keys())
            writer.writeheader()
            writer.writerows(trades)

    msg = f"Results saved to backtest_results/{clean_sym}_{fingerprint}_{ts}_summary.csv"
    if USE_RICH:
        console.print(f"[dim]{msg}[/dim]")
    else:
        print(msg)


# ---------------------------------------------------------------------------
# Strategy Pipeline
# ---------------------------------------------------------------------------

def run_strategy_pipeline(
    code_str: str,
    symbol: str = "^NSEI",
    period: str = "1y",
    interval: str = "1d",
    force_refresh: bool = False,
    stop_loss_pct: float = None,
    take_profit_pct: float = None,
    position_size_pct: float = DEFAULT_POSITION_SIZE,
):
    """
    Full pipeline: AST Safety → Compile → Market Data → Backtest → Display → Export.
    """
    # 1. AST Safety Check
    _show_info("Running AST Safety Check...")
    is_safe, msg = validate_strategy_code(code_str)
    if not is_safe:
        _show_error(f"AST Validation Failed:\n{msg}")
        return

    fingerprint = get_strategy_fingerprint(code_str)
    _show_success(f"AST Check Passed! Fingerprint: {fingerprint}")
    display_code(code_str)

    # 2. Compile strategy code
    exec_scope = {}
    try:
        exec(code_str, exec_scope)
        generate_signals_fn = exec_scope.get("generate_signals")
        if generate_signals_fn is None:
            _show_error("Compiled code does not define 'generate_signals' function.")
            return
    except Exception as e:
        _show_error(f"Code compilation error: {e}")
        return

    # 3. Load market data
    try:
        df = _run_with_spinner(
            f"Loading market data for {symbol} ({period}, {interval})...",
            get_market_data,
            symbol=symbol, period=period, interval=interval, force_refresh=force_refresh,
        )
    except Exception as e:
        _show_error(str(e))
        return

    _show_info(f"Loaded {len(df)} candles for {symbol}.")

    # 4. Run backtest
    try:
        results = _run_with_spinner(
            "Running backtest simulation...",
            run_backtest_simulation,
            df=df,
            generate_signals_fn=generate_signals_fn,
            params={},
            initial_capital=100000.0,
            brokerage_per_trade=COST_MODEL["brokerage_per_trade"],
            slippage_pct=COST_MODEL["slippage_pct"],
            stt_tax_pct=COST_MODEL["stt_tax_pct"],
            risk_free_rate=RISK_FREE_RATE,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            position_size_pct=position_size_pct,
        )
    except Exception as e:
        _show_error(f"Backtest simulation error: {e}")
        return

    # 5. Display results and save CSV
    display_backtest_results(results, symbol, fingerprint)
    _save_results_to_csv(results, symbol, fingerprint)


# ---------------------------------------------------------------------------
# Advanced Settings Prompt
# ---------------------------------------------------------------------------

def _prompt_advanced_settings() -> dict:
    """Asks optional stop-loss, take-profit, position-size settings."""
    print("\nAdvanced Settings (press Enter to skip / use defaults):")
    settings = {}

    sl = input("  Stop-Loss % (e.g. 3 for 3%, blank = none): ").strip()
    if sl:
        try:
            settings["stop_loss_pct"] = float(sl) / 100
        except ValueError:
            print("  Invalid stop-loss — skipping.")

    tp = input("  Take-Profit % (e.g. 8 for 8%, blank = none): ").strip()
    if tp:
        try:
            settings["take_profit_pct"] = float(tp) / 100
        except ValueError:
            print("  Invalid take-profit — skipping.")

    ps = input(f"  Position Size % of capital (1–100, default {int(DEFAULT_POSITION_SIZE*100)}): ").strip()
    if ps:
        try:
            settings["position_size_pct"] = float(ps) / 100
        except ValueError:
            print("  Invalid position size — using default.")

    return settings


# ---------------------------------------------------------------------------
# Main Interactive Loop
# ---------------------------------------------------------------------------

def main():
    print_header()
    current_symbol   = DEFAULT_SYMBOL
    current_period   = DEFAULT_PERIOD
    current_interval = DEFAULT_INTERVAL

    while True:
        print("\n" + "─" * 60)
        print(f"  Instrument : {current_symbol}   Period : {current_period}   Interval : {current_interval}")
        print("─" * 60)
        print(" 1.  Author Strategy from Natural Language (Gemini LLM)")
        print(" 2.  Run Preset Strategy (Instant Backtest)")
        print(" 3.  Change Instrument / Period / Interval")
        print(" 4.  API Key & Config Status")
        print(" 5.  Exit")
        print("─" * 60)

        try:
            choice = input("Enter choice (1–5): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            sys.exit(0)

        # ------------------------------------------------------------------
        # Option 1: Natural Language Strategy via Gemini
        # ------------------------------------------------------------------
        if choice == "1":
            print("\nDescribe your trading strategy in plain English.")
            print("Examples:")
            print("  - 'Buy when RSI(14) drops below 30, sell when it rises above 65'")
            print("  - 'Buy when 20 EMA crosses above 50 EMA and volume > 20-day average'")
            print("  - 'Buy Nifty when MACD crosses signal line with increasing volume'")

            prompt = input("\nStrategy Prompt: ").strip()

            if len(prompt) < 10:
                _show_error("Prompt must be at least 10 characters.")
                continue
            if len(prompt) > 2000:
                _show_error(f"Prompt too long ({len(prompt)} chars). Max 2000.")
                continue

            adv = _prompt_advanced_settings()
            refresh = input("Force refresh market data cache? (y/N): ").strip().lower() == "y"

            try:
                code_str = _run_with_spinner(
                    "Generating strategy via Gemini LLM...",
                    generate_strategy_from_nl,
                    prompt,
                )
                run_strategy_pipeline(
                    code_str,
                    symbol=current_symbol,
                    period=current_period,
                    interval=current_interval,
                    force_refresh=refresh,
                    **adv,
                )
            except ConnectionError as e:
                _show_error(f"Network issue: {e}")
            except Exception as e:
                _show_error(f"Strategy generation failed: {e}")

        # ------------------------------------------------------------------
        # Option 2: Preset Strategies
        # ------------------------------------------------------------------
        elif choice == "2":
            print("\nPreset Strategies:")
            for item in PRESET_STRATEGIES:
                print(f"  [{item['id']}] {item['name']}")

            p_choice = input(f"\nSelect preset (1–{len(PRESET_STRATEGIES)}): ").strip()
            selected = next((p for p in PRESET_STRATEGIES if p["id"] == p_choice), None)

            if not selected:
                _show_error("Invalid preset selection.")
                continue

            print(f"\nRunning: {selected['name']}")
            adv = _prompt_advanced_settings()
            refresh = input("Force refresh market data cache? (y/N): ").strip().lower() == "y"

            run_strategy_pipeline(
                selected["code"],
                symbol=current_symbol,
                period=current_period,
                interval=current_interval,
                force_refresh=refresh,
                **adv,
            )

        # ------------------------------------------------------------------
        # Option 3: Change Instrument / Period / Interval
        # ------------------------------------------------------------------
        elif choice == "3":
            print("\nPopular Indian Market Symbols:")
            print("  ^NSEI        → Nifty 50 Index")
            print("  ^NSEBANK     → Bank Nifty")
            print("  RELIANCE.NS  → Reliance Industries")
            print("  INFY.NS      → Infosys")
            print("  TATAMOTORS.NS→ Tata Motors")

            sym = input(f"\nSymbol (current: {current_symbol}, Enter to keep): ").strip().upper()
            if sym and sym != current_symbol:
                is_valid, vmsg = validate_symbol(sym)
                if not is_valid:
                    _show_error(vmsg)
                else:
                    current_symbol = sym
                    _show_success(f"Symbol updated to '{current_symbol}'.")

            print("\nPeriod options: 1d 5d 1mo 3mo 6mo 1y 2y 5y")
            per = input(f"Period (current: {current_period}, Enter to keep): ").strip()
            if per:
                current_period = per

            print("\nInterval options: 1m 5m 15m 30m 1h 1d 1wk 1mo")
            ivl = input(f"Interval (current: {current_interval}, Enter to keep): ").strip()
            if ivl:
                current_interval = ivl

            print(f"\nUpdated → Symbol: {current_symbol} | Period: {current_period} | Interval: {current_interval}")

        # ------------------------------------------------------------------
        # Option 4: Config Status
        # ------------------------------------------------------------------
        elif choice == "4":
            key_status = "CONFIGURED ✔" if GEMINI_API_KEY else "MISSING ✗ (add to .env)"
            print(f"\n  Gemini API Key  : {key_status}")
            print(f"  Symbol          : {current_symbol}")
            print(f"  Period          : {current_period}")
            print(f"  Interval        : {current_interval}")
            print(f"  Brokerage       : ₹{COST_MODEL['brokerage_per_trade']}/trade")
            print(f"  Slippage        : {COST_MODEL['slippage_pct']*100}%")
            print(f"  Risk-Free Rate  : {RISK_FREE_RATE*100}%")
            print(f"  Position Size   : {int(DEFAULT_POSITION_SIZE*100)}% of capital")
            print(f"  Results Saved To: backtest_results/")

        # ------------------------------------------------------------------
        # Option 5: Exit
        # ------------------------------------------------------------------
        elif choice == "5":
            print("Exiting studio.trade CLI. Good trading!")
            break

        else:
            print("Invalid choice. Enter 1–5.")


if __name__ == "__main__":
    main()
