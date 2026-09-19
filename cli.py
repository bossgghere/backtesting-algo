import sys
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Import internal modules
from config import GEMINI_API_KEY, DEFAULT_SYMBOL, DEFAULT_PERIOD, DEFAULT_INTERVAL, COST_MODEL
from data_loader import get_market_data
from validator import validate_strategy_code, get_strategy_fingerprint
from generator import generate_strategy_from_nl
from backtester import run_backtest_simulation
from presets import PRESET_STRATEGIES

# Try importing rich for enhanced CLI output
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

def print_header():
    title = "studio.trade — Universal Algo Trading Platform CLI"
    subtitle = "Natural Language Strategy Authoring, AST Validation & Real Market Backtesting"
    if USE_RICH:
        console.print(Panel(f"[bold green]{title}[/bold green]\n[dim]{subtitle}[/dim]", border_style="cyan"))
    else:
        print("=" * 70)
        print(f" {title}")
        print(f" {subtitle}")
        print("=" * 70)

def display_code(code_str: str):
    if USE_RICH:
        syntax = Syntax(code_str, "python", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title="Generated Strategy Code (AST Verified)", border_style="green"))
    else:
        print("\n--- GENERATED STRATEGY CODE ---")
        print(code_str)
        print("-------------------------------\n")

def display_backtest_results(results: dict, symbol: str, fingerprint: str):
    summary = results["summary"]
    trades = results["trades"]

    if USE_RICH:
        # Create summary table
        table = Table(title=f"📊 Backtest Performance Summary ({symbol}) | Fingerprint: {fingerprint}", border_style="blue")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="bold yellow")
        table.add_column("Benchmark / Notes", style="dim")

        table.add_row("Initial Capital", f"₹{summary['initial_capital']:,.2f}", "Starting account balance")
        table.add_row("Final Capital", f"₹{summary['final_capital']:,.2f}", f"Net PnL: ₹{(summary['final_capital'] - summary['initial_capital']):,.2f}")
        
        ret_color = "green" if summary['total_return_pct'] >= 0 else "red"
        table.add_row("Total Strategy Return", f"[{ret_color}]{summary['total_return_pct']}%[/{ret_color}]", f"Buy & Hold: {summary['buy_and_hold_pct']}%")
        
        table.add_row("Win Rate", f"{summary['win_rate_pct']}%", f"{summary['winning_trades']} Wins / {summary['losing_trades']} Losses out of {summary['total_trades']} Trades")
        
        dd_color = "red" if summary['max_drawdown_pct'] < -10 else "yellow"
        table.add_row("Max Drawdown", f"[{dd_color}]{summary['max_drawdown_pct']}%[/{dd_color}]", "Peak to trough peak risk")
        table.add_row("Profit Factor", f"{summary['profit_factor']}", "Gross Profit / Gross Loss")
        table.add_row("Sharpe Ratio", f"{summary['sharpe_ratio']}", "Risk-adjusted return ratio")

        console.print(table)

        # Recent Trades Table
        if trades:
            t_table = Table(title=f"📜 Trade Executions Log (Last {min(5, len(trades))} Trades)", border_style="magenta")
            t_table.add_column("#", style="dim")
            t_table.add_column("Type", style="bold")
            t_table.add_column("Entry Date")
            t_table.add_column("Exit Date")
            t_table.add_column("Entry ₹")
            t_table.add_column("Exit ₹")
            t_table.add_column("PnL %")
            t_table.add_column("Net PnL ₹")

            for t in trades[-5:]:
                p_color = "green" if t['net_pnl'] >= 0 else "red"
                t_table.add_row(
                    str(t['trade_num']),
                    t['type'],
                    t['entry_date'][:10],
                    t['exit_date'][:10],
                    f"₹{t['entry_price']}",
                    f"₹{t['exit_price']}",
                    f"[{p_color}]{t['pnl_pct']}%[/{p_color}]",
                    f"[{p_color}]₹{t['net_pnl']:.2f}[/{p_color}]"
                )
            console.print(t_table)
        else:
            console.print("[bold yellow]No trades triggered during this market period.[/bold yellow]")

    else:
        print("\n=== BACKTEST RESULTS ===")
        print(f"Symbol: {symbol} | Fingerprint: {fingerprint}")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print(f"Total Trades Logged: {len(trades)}")

def run_strategy_pipeline(code_str: str, symbol: str = "^NSEI", period: str = "1y"):
    """
    Full processing pipeline: AST Safety Check -> Code Execution -> Data Load -> Backtest Simulation.
    """
    # 1. AST Safety Check
    if USE_RICH:
        console.print("\n[bold cyan]1. Running AST Safety Check...[/bold cyan]")
    else:
        print("\n1. Running AST Safety Check...")

    is_safe, msg = validate_strategy_code(code_str)
    if not is_safe:
        if USE_RICH:
            console.print(f"[bold red]❌ AST Validation Failed:[/bold red]\n{msg}")
        else:
            print(f"❌ AST Validation Failed:\n{msg}")
        return

    fingerprint = get_strategy_fingerprint(code_str)
    if USE_RICH:
        console.print(f"[bold green]✔ AST Check Passed![/bold green] Strategy Fingerprint: [yellow]{fingerprint}[/yellow]")
    else:
        print(f"✔ AST Check Passed! Strategy Fingerprint: {fingerprint}")

    display_code(code_str)

    # 2. Compile code into executable namespace
    exec_scope = {}
    try:
        exec(code_str, exec_scope)
        generate_signals_fn = exec_scope.get("generate_signals")
    except Exception as e:
        print(f"❌ Code Execution Compilation Error: {e}")
        return

    # 3. Load Real Market Data
    try:
        df = get_market_data(symbol=symbol, period=period, interval="1d")
    except Exception as e:
        print(f"❌ Market Data Fetch Error: {e}")
        return

    # 4. Execute Backtest Simulation
    if USE_RICH:
        console.print(f"\n[bold cyan]2. Running Backtest on {symbol} market data ({len(df)} candles)...[/bold cyan]")
    else:
        print(f"\n2. Running Backtest on {symbol} market data ({len(df)} candles)...")

    results = run_backtest_simulation(
        df=df,
        generate_signals_fn=generate_signals_fn,
        params={},
        initial_capital=100000.0,
        brokerage_per_trade=COST_MODEL["brokerage_per_trade"],
        slippage_pct=COST_MODEL["slippage_pct"],
        stt_tax_pct=COST_MODEL["stt_tax_pct"]
    )

    # 5. Display Performance Metrics & Trade Log
    display_backtest_results(results, symbol, fingerprint)


def main():
    print_header()
    current_symbol = DEFAULT_SYMBOL

    while True:
        print("\n---------------------------------------------------------")
        print(f"Current Asset Instrument: [ {current_symbol} ]")
        print("---------------------------------------------------------")
        print("Select an option:")
        print(" 1. 🤖 Author Strategy from Natural Language Prompt (Gemini LLM)")
        print(" 2. ⚡ Run Preset Strategies (Instant Real Data Backtest)")
        print(" 3. 📈 Change Market Instrument (e.g., ^NSEI, ^NSEBANK, RELIANCE.NS, INFY.NS)")
        print(" 4. 🔑 API Key Status Check")
        print(" 5. 🚪 Exit")
        print("---------------------------------------------------------")

        try:
            choice = input("Enter choice (1-5): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            sys.exit(0)

        if choice == "1":
            print("\nDescribe your trading strategy idea in plain English.")
            print("Examples:")
            print(" - 'Buy Nifty when RSI 14 drops below 30, exit when RSI exceeds 65'")
            print(" - 'Buy when 20 EMA is above 50 EMA and volume is above 20-day average'")
            prompt = input("\nYour Prompt: ").strip()

            if not prompt:
                print("Prompt cannot be empty.")
                continue

            print("\nSending prompt to Gemini LLM for strategy code generation...")
            try:
                code_str = generate_strategy_from_nl(prompt)
                run_strategy_pipeline(code_str, symbol=current_symbol)
            except Exception as e:
                print(f"\n❌ Strategy Generation Failed: {e}")

        elif choice == "2":
            print("\nPreset Strategies:")
            for item in PRESET_STRATEGIES:
                print(f" [{item['id']}] {item['name']}")
                print(f"     Prompt: {item['prompt']}")
            
            p_choice = input("\nSelect preset number (1-3): ").strip()
            selected = next((p for p in PRESET_STRATEGIES if p["id"] == p_choice), None)
            
            if selected:
                print(f"\nExecuting preset strategy: {selected['name']}")
                run_strategy_pipeline(selected["code"], symbol=current_symbol)
            else:
                print("Invalid preset choice.")

        elif choice == "3":
            print("\nPopular Indian Market Symbols:")
            print(" - ^NSEI        : Nifty 50 Index")
            print(" - ^NSEBANK     : Bank Nifty Index")
            print(" - RELIANCE.NS  : Reliance Industries")
            print(" - INFY.NS      : Infosys")
            print(" - TATAMOTORS.NS: Tata Motors")
            sym = input(f"\nEnter symbol name (default {current_symbol}): ").strip().upper()
            if sym:
                current_symbol = sym
                print(f"Active symbol updated to '{current_symbol}'.")

        elif choice == "4":
            key_status = "CONFIGURED ✔" if GEMINI_API_KEY else "MISSING ❌ (Add to .env file)"
            print(f"\nGemini API Key: {key_status}")
            print(f"Default Asset : {current_symbol}")
            print(f"Cost Model    : Brokerage ₹{COST_MODEL['brokerage_per_trade']}/trade, Slippage {COST_MODEL['slippage_pct']*100}%")

        elif choice == "5":
            print("Exiting studio.trade CLI. Have a great trading day!")
            break

        else:
            print("Invalid choice. Please enter 1 to 5.")

if __name__ == "__main__":
    main()
