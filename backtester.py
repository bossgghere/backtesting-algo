import pandas as pd
import numpy as np

from config import RISK_FREE_RATE


def run_backtest_simulation(
    df: pd.DataFrame,
    generate_signals_fn,
    params: dict = None,
    initial_capital: float = 100000.0,
    brokerage_per_trade: float = 20.0,
    slippage_pct: float = 0.0005,
    stt_tax_pct: float = 0.00025,
    risk_free_rate: float = RISK_FREE_RATE,
    stop_loss_pct: float = None,       # e.g. 0.03 = exit if trade drops 3%
    take_profit_pct: float = None,     # e.g. 0.08 = exit if trade gains 8%
    position_size_pct: float = 1.0,   # fraction of cash to deploy per trade (0–1)
) -> dict:
    """
    Strategy-agnostic bar-by-bar backtesting engine for studio.trade.

    Signals from generate_signals_fn:
        +1  →  Long entry
        -1  →  Short entry / exit long
         0  →  Hold / no position

    Returns a dict with keys: 'summary', 'trades', 'equity_curve'.
    """
    if params is None:
        params = {}

    position_size_pct = max(0.01, min(float(position_size_pct), 1.0))  # clamp 1%–100%

    df_copy = df.copy()

    # Run strategy code to produce signals
    try:
        raw_signals = generate_signals_fn(df_copy, params)
        if not isinstance(raw_signals, pd.Series):
            raw_signals = pd.Series(raw_signals, index=df_copy.index)
    except Exception as e:
        raise RuntimeError(f"Runtime error executing strategy code: {e}")

    df_copy["signal"] = raw_signals.fillna(0).astype(int)

    # Shift by 1 bar — trade executes on the *next* bar to prevent lookahead bias
    df_copy["pos"] = df_copy["signal"].shift(1).fillna(0)

    if df_copy["signal"].abs().sum() == 0:
        print("[Warning] Strategy produced no buy/sell signals for this period.")

    trades = []
    cash = initial_capital
    equity_curve = []

    current_pos = 0       # 0 = flat, 1 = long, -1 = short
    entry_price = 0.0
    entry_date = None
    capital_deployed = 0.0  # actual ₹ put into the trade (cash × position_size_pct)

    for i in range(len(df_copy)):
        date = df_copy.index[i]
        price = df_copy["close"].iloc[i]
        target_pos = int(df_copy["pos"].iloc[i])
        exit_reason = "SIGNAL"

        # ---------------------------------------------------------------
        # Stop-loss / Take-profit check (overrides signal if hit)
        # ---------------------------------------------------------------
        if current_pos != 0 and (stop_loss_pct is not None or take_profit_pct is not None):
            if current_pos > 0:
                current_pnl_pct = (price - entry_price) / entry_price
            else:
                current_pnl_pct = (entry_price - price) / entry_price

            sl_hit = stop_loss_pct is not None and current_pnl_pct <= -abs(stop_loss_pct)
            tp_hit = take_profit_pct is not None and current_pnl_pct >= abs(take_profit_pct)

            if sl_hit or tp_hit:
                target_pos = 0  # Force flat regardless of signal
                exit_reason = "STOP_LOSS" if sl_hit else "TAKE_PROFIT"

        # ---------------------------------------------------------------
        # Position change handling
        # ---------------------------------------------------------------
        if target_pos != current_pos:

            # Close existing position
            if current_pos != 0:
                exit_price = (
                    price * (1 - slippage_pct) if current_pos > 0
                    else price * (1 + slippage_pct)
                )
                pnl_pct = (
                    (exit_price - entry_price) / entry_price if current_pos > 0
                    else (entry_price - exit_price) / entry_price
                )

                trade_gross = capital_deployed * pnl_pct
                # STT is charged on trade notional at exit, not on total account balance
                exit_notional = capital_deployed * abs(1 + pnl_pct)
                trade_costs = brokerage_per_trade + (exit_notional * stt_tax_pct)
                trade_net = trade_gross - trade_costs

                cash += capital_deployed + trade_net  # return deployed capital + PnL
                capital_deployed = 0.0

                trades.append({
                    "trade_num":     len(trades) + 1,
                    "type":          "LONG" if current_pos > 0 else "SHORT",
                    "entry_date":    str(entry_date)[:10],
                    "exit_date":     str(date)[:10],
                    "entry_price":   round(entry_price, 2),
                    "exit_price":    round(exit_price, 2),
                    "pnl_pct":       round(pnl_pct * 100, 2),
                    "net_pnl":       round(trade_net, 2),
                    "capital_after": round(cash, 2),
                    "exit_reason":   exit_reason,
                })

            # Open new position
            if target_pos != 0:
                entry_price = (
                    price * (1 + slippage_pct) if target_pos > 0
                    else price * (1 - slippage_pct)
                )
                entry_date = date
                capital_deployed = cash * position_size_pct
                cash -= capital_deployed   # remove deployed capital from free cash
                cash -= brokerage_per_trade  # deduct entry brokerage

            current_pos = target_pos

        # ---------------------------------------------------------------
        # Mark-to-market equity tracking
        # ---------------------------------------------------------------
        if current_pos != 0:
            unrealized_pnl = capital_deployed * (
                (price - entry_price) / entry_price if current_pos > 0
                else (entry_price - price) / entry_price
            )
            current_equity = cash + capital_deployed + unrealized_pnl
        else:
            current_equity = cash

        equity_curve.append(current_equity)

    # Close any position still open at end of data
    if current_pos != 0:
        last_price = df_copy["close"].iloc[-1]
        exit_price = (
            last_price * (1 - slippage_pct) if current_pos > 0
            else last_price * (1 + slippage_pct)
        )
        pnl_pct = (
            (exit_price - entry_price) / entry_price if current_pos > 0
            else (entry_price - exit_price) / entry_price
        )
        trade_gross = capital_deployed * pnl_pct
        exit_notional = capital_deployed * abs(1 + pnl_pct)
        trade_costs = brokerage_per_trade + (exit_notional * stt_tax_pct)
        trade_net = trade_gross - trade_costs
        cash += capital_deployed + trade_net

        trades.append({
            "trade_num":     len(trades) + 1,
            "type":          "LONG" if current_pos > 0 else "SHORT",
            "entry_date":    str(entry_date)[:10],
            "exit_date":     str(df_copy.index[-1])[:10],
            "entry_price":   round(entry_price, 2),
            "exit_price":    round(exit_price, 2),
            "pnl_pct":       round(pnl_pct * 100, 2),
            "net_pnl":       round(trade_net, 2),
            "capital_after": round(cash, 2),
            "exit_reason":   "END_OF_DATA",
        })

    df_copy["equity"] = equity_curve

    # -------------------------------------------------------------------
    # Performance Metrics
    # -------------------------------------------------------------------
    total_return_pct = ((cash - initial_capital) / initial_capital) * 100
    buy_and_hold_pct = (
        (df_copy["close"].iloc[-1] - df_copy["close"].iloc[0])
        / df_copy["close"].iloc[0]
    ) * 100

    # Drawdown
    equity_series = pd.Series(equity_curve, index=df_copy.index)
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    max_drawdown_pct = float(drawdown.min()) * 100

    # Win / Loss split
    winning_trades = [t for t in trades if t["net_pnl"] > 0]
    losing_trades  = [t for t in trades if t["net_pnl"] <= 0]
    win_rate_pct   = (len(winning_trades) / len(trades) * 100) if trades else 0.0

    gross_profits = sum(t["net_pnl"] for t in winning_trades)
    gross_losses  = abs(sum(t["net_pnl"] for t in losing_trades))
    profit_factor = (
        (gross_profits / gross_losses) if gross_losses > 0
        else (gross_profits if gross_profits > 0 else 1.0)
    )

    # Sharpe Ratio — properly subtract daily risk-free rate
    daily_returns = equity_series.pct_change().dropna()
    mean_ret  = daily_returns.mean()
    std_ret   = daily_returns.std()
    rf_daily  = (1 + risk_free_rate) ** (1 / 252) - 1
    sharpe    = (
        float(((mean_ret - rf_daily) / (std_ret + 1e-10)) * np.sqrt(252))
        if std_ret > 0 else 0.0
    )

    # Average win / loss in ₹
    avg_win  = (gross_profits / len(winning_trades)) if winning_trades else 0.0
    avg_loss = (gross_losses  / len(losing_trades))  if losing_trades  else 0.0

    # Max consecutive losses
    max_consec_losses = current_consec = 0
    for t in trades:
        if t["net_pnl"] <= 0:
            current_consec += 1
            max_consec_losses = max(max_consec_losses, current_consec)
        else:
            current_consec = 0

    # Average holding period in days
    if trades:
        holding_days = []
        for t in trades:
            try:
                diff = (pd.Timestamp(t["exit_date"]) - pd.Timestamp(t["entry_date"])).days
                holding_days.append(diff)
            except Exception:
                holding_days.append(0)
        avg_holding_days = round(sum(holding_days) / len(holding_days), 1)
    else:
        avg_holding_days = 0.0

    return {
        "summary": {
            "initial_capital":        round(initial_capital, 2),
            "final_capital":          round(cash, 2),
            "total_return_pct":       round(total_return_pct, 2),
            "buy_and_hold_pct":       round(buy_and_hold_pct, 2),
            "win_rate_pct":           round(win_rate_pct, 2),
            "max_drawdown_pct":       round(max_drawdown_pct, 2),
            "profit_factor":          round(profit_factor, 2),
            "sharpe_ratio":           round(sharpe, 2),
            "total_trades":           len(trades),
            "winning_trades":         len(winning_trades),
            "losing_trades":          len(losing_trades),
            "avg_win_inr":            round(avg_win, 2),
            "avg_loss_inr":           round(avg_loss, 2),
            "max_consecutive_losses": max_consec_losses,
            "avg_holding_days":       avg_holding_days,
        },
        "trades":       trades,
        "equity_curve": df_copy[["close", "equity"]].copy(),
    }
