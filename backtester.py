import math
import numpy as np
import pandas as pd


def run_backtest_simulation(
    df, generate_signals_fn, params=None, initial_capital=100000.0,
    brokerage_per_trade=20.0, slippage_pct=0.0005, stt_tax_pct=0.00025,
    *, quantity=None, lot_size=1, periods_per_year=252,
):
    """Long-only, single-instrument simulator.

    Signals are events: +1 buy if flat, -1 sell if long, 0 hold.
    Signals formed on a completed bar execute at the NEXT bar's open.
    Tax is a configurable sell-turnover charge, not a full statutory cost model.
    Open positions are marked at the final close, without hypothetical exit fees.
    Strategies must be causal; shifting signals cannot fix future-data use inside
    a strategy. No derivative margin, expiry, borrowing, or liquidity simulation.
    """
    settings = [initial_capital, brokerage_per_trade, slippage_pct, stt_tax_pct, periods_per_year]
    if not all(np.isfinite(x) for x in settings):
        raise ValueError('Settings must be finite.')
    if initial_capital <= 0 or brokerage_per_trade < 0 or not 0 <= slippage_pct < 1 or not 0 <= stt_tax_pct < 1 or periods_per_year <= 0:
        raise ValueError('Invalid capital, costs, slippage, or annualization setting.')
    if isinstance(lot_size, bool) or not isinstance(lot_size, (int, np.integer)) or lot_size <= 0:
        raise ValueError('lot_size must be a positive integer.')
    if quantity is not None and (isinstance(quantity, bool) or not isinstance(quantity, (int, np.integer)) or quantity <= 0 or quantity % lot_size):
        raise ValueError('quantity must be a positive integer multiple of lot_size.')
    if df.empty or not {'open', 'close'}.issubset(df.columns):
        raise ValueError('Nonempty data with open and close columns is required.')
    if df.index.has_duplicates or df.index.hasnans or not df.index.is_monotonic_increasing:
        raise ValueError('Data index must be unique, nonmissing, and increasing.')
    data = df.copy(deep=True)
    prices = data[['open', 'close']].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError('Open and close prices must be finite and positive.')
    try:
        signals = generate_signals_fn(data.copy(deep=True), {} if params is None else dict(params))
        if not isinstance(signals, pd.Series):
            signals = pd.Series(signals, index=data.index)
    except Exception as exc:
        raise RuntimeError(f'Error while generating signals: {exc}') from exc
    if not signals.index.equals(data.index):
        raise ValueError('Signals must have the exact market-data index.')
    if not signals.isin([-1, 0, 1]).all():
        raise ValueError('Signals must contain only -1, 0, or 1; missing values are invalid.')

    cash = float(initial_capital)
    units = 0
    entry_price = entry_fee = 0.0
    entry_date = None
    trades, equity, rejected_orders = [], [], []
    for i, date in enumerate(data.index):
        signal = int(signals.iloc[i - 1]) if i else 0
        opening = float(data['open'].iloc[i])
        if signal == 1 and units == 0:
            fill = opening * (1 + slippage_pct)
            affordable = max(0, math.floor((cash - brokerage_per_trade) / (fill * lot_size))) * lot_size
            requested = affordable if quantity is None else quantity
            if requested <= 0 or requested > affordable:
                rejected_orders.append({'date': str(date), 'reason': 'Insufficient cash for requested quantity and entry fee'})
            else:
                units = int(requested)
                entry_price, entry_fee, entry_date = fill, float(brokerage_per_trade), date
                cash -= units * entry_price + entry_fee
        elif signal == -1 and units:
            fill = opening * (1 - slippage_pct)
            exit_fee = brokerage_per_trade + units * fill * stt_tax_pct
            gross = units * (fill - entry_price)
            net = gross - entry_fee - exit_fee
            cash += units * fill - exit_fee
            trades.append({
                'trade_num': len(trades) + 1, 'type': 'LONG', 'quantity': units,
                'entry_date': str(entry_date), 'exit_date': str(date),
                'entry_price': entry_price, 'exit_price': fill,
                'gross_pnl': gross, 'entry_costs': entry_fee, 'exit_costs': exit_fee,
                'net_pnl': net, 'pnl_pct': net / (units * entry_price + entry_fee) * 100,
                'capital_after': cash,
            })
            units = 0
        equity.append(cash + units * float(data['close'].iloc[i]))

    equity_series = pd.Series(equity, index=data.index, dtype=float)
    # Include starting capital so the first loss cannot disappear from drawdown.
    values = np.r_[initial_capital, equity_series.to_numpy()]
    peaks = np.maximum.accumulate(values)
    drawdown = (values - peaks) / peaks
    returns = pd.Series(values).pct_change().iloc[1:]
    std = returns.std(ddof=1)
    sharpe = float(returns.mean() / std * np.sqrt(periods_per_year)) if len(returns) > 1 and std > 0 else None
    wins = [t for t in trades if t['net_pnl'] > 0]
    losses = [t for t in trades if t['net_pnl'] < 0]
    total_profits = sum(t['net_pnl'] for t in wins)
    total_losses = -sum(t['net_pnl'] for t in losses)
    profit_factor = total_profits / total_losses if total_losses else (float('inf') if total_profits else None)
    open_position = None
    if units:
        last = float(data['close'].iloc[-1])
        open_position = {
            'quantity': units, 'entry_date': str(entry_date), 'entry_price': entry_price,
            'mark_price': last, 'market_value': units * last,
            'unrealized_pnl': units * (last - entry_price),
            'net_pnl_to_date': units * (last - entry_price) - entry_fee,
        }
    final = equity[-1]
    return {
        'summary': {
            'initial_capital': initial_capital, 'final_capital': final, 'cash_balance': cash,
            'total_return_pct': (final / initial_capital - 1) * 100,
            'buy_and_hold_pct': (float(data['close'].iloc[-1]) / float(data['close'].iloc[0]) - 1) * 100,
            'win_rate_pct': len(wins) / len(trades) * 100 if trades else 0.0,
            'max_drawdown_pct': float(drawdown.min()) * 100,
            'profit_factor': profit_factor, 'sharpe_ratio': sharpe,
            'total_trades': len(trades), 'winning_trades': len(wins),
            'losing_trades': len(losses), 'breakeven_trades': len(trades) - len(wins) - len(losses),
            'open_positions': int(bool(units)),
            'expectancy': sum(t['net_pnl'] for t in trades) / len(trades) if trades else None,
        },
        'trades': trades, 'open_position': open_position, 'rejected_orders': rejected_orders,
        'equity_curve': data[['close']].assign(equity=equity_series),
    }
