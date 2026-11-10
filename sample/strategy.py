"""Transparent daily-bar EMA crossover engine adapted from sample 1."""
from __future__ import annotations
from dataclasses import dataclass
import math
from datetime import datetime
import pandas as pd

@dataclass(frozen=True)
class Config:
    fast_length: int = 20
    slow_length: int = 50
    stop_percent: float = 0.05
    commission_rate: float = 0.001
    initial_cash: float = 10_000.0

def indicators(bars: pd.DataFrame, config: Config) -> pd.DataFrame:
    out = bars.copy()
    out["fast_ema"] = out.close.ewm(span=config.fast_length, adjust=False).mean()
    out["slow_ema"] = out.close.ewm(span=config.slow_length, adjust=False).mean()
    pf, ps = out.fast_ema.shift(), out.slow_ema.shift()
    out["cross_up"] = (out.fast_ema > out.slow_ema) & (pf <= ps)
    out["cross_down"] = (out.fast_ema < out.slow_ema) & (pf >= ps)
    return out

def backtest(bars: pd.DataFrame, config: Config, signal_bars: pd.DataFrame | None = None):
    """Signals act next open; stops use OHLC and activate after entry bar."""
    df = (signal_bars if signal_bars is not None else indicators(bars, config)).reset_index(drop=True)
    cash, qty, stop, pending, active = config.initial_cash, 0.0, None, None, None
    curve, trades = [], []
    for i, row in enumerate(df.itertuples(index=False)):
        date = row.date
        if pending == "entry" and qty == 0:
            price = float(row.open); qty = cash / price
            fee = qty * price * config.commission_rate; cash -= qty * price + fee
            stop = price * (1 - config.stop_percent)
            active = dict(entry_date=date, entry_price=price, quantity=qty, entry_fee=fee)
            pending = None
        elif pending == "exit" and qty > 0:
            price = float(row.open); fee = qty * price * config.commission_rate
            cash += qty * price - fee
            trades.append({**active, "exit_date": date, "exit_price": price,
                           "exit_fee": fee, "exit_reason": "crossunder"})
            qty, stop, active, pending = 0.0, None, None, None
        if qty > 0 and active and active["entry_date"] != date and (row.open <= stop or row.low <= stop):
            price = min(float(row.open), stop); fee = qty * price * config.commission_rate
            cash += qty * price - fee
            trades.append({**active, "exit_date": date, "exit_price": price,
                           "exit_fee": fee, "exit_reason": "stop"})
            qty, stop, active, pending = 0.0, None, None, None
        if qty == 0 and bool(row.cross_up) and i + 1 < len(df): pending = "entry"
        elif qty > 0 and bool(row.cross_down) and i + 1 < len(df): pending = "exit"
        curve.append({"date": date, "equity": cash + qty * float(row.close), "cash": cash, "position_qty": qty})
    return pd.DataFrame(curve), pd.DataFrame(trades)

def performance(equity: pd.DataFrame, trades: pd.DataFrame, initial_cash=10_000.0):
    values = equity.equity.astype(float); daily = values.pct_change().dropna()
    total = values.iloc[-1] / initial_cash - 1 if len(values) else 0.0
    first, last = equity.date.iloc[0], equity.date.iloc[-1]
    first = first.to_pydatetime() if hasattr(first, "to_pydatetime") else datetime.fromisoformat(str(first))
    last = last.to_pydatetime() if hasattr(last, "to_pydatetime") else datetime.fromisoformat(str(last))
    years = max((last-first).days/365.25, 1/365.25)
    dd = values / values.cummax() - 1
    sharpe = daily.mean()/daily.std(ddof=1)*math.sqrt(365) if len(daily)>1 and daily.std(ddof=1) else 0.0
    return {"total_return": total, "annualized_return": (values.iloc[-1]/initial_cash)**(1/years)-1 if values.iloc[-1]>0 else -1,
            "max_drawdown": float(dd.min()), "sharpe": float(sharpe), "trade_count": len(trades),
            "final_equity": float(values.iloc[-1])}
