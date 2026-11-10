"""Parameter grid, expanding-history walk-forward, and robustness metrics."""
from __future__ import annotations
from dataclasses import asdict
import numpy as np
import pandas as pd
from .strategy import Config, backtest, indicators, performance

FAST_VALUES = tuple(range(5, 51, 5))
SLOW_VALUES = tuple(range(20, 201, 10))
STOP_VALUES = tuple(x / 100 for x in range(2, 11))

def parameter_grid():
    return [Config(fast_length=f, slow_length=s, stop_percent=p)
            for f in FAST_VALUES for s in SLOW_VALUES if f < s for p in STOP_VALUES]

def rank_grid(bars, configs=None):
    rows = []
    for c in configs or parameter_grid():
        e, t = backtest(bars, c); m = performance(e, t, c.initial_cash)
        rows.append({**asdict(c), **m})
    return pd.DataFrame(rows).sort_values(["sharpe", "total_return"], ascending=False).reset_index(drop=True)

def split_windows(bars, in_days=365, out_days=90, step_days=90):
    """Calendar-day windows; each OOS starts after its own IS ends."""
    dates = pd.to_datetime(bars.date).reset_index(drop=True)
    start = dates.iloc[0]
    windows = []
    while True:
        train_end = start + pd.Timedelta(days=in_days)
        test_end = train_end + pd.Timedelta(days=out_days)
        if test_end > dates.iloc[-1] + pd.Timedelta(days=1): break
        train_idx = np.flatnonzero((dates >= start) & (dates < train_end))
        test_idx = np.flatnonzero((dates >= train_end) & (dates < test_end))
        if len(train_idx) == 0 or len(test_idx) == 0: break
        windows.append((train_idx, test_idx))
        start += pd.Timedelta(days=step_days)
    return windows

def stitch_oos_equity(pieces, initial_cash=10_000.0):
    """Scale each independently simulated OOS slice by prior carried wealth."""
    stitched = []
    for piece in pieces:
        part = piece.copy()
        factor = float(stitched[-1].equity.iloc[-1]) / initial_cash if stitched else 1.0
        part["equity"] *= factor
        stitched.append(part)
    return pd.concat(stitched, ignore_index=True) if stitched else pd.DataFrame(columns=["date", "equity", "cash", "position_qty"])

def walk_forward(bars, configs=None):
    configs = configs or parameter_grid(); windows = split_windows(bars)
    records, pieces, oos_returns, is_returns = [], [], [], []
    for number, (train_idx, test_idx) in enumerate(windows, 1):
        train, test = bars.iloc[train_idx].reset_index(drop=True), bars.iloc[test_idx].reset_index(drop=True)
        ranked = rank_grid(train, configs)
        best = Config(fast_length=int(ranked.iloc[0].fast_length), slow_length=int(ranked.iloc[0].slow_length),
                      stop_percent=float(ranked.iloc[0].stop_percent))
        # Train history warms up EMA; the OOS engine starts flat with fresh capital.
        signals = indicators(pd.concat([train, test], ignore_index=True), best).iloc[-len(test):].reset_index(drop=True)
        out_eq, t = backtest(test, best, signal_bars=signals)
        out_stats = performance(out_eq, t, best.initial_cash)
        pieces.append(out_eq)
        in_e, in_t = backtest(train, best); in_stats = performance(in_e, in_t, best.initial_cash)
        oos_returns.append(out_eq.equity.pct_change().dropna().to_numpy())
        is_returns.append(in_e.equity.pct_change().dropna().to_numpy())
        records.append({"window": number, "train_start": str(train.date.iloc[0]), "train_end": str(train.date.iloc[-1]),
                        "test_start": str(test.date.iloc[0]), "test_end": str(test.date.iloc[-1]), **asdict(best),
                        "in_sample_sharpe": in_stats["sharpe"], "in_sample_return": in_stats["total_return"],
                        "out_sample_sharpe": out_stats["sharpe"], "out_sample_return": out_stats["total_return"],
                        "efficiency": out_stats["sharpe"] / in_stats["sharpe"] if in_stats["sharpe"] else np.nan})
    curve = stitch_oos_equity(pieces, configs[0].initial_cash if configs else 10_000.0)
    result = pd.DataFrame(records)
    result.attrs["oos_returns"] = oos_returns
    result.attrs["is_returns"] = is_returns
    return result, curve

def stitched_sharpe_summary(windows):
    """Compare Sharpe on concatenated OOS and IS daily returns, avoiding per-window ratios."""
    if windows.empty:
        return {"stitched_oos_sharpe": np.nan, "stitched_is_sharpe": np.nan,
                "stitched_sharpe_efficiency": np.nan}
    # walk_forward attaches exact equity-return streams before serialization.
    oos = np.concatenate([np.asarray(x, dtype=float) for x in windows.attrs.get("oos_returns", [])]) if windows.attrs.get("oos_returns") else np.array([])
    ins = np.concatenate([np.asarray(x, dtype=float) for x in windows.attrs.get("is_returns", [])]) if windows.attrs.get("is_returns") else np.array([])
    def sharpe(r):
        return float(r.mean()/r.std(ddof=1)*np.sqrt(365)) if len(r)>1 and r.std(ddof=1)>0 else np.nan
    so, si = sharpe(oos), sharpe(ins)
    return {"stitched_oos_sharpe": so, "stitched_is_sharpe": si,
            "stitched_sharpe_efficiency": so / si if pd.notna(si) and abs(si) > 1e-12 else np.nan}

def robustness(grid, windows):
    if grid.empty: return {"plateau": "unavailable", "plateau_points": 0, "neighbor_decay": np.nan, "parameter_drift": np.nan}
    best = grid.iloc[0]; level = grid.sharpe.quantile(.9)
    close = grid[((grid.fast_length-best.fast_length).abs() <= 5) & ((grid.slow_length-best.slow_length).abs() <= 10) & ((grid.stop_percent-best.stop_percent).abs() <= .01)]
    # A broad local plateau has at least half the available adjacent grid points
    # within 10% of the optimum; this explicit rule is reported with the result.
    plateau_points = int((close.sharpe >= best.sharpe * .9).sum())
    plateau = "platform" if len(close) > 1 and plateau_points >= (len(close)+1)//2 else "isolated peak"
    decay = 1 - close.sharpe.mean()/best.sharpe if best.sharpe else np.nan
    if len(windows) > 1:
        fast_drift = float(np.sqrt((windows.fast_length.diff().dropna()**2).mean()))
        slow_drift = float(np.sqrt((windows.slow_length.diff().dropna()**2).mean()))
        stop_drift = float(np.sqrt((windows.stop_percent.diff().dropna()**2).mean()))
    else:
        fast_drift = slow_drift = stop_drift = np.nan
    return {"plateau": plateau, "plateau_points": plateau_points, "neighbor_decay": float(decay),
            "fast_period_drift": fast_drift, "slow_period_drift": slow_drift, "stop_drift": stop_drift}
