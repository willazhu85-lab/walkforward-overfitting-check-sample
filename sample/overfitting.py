"""Public-literature PBO/CSCV and Deflated Sharpe Ratio calculations."""
from __future__ import annotations

from itertools import combinations
import math

import numpy as np
import pandas as pd


def daily_sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1) if r.size > 1 else 0.0
    return float(r.mean() / sd) if sd > 0 else 0.0


def grid_daily_returns(bars: pd.DataFrame, configs) -> np.ndarray:
    """Return T x N daily equity returns from the same backtest engine as ranking."""
    from .strategy import backtest
    columns = []
    for config in configs:
        equity, _ = backtest(bars, config)
        columns.append(equity.equity.astype(float).pct_change().fillna(0).to_numpy())
    return np.column_stack(columns)


def cscv_pbo(returns: np.ndarray, segments: int = 16) -> dict:
    """Compute CSCV PBO; rows are dates and columns are tested strategies.

    Ties use average ranks. Omega is rank/(N+1), hence a complete tie has
    omega=0.5 and lambda=0 (counted as overfit by the specified <= 0 rule).
    """
    values = np.asarray(returns, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("returns must be a T x N matrix with N >= 2")
    if segments < 2 or segments % 2:
        raise ValueError("segments must be an even integer >= 2")
    if values.shape[0] < segments:
        raise ValueError("need at least one observation per segment")
    pieces = np.array_split(np.arange(len(values)), segments)
    sums = np.stack([values[ix].sum(axis=0) for ix in pieces])
    squares = np.stack([(values[ix] ** 2).sum(axis=0) for ix in pieces])
    counts = np.array([len(ix) for ix in pieces])
    rows = []
    half = segments // 2
    for chosen in combinations(range(segments), half):
        in_mask = np.zeros(segments, dtype=bool); in_mask[list(chosen)] = True
        out_mask = ~in_mask
        def sharpes(mask):
            n = int(counts[mask].sum())
            total = sums[mask].sum(axis=0)
            sq = squares[mask].sum(axis=0)
            mean = total / n
            var = np.maximum((sq - n * mean * mean) / max(n - 1, 1), 0.0)
            return np.divide(mean, np.sqrt(var), out=np.zeros_like(mean), where=var > 0)
        si, so = sharpes(in_mask), sharpes(out_mask)
        selected = int(np.argmax(si))
        score = so[selected]
        below = int(np.count_nonzero(so < score))
        equal = int(np.count_nonzero(so == score))
        rank = below + (equal + 1) / 2
        omega = rank / (len(so) + 1)
        omega = float(np.clip(omega, np.finfo(float).eps, 1 - np.finfo(float).eps))
        rows.append((math.log(omega / (1 - omega)), si[selected], score))
    result = np.asarray(rows)
    lambdas = result[:, 0]
    slope = float(np.polyfit(result[:, 1], result[:, 2], 1)[0]) if len(result) > 1 and np.ptp(result[:, 1]) > 0 else float("nan")
    return {"pbo": float(np.mean(lambdas <= 0)), "lambdas": lambdas,
            "in_sample_sharpes": result[:, 1], "out_sample_sharpes": result[:, 2],
            "attenuation_slope": slope, "combinations": len(result)}


def expected_max_sharpe(sharpes: np.ndarray) -> float:
    """Euler-Mascheroni approximation for the expected max of N trials."""
    x = np.asarray(sharpes, dtype=float)
    if x.size <= 1:
        return 0.0
    sigma = float(x.std(ddof=1)) if x.size > 1 else 0.0
    if sigma == 0:
        return 0.0
    n = x.size
    euler = 0.5772156649015329
    return float(sigma * ((1 - euler) * _inv_norm(1 - 1 / n) + euler * _inv_norm(1 - 1 / (n * math.e))))


def _inv_norm(p: float) -> float:
    # Acklam approximation; avoids adding a heavy statistics dependency.
    a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
         1.383577518672690e2, -3.066479806614716e1, 2.506628277459239]
    b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
         6.680131188771972e1, -1.328068155288572e1]
    c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416]
    if p < .02425:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > 1-.02425:
        return -_inv_norm(1-p)
    q = p-.5; r=q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def deflated_sharpe(sr: float, sr0: float, sample_size: int, skewness: float, kurtosis: float) -> float:
    """DSR probability using non-annualized Sharpe and Pearson kurtosis."""
    if sample_size <= 1:
        raise ValueError("sample_size must exceed 1")
    denominator_sq = 1 - skewness * sr + (kurtosis - 1) / 4 * sr * sr
    if denominator_sq <= 0:
        raise ValueError("DSR variance adjustment must be positive")
    z = (sr - sr0) * math.sqrt(sample_size - 1) / math.sqrt(denominator_sq)
    return float(0.5 * (1 + math.erf(z / math.sqrt(2))))
