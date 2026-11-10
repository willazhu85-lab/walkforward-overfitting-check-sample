from __future__ import annotations
import argparse, os
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from .data import fetch_binance, load_offline
from .strategy import Config, backtest, performance
from .walkforward import parameter_grid, rank_grid, walk_forward, robustness, stitched_sharpe_summary
from .overfitting import grid_daily_returns, cscv_pbo, expected_max_sharpe, deflated_sharpe
ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--offline', action='store_true'); args = parser.parse_args()
    mode = 'offline' if args.offline else 'online'; out = ROOT / 'reports' / mode; out.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(out / '.mplconfig'); os.environ['XDG_CACHE_HOME'] = str(out / '.cache')
    Path(os.environ['MPLCONFIGDIR']).mkdir(exist_ok=True); Path(os.environ['XDG_CACHE_HOME']).mkdir(exist_ok=True)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    bars = load_offline(ROOT / 'data/sample_btcusdt_1d.csv') if args.offline else fetch_binance(int(pd.Timestamp('2021-01-01', tz='UTC').timestamp()*1000))
    if not args.offline:
        bars = bars[pd.to_datetime(bars.date).dt.date < datetime.now(timezone.utc).date()].reset_index(drop=True)
    bars = bars.reset_index(drop=True)
    configs = parameter_grid()
    grid = rank_grid(bars, configs); grid.to_csv(out/'parameter_grid.csv', index=False)
    best = Config(fast_length=int(grid.iloc[0].fast_length), slow_length=int(grid.iloc[0].slow_length), stop_percent=float(grid.iloc[0].stop_percent))
    full_eq, full_trades = backtest(bars, best); full_stats = performance(full_eq, full_trades, best.initial_cash)
    windows, wf_eq = walk_forward(bars, configs); windows.to_csv(out/'walk_forward.csv', index=False); wf_eq.to_csv(out/'walk_forward_equity.csv', index=False)
    stitched = stitched_sharpe_summary(windows)
    stats = robustness(grid, windows)
    wf_return = float(wf_eq.equity.iloc[-1]/best.initial_cash-1) if len(wf_eq) else float('nan')
    bh = best.initial_cash * bars.close / float(bars.close.iloc[0]); bh_return = float(bh.iloc[-1]/best.initial_cash-1)
    wf_stats = performance(wf_eq, pd.DataFrame(), best.initial_cash) if len(wf_eq) else {}
    if len(windows):
        first_oos, last_oos = pd.to_datetime(windows.test_start.iloc[0]), pd.to_datetime(windows.test_end.iloc[-1])
        full_dates = pd.to_datetime(full_eq.date); same_span = full_eq[(full_dates >= first_oos) & (full_dates <= last_oos)].reset_index(drop=True)
        full_oos_return = float(same_span.equity.iloc[-1]/same_span.equity.iloc[0]-1)
        full_oos_stats = performance(same_span, pd.DataFrame(), float(same_span.equity.iloc[0]))
        oos_bars = bars[(pd.to_datetime(bars.date) >= first_oos) & (pd.to_datetime(bars.date) <= last_oos)]
        oos_bh_return = float(oos_bars.close.iloc[-1]/oos_bars.close.iloc[0]-1)
    else:
        first_oos = last_oos = None; full_oos_return = oos_bh_return = float('nan'); full_oos_stats = {}
    fig, ax = plt.subplots(figsize=(11,5))
    if len(windows):
        full_curve = same_span.equity / same_span.equity.iloc[0] * best.initial_cash
        bh_curve = oos_bars.close / oos_bars.close.iloc[0] * best.initial_cash
        ax.plot(pd.to_datetime(same_span.date), full_curve, label='Full-sample-best (biased), OOS dates')
        ax.plot(pd.to_datetime(oos_bars.date), bh_curve, label='Buy and hold, same OOS dates')
        ax.plot(pd.to_datetime(wf_eq.date), wf_eq.equity, label='Walk-forward OOS')
    else:
        ax.plot(pd.to_datetime(full_eq.date), full_eq.equity, label='Full sample best (in-sample)')
        ax.plot(pd.to_datetime(bh.index.map(lambda i: bars.date.iloc[i])), bh, label='Buy and hold')
    ax.set(title='BTCUSDT daily EMA parameter study', ylabel='Equity (USDT)'); ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out/'equity.png', dpi=150); plt.close(fig)
    pivot = grid.pivot_table(index='slow_length', columns='fast_length', values='sharpe', aggfunc='max')
    fig, ax = plt.subplots(figsize=(10,6)); im=ax.imshow(pivot, aspect='auto', origin='lower', cmap='viridis'); ax.set_xticks(range(len(pivot.columns)), pivot.columns); ax.set_yticks(range(len(pivot.index)), pivot.index); ax.set(xlabel='Fast EMA', ylabel='Slow EMA', title='Best Sharpe over stop-loss grid'); ax.scatter(list(pivot.columns).index(best.fast_length), list(pivot.index).index(best.slow_length), marker='*', s=180, c='red', edgecolors='white', label='Full-sample optimum'); ax.legend(); fig.colorbar(im, ax=ax, label='Sharpe'); fig.tight_layout(); fig.savefig(out/'heatmap.png', dpi=150); plt.close(fig)
    return_matrix = grid_daily_returns(bars, configs)
    pbo = cscv_pbo(return_matrix, segments=16)
    best_daily = full_eq.equity.astype(float).pct_change().dropna()
    raw_sr = float(best_daily.mean()/best_daily.std(ddof=1)) if len(best_daily)>1 and best_daily.std(ddof=1)>0 else 0.0
    trial_returns = return_matrix[1:]
    trial_sd = trial_returns.std(axis=0, ddof=1)
    trial_sr = np.divide(trial_returns.mean(axis=0), trial_sd, out=np.zeros_like(trial_sd), where=trial_sd > 0)
    sr0 = expected_max_sharpe(trial_sr)
    skewness, kurtosis = float(best_daily.skew()), float(best_daily.kurt()+3)
    dsr = deflated_sharpe(raw_sr, sr0, len(best_daily), skewness, kurtosis)
    fig, ax = plt.subplots(figsize=(9,4.5)); ax.hist(pbo['lambdas'], bins=40, color='#4778a8', edgecolor='white'); ax.axvline(0, color='black', linestyle='--'); ax.set(title=f"CSCV lambda distribution (PBO={pbo['pbo']:.3f})", xlabel='lambda = logit(omega)', ylabel='Combinations'); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out/'pbo_lambda.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6,5)); ax.scatter(pbo['in_sample_sharpes'], pbo['out_sample_sharpes'], s=9, alpha=.25); xlo, xhi = ax.get_xlim(); ax.plot([xlo, xhi], np.polyval(np.polyfit(pbo['in_sample_sharpes'], pbo['out_sample_sharpes'], 1), [xlo, xhi]), color='darkorange', label=f'attenuation slope={pbo["attenuation_slope"]:.3f}'); ax.set(title='CSCV selected strategy: IS vs OOS Sharpe', xlabel='Selected IS daily Sharpe', ylabel='Selected OOS daily Sharpe'); ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out/'pbo_sharpe_scatter.png', dpi=150); plt.close(fig)
    status = 'completed' if len(windows) else f'insufficient history: need at least 455 calendar days; have {len(bars)} daily bars'
    oos_sharpe = windows.out_sample_sharpe.mean() if len(windows) else float('nan')
    positive_oos = int((windows.out_sample_return > 0).sum()) if len(windows) else 0
    eff = windows.efficiency.replace([float('inf'), -float('inf')], pd.NA).mean() if len(windows) else float('nan')
    conclusion = (f"The evidence is mixed and does not establish a stable edge: stitched walk-forward OOS returned {wf_return:.2%} versus {oos_bh_return:.2%} for buy-and-hold over the same dates, stitched Sharpe efficiency was {stitched['stitched_sharpe_efficiency']:.3f}, only {positive_oos} of {len(windows)} windows were positive, mean window OOS Sharpe was {oos_sharpe:.3f}, PBO was {pbo['pbo']:.3f}, and DSR was {dsr:.3f}. The full-sample optimum is selected using future observations and is not independent evidence." if len(windows) else f"No walk-forward conclusion can be drawn because the sample is shorter than the required training and test windows; full-sample diagnostics show PBO={pbo['pbo']:.3f} and DSR={dsr:.3f}, which do not replace walk-forward evidence.")
    fmt = lambda value, spec: format(value, spec) if pd.notna(value) else 'N/A'
    ratio_median = windows.efficiency.median() if len(windows) else float('nan')
    oos_span = f'{first_oos.date()} to {last_oos.date()}' if first_oos is not None else 'unavailable'
    report = f'''# Walk-forward robustness report\n\nResearch demonstration only; historical performance is not predictive.\n\nData: {'bundled offline CSV' if args.offline else 'Binance BTCUSDT daily from 2021-01-01'}; {len(bars)} bars ({bars.date.iloc[0]} to {bars.date.iloc[-1]}).\n\n## Full-sample grid\n\nGrid: fast EMA 5–50 step 5; slow EMA 20–200 step 10 with fast < slow; stop 2%–10% step 1%. Best full-sample parameters: fast={best.fast_length}, slow={best.slow_length}, stop={best.stop_percent:.0%}. This is an in-sample selection and is shown as a biased benchmark.\n\n## Comparable out-of-sample period\n\nThe walk-forward OOS span is {oos_span}. The full-sample optimum uses all dates to select its parameters, so its OOS-span result is a deliberately biased reference. Buy-and-hold uses the same OOS dates.\n\n| Metric | Full-sample-best, same OOS span (biased) | Walk-forward OOS | Buy and hold, same OOS span |\n|---|---:|---:|---:|\n| Total return | {fmt(full_oos_return, '.2%')} | {fmt(wf_return, '.2%')} | {fmt(oos_bh_return, '.2%')} |\n| Sharpe | {fmt(full_oos_stats.get('sharpe', float('nan')), '.3f')} (biased) | {fmt(wf_stats.get('sharpe', float('nan')), '.3f')} (stitched OOS) | — |\n| Max drawdown | {fmt(full_oos_stats.get('max_drawdown', float('nan')), '.2%')} (biased) | {fmt(wf_stats.get('max_drawdown', float('nan')), '.2%')} | — |\n| OOS windows | — | {len(windows)} | — |\n\nWalk-forward status: {status}. Each completed window optimizes only on the immediately preceding 365 calendar days and tests on the following 90 days, stepping 90 days.\n\n## Robustness\n\n- Best point neighborhood: **{stats['plateau']}** ({stats['plateau_points']} grid points within 10% of optimum in its ±1-step fast, slow, and stop neighborhood).\n- **Secondary:** mean per-window Sharpe efficiency (OOS / IS): {fmt(eff, '.3f')}; median: {fmt(ratio_median, '.3f')}. Ratios are unstable when in-sample Sharpe is near zero.\n- **Primary stitched efficiency:** concatenated OOS Sharpe {fmt(stitched['stitched_oos_sharpe'], '.3f')} / concatenated IS Sharpe {fmt(stitched['stitched_is_sharpe'], '.3f')} = {fmt(stitched['stitched_sharpe_efficiency'], '.3f')}.\n- Mean per-window OOS Sharpe: {fmt(oos_sharpe, '.3f')}; positive-return OOS windows: {positive_oos} / {len(windows)}.\n- Parameter drift, RMS step-to-step: fast period {fmt(stats['fast_period_drift'], '.3f')}; slow period {fmt(stats['slow_period_drift'], '.3f')}; stop fraction {fmt(stats['stop_drift'], '.4f')}.\n- Neighbor Sharpe decay (±1 grid step in fast, slow, and stop): {fmt(stats['neighbor_decay'], '.2%')}\n\n## Overfitting diagnostics\n\nCSCV uses {pbo['combinations']:,} complementary 8-of-16 segment splits over {len(configs):,} parameter trials. PBO = **{pbo['pbo']:.3f}** (fraction of selected in-sample winners with out-of-sample lambda <= 0); IS-to-OOS Sharpe attenuation slope = **{pbo['attenuation_slope']:.3f}**. Equal scores use average ranks, so a complete tie has omega=0.5 and lambda=0. DSR for the full-sample winner: raw daily Sharpe **{raw_sr:.5f}** (annualized **{raw_sr*365**.5:.3f}**), SR0 **{sr0:.5f}**, skewness **{skewness:.3f}**, Pearson kurtosis **{kurtosis:.3f}**, DSR **{dsr:.3f}**. {('A DSR at or below 0.5 cannot distinguish this Sharpe from a multiple-testing winner under this approximation.' if dsr <= .5 else 'DSR exceeds 0.5 under this approximation; it does not provide independent confirmation or override walk-forward evidence.')}\n\n![CSCV lambda distribution](pbo_lambda.png)\n\n![Selected in-sample versus out-of-sample Sharpe](pbo_sharpe_scatter.png)\n\n## Conclusion\n\n{conclusion}\n\nThe stop-loss dimension is included in optimization; the heatmap shows best Sharpe across stop values and marks the optimum. Fees are 0.1% per side; slippage, funding, taxes, liquidity and intraday price ordering are not modeled.\n'''
    (out/'report.md').write_text(report, encoding='utf-8')
    full_eq.to_csv(out/'full_sample_equity.csv', index=False); full_trades.to_csv(out/'full_sample_trades.csv', index=False)
    print(f'Mode: {mode}; bars: {len(bars)}; grid: {len(grid)}; walk-forward windows: {len(windows)} ({status})')
    print(f'Best: fast={best.fast_length}, slow={best.slow_length}, stop={best.stop_percent:.2%}; full-sample Sharpe={full_stats["sharpe"]:.6f}, return={full_stats["total_return"]:.6f}')
    print(f'Walk-forward OOS return={fmt(wf_return, ".6f")}; mean OOS Sharpe={fmt(oos_sharpe, ".6f")}; buy-and-hold full-period return={bh_return:.6f}')
    print(f'CSCV PBO={pbo["pbo"]:.6f} ({pbo["combinations"]} splits); raw daily SR={raw_sr:.6f}; SR0={sr0:.6f}; DSR={dsr:.6f}')
    print(f'Reports: {out}')
if __name__ == '__main__': main()
