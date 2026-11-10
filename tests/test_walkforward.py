import numpy as np
import pandas as pd
from sample.strategy import Config
from sample.walkforward import parameter_grid, split_windows, stitch_oos_equity, walk_forward


def synthetic(n=500):
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    close = 100 + np.arange(n) * 0.1 + 5 * np.sin(np.arange(n) / 13)
    return pd.DataFrame({'date': dates, 'open': close, 'high': close + 1, 'low': close - 1,
                         'close': close, 'volume': 1.0})


def test_grid_respects_requested_bounds_and_fast_less_than_slow():
    grid = parameter_grid()
    assert min(c.fast_length for c in grid) == 5
    assert max(c.fast_length for c in grid) == 50
    assert min(c.slow_length for c in grid) == 20
    assert max(c.slow_length for c in grid) == 200
    assert min(c.stop_percent for c in grid) == 0.02
    assert max(c.stop_percent for c in grid) == 0.10
    assert all(c.fast_length < c.slow_length for c in grid)


def test_windows_are_disjoint_and_oos_follows_its_training_window():
    bars = synthetic()
    windows = split_windows(bars)
    assert len(windows) == 1
    train, test = windows[0]
    assert bars.date.iloc[train[-1]] < bars.date.iloc[test[0]]
    assert set(train).isdisjoint(set(test))
    assert bars.date.iloc[train[0]] == pd.Timestamp('2020-01-01')
    assert bars.date.iloc[test[0]] == bars.date.iloc[train[0]] + pd.Timedelta(days=365)


def test_first_oos_parameter_choice_cannot_see_oos_prices():
    bars = synthetic()
    configs = [Config(fast_length=5, slow_length=20, stop_percent=0.02),
               Config(fast_length=10, slow_length=30, stop_percent=0.05)]
    before = walk_forward(bars, configs)[0].iloc[0]
    mutated = bars.copy()
    mutated.loc[365:, ['open', 'high', 'low', 'close']] *= 7
    after = walk_forward(mutated, configs)[0].iloc[0]
    assert (before.fast_length, before.slow_length, before.stop_percent) == (
        after.fast_length, after.slow_length, after.stop_percent)
    assert before.test_start == after.test_start
    assert not np.isclose(before.out_sample_return, after.out_sample_return)


def test_walkforward_oos_curve_is_date_ordered_and_continuous():
    # 550 days produce two adjacent 90-day OOS slices; rebasing carries wealth forward.
    bars = synthetic(550)
    configs = [Config(fast_length=5, slow_length=20, stop_percent=0.05)]
    windows, curve = walk_forward(bars, configs)
    assert len(windows) == 2
    assert len(curve) == 180
    assert curve.date.is_monotonic_increasing
    assert curve.date.iloc[90] == curve.date.iloc[89] + pd.Timedelta(days=1)


def test_stitched_curve_carries_prior_wealth_with_hand_calculated_values():
    first = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=2), 'equity': [10_000., 11_000.]})
    second = pd.DataFrame({'date': pd.date_range('2024-01-03', periods=2), 'equity': [10_050., 9_900.]})
    curve = stitch_oos_equity([first, second], initial_cash=10_000.)
    # Second segment scale is 11,000 / 10,000; values are independently hand-calculated.
    assert curve.equity.tolist() == [10_000., 11_000., 11_055., 10_890.]
    assert curve.date.diff().dropna().dt.days.tolist() == [1, 1, 1]
