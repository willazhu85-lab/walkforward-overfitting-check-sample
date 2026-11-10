import pandas as pd
import pytest

from sample.strategy import Config, backtest, indicators


def bars(closes, opens=None, lows=None):
    opens = opens or closes
    lows = lows or [x - 0.25 for x in closes]
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=len(closes), freq="D"),
        "open": opens,
        "high": [x + 1 for x in closes],
        "low": lows,
        "close": closes,
        "volume": [1] * len(closes),
    })


def test_ema_uses_first_close_as_seed():
    result = indicators(bars([10, 8, 9]), Config(fast_length=1, slow_length=2))
    assert result.fast_ema.tolist() == [10, 8, 9]
    assert result.slow_ema.tolist() == pytest.approx([10, 26 / 3, 80 / 9])


def test_crossunder_and_crossover_use_current_and_prior_values():
    result = indicators(bars([10, 8, 9]), Config(fast_length=1, slow_length=2))
    assert result.cross_down.tolist() == [False, True, False]
    assert result.cross_up.tolist() == [False, False, True]


def test_signal_at_close_fills_at_next_bar_open():
    data = bars([10, 8, 9, 10], opens=[10, 8, 9, 12], lows=[9.75, 7.75, 8.75, 11.75])
    equity, trades = backtest(data, Config(fast_length=1, slow_length=2, initial_cash=100))
    # Cross-up occurs on Jan 3 close; entry fills on Jan 4 open at 12.
    assert len(trades) == 0  # position is still open at the end; no fabricated exit
    # Notional is 100/12 units; entry fee 0.10 is charged against cash.
    assert equity.iloc[-1].equity == pytest.approx(83.23333)


def test_stop_triggers_on_low_and_fills_at_stop_price():
    data = bars([10, 8, 9, 10, 10], opens=[10, 8, 9, 12, 11], lows=[9.75, 7.75, 8.75, 11.5, 10])
    _, trades = backtest(data, Config(fast_length=1, slow_length=2, initial_cash=100, stop_percent=0.1))
    assert len(trades) == 1
    assert trades.iloc[0].entry_price == 12
    assert trades.iloc[0].exit_price == pytest.approx(10.8)
    assert trades.iloc[0].exit_reason == "stop"


def test_commission_is_charged_on_entry_and_exit():
    data = bars([10, 8, 9, 10, 10], opens=[10, 8, 9, 12, 11], lows=[9.75, 7.75, 8.75, 11.5, 10])
    _, trades = backtest(data, Config(fast_length=1, slow_length=2, initial_cash=100,
                                      stop_percent=0.1, commission_rate=0.001))
    assert trades.iloc[0].entry_fee == pytest.approx(0.1)
    assert trades.iloc[0].exit_fee == pytest.approx(0.09)
    # Hand calculation: 100/12 units, then proceeds at 10.8 less 0.1% fee.
    assert trades.iloc[0].quantity * 10.8 - trades.iloc[0].exit_fee == pytest.approx(89.91)


def test_crossunder_exit_fills_at_next_bar_open():
    data = bars([10, 8, 9, 10, 7, 6], opens=[10, 8, 9, 12, 8, 5],
                lows=[9.75, 7.75, 8.75, 9.75, 6.75, 5.75])
    _, trades = backtest(data, Config(fast_length=1, slow_length=2, initial_cash=100,
                                      stop_percent=0.5, commission_rate=0))
    assert len(trades) == 1
    # Crossunder is on Jan 5 close; the order fills at Jan 6 open, exactly 5.
    assert trades.iloc[0].exit_date == pd.Timestamp("2024-01-06")
    assert trades.iloc[0].exit_price == 5
    assert trades.iloc[0].exit_reason == "crossunder"


def test_gap_below_stop_fills_at_open():
    data = bars([10, 8, 9, 10, 10], opens=[10, 8, 9, 12, 9],
                lows=[9.75, 7.75, 8.75, 11.5, 8])
    _, trades = backtest(data, Config(fast_length=1, slow_length=2, initial_cash=100,
                                      stop_percent=0.1, commission_rate=0))
    assert len(trades) == 1
    # Entry is 12, so the stop is 10.8; the next open gaps to 9 and fills there.
    assert trades.iloc[0].entry_price == 12
    assert trades.iloc[0].exit_price == 9
    assert trades.iloc[0].exit_reason == "stop"
