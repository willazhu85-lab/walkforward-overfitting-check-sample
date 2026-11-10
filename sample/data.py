"""Public Binance daily candle download and local sample data helpers."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import pandas as pd

BASE_URL = "https://api.binance.com/api/v3/klines"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trade_count", "taker_buy_base", "taker_buy_quote", "ignore"]


def fetch_binance(start_ms: int | None = None, limit: int | None = None) -> pd.DataFrame:
    """Fetch BTCUSDT 1d klines; any HTTP/parse failure propagates to caller."""
    rows, cursor = [], start_ms
    while True:
        params = {"symbol": "BTCUSDT", "interval": "1d", "limit": min(limit or 1000, 1000)}
        if cursor is not None:
            params["startTime"] = cursor
        req = Request(BASE_URL + "?" + urlencode(params), headers={"User-Agent": "willa-backtest-sample/1.0"})
        with urlopen(req, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected Binance response: {payload}")
        if not payload:
            break
        rows.extend(payload)
        if limit is not None or len(payload) < 1000:
            break
        cursor = int(payload[-1][0]) + 86_400_000
    if not rows:
        raise RuntimeError("Binance returned no klines")
    raw = pd.DataFrame(rows, columns=COLS)
    # Build ISO dates with the standard library. This also avoids relying on
    # optional timezone dtype code in pandas for this simple UTC field.
    dates = [datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d") for ms in raw.open_time]
    result = pd.DataFrame({"date": dates,
                           **{col: pd.to_numeric(raw[col]) for col in ["open", "high", "low", "close", "volume"]}})
    return result.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def load_offline(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(data.columns) or data.empty:
        raise ValueError(f"Offline CSV must contain non-empty columns: {sorted(required)}")
    return data
