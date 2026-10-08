"""What a strategy is allowed to see on a signal day.

The context is the only thing passed to a strategy. Every accessor is cut off
at the signal day, so look-ahead is impossible by construction rather than by
discipline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .market import MarketData


class Context:
    def __init__(self, market: MarketData, t: int) -> None:
        self._m = market
        self._t = t
        self.date: pd.Timestamp = market.dates[t]

    def universe(self) -> list[str]:
        """Stocks that have traded by today and are not delisted (benchmarks excluded)."""
        m, t = self._m, self._t
        ok = (m.first_quote <= t) & (m.delist_idx > t)
        return [sid for sid, good in zip(m.ids, ok, strict=True)
                if good and sid in m.universe_ids]

    def _window(self, arr: np.ndarray, days: int) -> pd.DataFrame:
        lo = max(0, self._t - days + 1)
        return pd.DataFrame(arr[lo : self._t + 1], index=self._m.dates[lo : self._t + 1],
                            columns=self._m.ids)

    def close(self, days: int = 1) -> pd.DataFrame:
        """Raw closes for the last ``days`` trading days up to and including today."""
        return self._window(self._m.close, days)

    def value(self, days: int = 1) -> pd.DataFrame:
        """Traded value (NT$) for the last ``days`` trading days."""
        return self._window(self._m.value, days)

    def revenue(self, months: int | None = None) -> pd.DataFrame:
        """Monthly revenue already public today; latest ``months`` per stock if given."""
        rev = self._m.revenue
        df = rev[rev["available_date"] <= self.date]
        if months is not None and not df.empty:
            df = (df.sort_values(["stock_id", "revenue_year", "revenue_month"])
                  .groupby("stock_id").tail(months))
        return df.reset_index(drop=True)
