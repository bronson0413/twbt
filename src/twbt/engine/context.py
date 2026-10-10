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
        # Revenue published up to this date is visible. Backtests keep it equal
        # to ``date``; live runs set it to the run date, because revenue made
        # public after the last downloaded price is still known when trading.
        self.revenue_asof: pd.Timestamp = self.date

    def universe(self) -> list[str]:
        """Stocks that have traded by today and are not delisted (benchmarks excluded)."""
        m, t = self._m, self._t
        ok = (m.first_quote <= t) & (m.delist_idx > t)
        return [sid for sid, good in zip(m.ids, ok, strict=True)
                if good and sid in m.universe_ids]

    def industry(self) -> pd.Series:
        """Industry category per stock id ('' when unknown, e.g. delisted stocks)."""
        return pd.Series(self._m.industry, dtype=object)

    def _window(self, arr: np.ndarray, days: int) -> pd.DataFrame:
        lo = max(0, self._t - days + 1)
        return pd.DataFrame(arr[lo : self._t + 1], index=self._m.dates[lo : self._t + 1],
                            columns=self._m.ids)

    def close(self, days: int = 1) -> pd.DataFrame:
        """Raw closes for the last ``days`` trading days up to and including today."""
        return self._window(self._m.close, days)

    def total_return(self, days: int = 1) -> pd.DataFrame:
        """Total return index (dividend-adjusted) for the last ``days`` trading days.

        Only ratios between two dates are meaningful, e.g. 12-month momentum is
        ``tr.iloc[-22] / tr.iloc[-253] - 1``.
        """
        return self._window(self._m.tri, days)

    def has_history(self, days: int) -> list[str]:
        """Stocks quoted at least ``days`` trading days before today."""
        m, t = self._m, self._t
        return [sid for sid, f in zip(m.ids, m.first_quote, strict=True) if f <= t - days]

    def value(self, days: int = 1) -> pd.DataFrame:
        """Traded value (NT$) for the last ``days`` trading days."""
        return self._window(self._m.value, days)

    def revenue(self, months: int | None = None) -> pd.DataFrame:
        """Monthly revenue already public today; latest ``months`` per stock if given."""
        rev = self._m.revenue
        df = rev[rev["available_date"] <= self.revenue_asof]
        if months is not None and not df.empty:
            df = (df.sort_values(["stock_id", "revenue_year", "revenue_month"])
                  .groupby("stock_id").tail(months))
        return df.reset_index(drop=True)

    def volume(self, days: int = 1) -> pd.DataFrame:
        """Traded shares for the last ``days`` trading days."""
        return self._window(self._m.volume, days)

    def valuation(self, field: str, max_age_days: int = 10) -> pd.Series:
        """Latest published ``pe`` / ``pb`` / ``dividend_yield`` per stock as of today.

        The exchange publishes these after each close, before the next open
        when orders fill. A value older than ``max_age_days`` trading days
        (suspended stock, missing data) is dropped rather than carried forward.
        """
        if field not in ("pe", "pb", "dividend_yield"):
            raise ValueError(f"unknown valuation field {field!r}")
        m = self._m
        lo = max(0, self._t - max_age_days + 1)
        # A published row always has a dividend yield (0 if none) and usually a
        # P/B; P/E is blank for loss makers. Take each stock's latest *row* and
        # read the field from it, so a loss maker's blank P/E is not filled
        # with the P/E from before it turned to a loss.
        present = ~(np.isnan(m.pb[lo : self._t + 1]) & np.isnan(m.dividend_yield[lo : self._t + 1]))
        if not present.size:
            return pd.Series(dtype=float)
        has = present.any(axis=0)
        last = present.shape[0] - 1 - present[::-1].argmax(axis=0)
        vals = getattr(m, field)[lo : self._t + 1][last, np.arange(present.shape[1])]
        s = pd.Series(np.where(has, vals, np.nan), index=m.ids)
        return s.dropna()

    def flows(self, days: int = 20) -> pd.DataFrame:
        """Net buy in shares by foreign investors plus investment trusts, last ``days`` days.

        A day with no row counts as zero net buying.
        """
        return self._window(self._m.inst_net, days)
