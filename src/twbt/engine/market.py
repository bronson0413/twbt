"""All market data a backtest needs, held in memory as date × stock arrays."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from ..data.query import Data


class MarketData:
    """Wide arrays indexed ``[day, stock]``; NaN where there was no quote."""

    def __init__(
        self,
        prices: pd.DataFrame,
        universe: pd.DataFrame,
        revenue: pd.DataFrame | None = None,
        actions: pd.DataFrame | None = None,
        trading_days: Iterable | None = None,
    ) -> None:
        prices = prices.copy()
        prices["date"] = pd.to_datetime(prices["date"])
        prices["stock_id"] = prices["stock_id"].astype(str)
        if trading_days is None:
            days = pd.DatetimeIndex(sorted(prices["date"].unique()))
        else:
            days = pd.DatetimeIndex(sorted(pd.to_datetime(list(trading_days))))
        self.dates = days
        self.ids: list[str] = sorted(prices["stock_id"].unique())
        self.col = {sid: i for i, sid in enumerate(self.ids)}

        prices = prices.drop_duplicates(["date", "stock_id"], keep="last")

        def wide(field: str) -> np.ndarray:
            w = prices.pivot(index="date", columns="stock_id", values=field)
            return w.reindex(index=days, columns=self.ids).to_numpy(dtype=float)

        self.open = wide("open")
        self.high = wide("high")
        self.low = wide("low")
        self.close = wide("close")
        self.value = wide("value") if "value" in prices else np.full_like(self.close, np.nan)

        # first day each stock had a real quote; len(days) if never
        has = ~np.isnan(self.close)
        self.first_quote = np.where(has.any(axis=0), has.argmax(axis=0), len(days))

        universe = universe.copy()
        universe["stock_id"] = universe["stock_id"].astype(str)
        self.universe_ids = set(universe["stock_id"]) & set(self.ids)
        self.industry: dict[str, str] = (
            dict(zip(universe["stock_id"], universe["industry"].fillna(""), strict=True))
            if "industry" in universe else {}
        )
        self.delist_idx = np.full(len(self.ids), len(days), dtype=int)
        if "delisted_date" in universe:
            for sid, d in zip(universe["stock_id"], universe["delisted_date"], strict=True):
                if sid in self.col and pd.notna(d):
                    self.delist_idx[self.col[sid]] = days.searchsorted(pd.Timestamp(d))

        self.revenue = revenue if revenue is not None else pd.DataFrame(
            columns=["stock_id", "period", "revenue_year", "revenue_month", "revenue",
                     "available_date"]
        )
        if not self.revenue.empty:
            self.revenue = self.revenue.copy()
            self.revenue["stock_id"] = self.revenue["stock_id"].astype(str)
            self.revenue["available_date"] = pd.to_datetime(self.revenue["available_date"])

        # corporate actions keyed by day index
        self.actions_on: dict[int, list[tuple[int, str, str, float, float, float]]] = {}
        if actions is not None and not actions.empty:
            for row in actions.itertuples(index=False):
                sid = str(row.stock_id)
                if sid not in self.col:
                    continue
                t = days.searchsorted(pd.Timestamp(row.date))
                if t >= len(days) or days[t] != pd.Timestamp(row.date):
                    continue  # event on a day outside the calendar window
                self.actions_on.setdefault(t, []).append(
                    (self.col[sid], row.kind, str(row.detail), float(row.before_price),
                     float(row.reference_price), float(row.factor))
                )

    @classmethod
    def from_data(
        cls,
        data: Data,
        start: str,
        end: str,
        lookback_days: int = 400,
        benchmarks: Iterable[str] = ("0050",),
    ) -> MarketData:
        """Load [start - lookback, end] so strategies can look back from day one."""
        start_ts = pd.Timestamp(start)
        load_from = start_ts - pd.Timedelta(days=lookback_days)
        uni = data.sql(
            "SELECT * FROM universe WHERE delisted_date IS NULL OR delisted_date >= ?",
            [load_from],
        )
        ids = list(uni["stock_id"]) + list(benchmarks)
        prices = data.prices(load_from, end, stocks=ids)
        return cls(
            prices=prices,
            universe=uni,
            revenue=data.sql("SELECT * FROM revenue WHERE available_date <= ?",
                             [pd.Timestamp(end)]),
            actions=data.corporate_actions(load_from, end),
            trading_days=data.trading_days(load_from, end),
        )

    def index_of(self, date) -> int:
        """Index of the first trading day on or after ``date``."""
        return int(self.dates.searchsorted(pd.Timestamp(date)))
