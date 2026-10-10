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
        valuation: pd.DataFrame | None = None,
        flows: pd.DataFrame | None = None,
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
        self.volume = wide("volume") if "volume" in prices else np.full_like(self.close, np.nan)

        def wide_other(df: pd.DataFrame | None, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
            if df is None or df.empty:
                return {f: np.full_like(self.close, np.nan) for f in fields}
            # Direct numpy assignment: a pivot of ~10M daily rows needs several GB.
            dates = pd.to_datetime(df["date"]).to_numpy()
            r = days.searchsorted(dates)
            sid = df["stock_id"].astype(str).map(self.col)
            ok = sid.notna().to_numpy() & (r < len(days))
            ok[ok] &= days.to_numpy()[r[ok]] == dates[ok]
            r, c = r[ok], sid.to_numpy()[ok].astype(int)
            out = {}
            for f in fields:
                arr = np.full_like(self.close, np.nan)
                arr[r, c] = pd.to_numeric(df[f], errors="coerce").to_numpy(dtype=float)[ok]
                out[f] = arr
            return out

        # Daily exchange valuation ratios (NaN where not published, e.g. P/E of a loss maker)
        v = wide_other(valuation, ("pe", "pb", "dividend_yield"))
        self.pe, self.pb, self.dividend_yield = v["pe"], v["pb"], v["dividend_yield"]
        # Daily net buy in shares; NaN = no row that day (treated as 0 by accessors)
        f = wide_other(flows, ("foreign_net", "trust_net"))
        self.foreign_net, self.trust_net = f["foreign_net"], f["trust_net"]
        self.inst_net = np.nan_to_num(self.foreign_net) + np.nan_to_num(self.trust_net)
        self.has_flows = flows is not None and not flows.empty
        self.has_valuation = valuation is not None and not valuation.empty

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
        lookback_days: int = 420,
        benchmarks: Iterable[str] = ("0050",),
        revenue_end: str | pd.Timestamp | None = None,
    ) -> MarketData:
        """Load [start - lookback, end] so strategies can look back from day one."""
        start_ts = pd.Timestamp(start)
        load_from = start_ts - pd.Timedelta(days=lookback_days)
        uni = data.sql(
            "SELECT * FROM universe WHERE delisted_date IS NULL OR delisted_date >= ? "
            "ORDER BY stock_id",
            [load_from],
        )
        ids = list(uni["stock_id"]) + list(benchmarks)
        prices = data.prices(load_from, end, stocks=ids)
        return cls(
            prices=prices,
            universe=uni,
            revenue=data.sql("SELECT * FROM revenue WHERE available_date <= ? "
                             "ORDER BY stock_id, revenue_year, revenue_month",
                             [pd.Timestamp(revenue_end or end)]),
            actions=data.corporate_actions(load_from, end),
            trading_days=data.trading_days(load_from, end),
            valuation=_between(data, "valuation", load_from, end, ids),
            flows=_between(data, "flows", load_from, end, ids),
        )

    @property
    def tri(self) -> np.ndarray:
        """Total return index per stock (dividends reinvested), ``[day, stock]``.

        Days without a quote carry the last price; corporate-action factors are
        applied on the event day, so over any window the ratio is the holder's
        true return. After delisting the value stays flat (last price recovered).
        Computed once and cached.
        """
        if getattr(self, "_tri", None) is None:
            close = pd.DataFrame(self.close).ffill().to_numpy()
            factor = np.ones_like(close)
            for t, events in self.actions_on.items():
                for col, *_rest, f in events:
                    factor[t, col] *= f
            ratio = np.ones_like(close)
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio[1:] = close[1:] / close[:-1] * factor[1:]
            ratio[~np.isfinite(ratio)] = 1.0
            self._tri = np.cumprod(ratio, axis=0)
        return self._tri

    def index_of(self, date) -> int:
        """Index of the first trading day on or after ``date``."""
        return int(self.dates.searchsorted(pd.Timestamp(date)))


def _between(data: Data, table: str, start, end, ids: list[str]) -> pd.DataFrame | None:
    """Rows of an optional daily table in [start, end] for ``ids``; None if absent."""
    if not data.has(table):
        return None
    cols = ("date, stock_id, pe, pb, dividend_yield" if table == "valuation"
            else "date, stock_id, foreign_net, trust_net")
    return data.sql(f"SELECT {cols} FROM {table} WHERE date BETWEEN ? AND ? "
                    "AND stock_id IN (SELECT UNNEST(?))",
                    [pd.Timestamp(start), pd.Timestamp(end), list(ids)])
