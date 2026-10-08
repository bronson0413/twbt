"""Build the clean layer from raw data.

Outputs under ``<root>/clean``:

    universe.parquet          stock_id, stock_name, industry, market, delisted_date
    trading_days.parquet      date
    prices.parquet            raw (unadjusted) daily OHLCV
    revenue.parquet           monthly revenue with ``available_date``
    corporate_actions.parquet ex-dividend / capital reduction / split / par-value events

The clean layer can always be rebuilt from raw; it is never edited by hand.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import (
    CAPITAL_REDUCTION,
    DELISTING,
    DIVIDEND_RESULT,
    MONTH_REVENUE,
    PAR_VALUE,
    PRICE,
    SPLIT,
    STOCK_INFO,
    TRADING_DATE,
)
from .store import RawStore
from .universe import build_universe

log = logging.getLogger(__name__)

# FinMind started recording create_time on this date and stamped every row
# written that day with it, so on this date the value means nothing.
CREATE_TIME_BACKFILL_DATE = pd.Timestamp("2026-04-21")
# Listed companies must publish monthly revenue by the 10th of the next month.
REVENUE_DEADLINE_DAY = 10


# -- prices ---------------------------------------------------------------


def clean_prices(raw: pd.DataFrame) -> pd.DataFrame:
    cols = ["date", "stock_id", "open", "high", "low", "close", "volume", "value", "has_quote"]
    if raw.empty:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["date"]),
            "stock_id": raw["stock_id"].astype(str),
            "open": raw["open"].astype(float),
            "high": raw["max"].astype(float),
            "low": raw["min"].astype(float),
            "close": raw["close"].astype(float),
            "volume": raw["Trading_Volume"].astype("int64"),  # shares, not 張
            "value": raw["Trading_money"].astype("int64"),  # NT$
        }
    )
    # FinMind writes 0 for every price field on days with no published price
    # (no trades, or trades without an official price). Turning these into NaN
    # stops a 0 from ever being used as a fill price or a -100% return.
    df["has_quote"] = df["close"] > 0
    for c in ("open", "high", "low", "close"):
        df.loc[~df["has_quote"], c] = np.nan
    return df.sort_values(["stock_id", "date"]).reset_index(drop=True)[cols]


# -- monthly revenue --------------------------------------------------------


def revenue_available_date(year: pd.Series, month: pd.Series) -> pd.Series:
    """Earliest date we allow a backtest to use revenue for (year, month).

    The legal deadline is the 10th of the following month, so the number is
    only known for sure from the 11th. Using an earlier date for companies that
    happened to publish early would make the backtest look better than what a
    systematic monthly process could actually do.
    """
    nxt = pd.to_datetime(
        {"year": year.astype(int), "month": month.astype(int), "day": 1}
    ) + pd.DateOffset(months=1)
    return nxt + pd.Timedelta(days=REVENUE_DEADLINE_DAY)


def clean_revenue(raw: pd.DataFrame) -> pd.DataFrame:
    cols = ["stock_id", "period", "revenue_year", "revenue_month", "revenue", "available_date"]
    if raw.empty:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(
        {
            "stock_id": raw["stock_id"].astype(str),
            "revenue_year": raw["revenue_year"].astype(int),
            "revenue_month": raw["revenue_month"].astype(int),
            "revenue": raw["revenue"].astype("int64"),
        }
    )
    df["period"] = pd.to_datetime(
        {"year": df["revenue_year"], "month": df["revenue_month"], "day": 1}
    ).dt.to_period("M")
    available = revenue_available_date(df["revenue_year"], df["revenue_month"])

    # Late filers: if FinMind first saw the row after the deadline, the market
    # couldn't have used it earlier either. create_time is a date with no time,
    # so we assume it arrived after the close and allow it from the next day.
    if "create_time" in raw.columns:
        ct = pd.to_datetime(raw["create_time"], errors="coerce")  # "" → NaT
        ct = ct.where(ct != CREATE_TIME_BACKFILL_DATE)
        late = ct + pd.Timedelta(days=1)
        available = available.where(~(late > available), late)

    df["available_date"] = available
    df = df.drop_duplicates(["stock_id", "period"], keep="last")
    return df.sort_values(["stock_id", "period"]).reset_index(drop=True)[cols]


# -- corporate actions --------------------------------------------------------


def _events(
    raw: pd.DataFrame, kind: str, before: str, reference: str, label: str | None = None
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["date"]),
            "stock_id": raw["stock_id"].astype(str),
            "kind": kind,
            "detail": raw[label].astype(str) if label in raw.columns else "",
            "before_price": pd.to_numeric(raw[before], errors="coerce"),
            "reference_price": pd.to_numeric(raw[reference], errors="coerce"),
        }
    )
    return out


def clean_corporate_actions(
    dividends: pd.DataFrame,
    reductions: pd.DataFrame,
    splits: pd.DataFrame,
    par_changes: pd.DataFrame,
) -> pd.DataFrame:
    """One row per event with ``factor = before_price / reference_price``.

    The factor is how much the price "drops" mechanically on the event date
    without the holder losing value. A backtest holding through the event keeps
    its value by multiplying shares by the factor (total-return convention:
    cash dividends are treated as reinvested on the ex-date).
    """
    cols = ["date", "stock_id", "kind", "detail", "before_price", "reference_price", "factor"]
    parts = [
        _events(dividends, "dividend", "before_price", "reference_price",
                "stock_or_cache_dividend"),
        _events(reductions, "capital_reduction", "ClosingPriceonTheLastTradingDay",
                "PostReductionReferencePrice", "ReasonforCapitalReduction"),
        _events(splits, "split", "before_price", "after_price", "type"),
        _events(par_changes, "par_value_change", "before_close", "after_ref_close"),
    ]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=cols)
    ev = pd.concat(parts, ignore_index=True)
    bad = (ev["before_price"] <= 0) | (ev["reference_price"] <= 0) | ev.isna()[
        ["before_price", "reference_price"]
    ].any(axis=1)
    if bad.any():
        log.warning("dropping %d corporate-action rows with invalid prices", int(bad.sum()))
    ev = ev[~bad].copy()
    ev["factor"] = ev["before_price"] / ev["reference_price"]
    ev = ev.drop_duplicates(["stock_id", "date", "kind"], keep="last")
    return ev.sort_values(["stock_id", "date"]).reset_index(drop=True)[cols]


def adjusted_close(prices: pd.DataFrame, actions: pd.DataFrame) -> pd.Series:
    """Backward-adjusted (total return) close, aligned to ``prices``' index.

    Prices before an event are multiplied by ``1 / factor`` so the series has
    no artificial drop on ex-dates. The latest price equals the raw price.
    For signals (momentum etc.) only; fills always use raw prices.
    """
    mult = pd.Series(1.0, index=prices.index)
    if actions.empty:
        return prices["close"] * mult
    for sid, ev in actions.groupby("stock_id"):
        rows = prices.index[prices["stock_id"] == sid]
        if len(rows) == 0:
            continue
        ev = ev.sort_values("date")
        ratios = (1.0 / ev["factor"]).to_numpy()
        # suffix[i] = product of ratios of events i..end; suffix[n] = 1
        suffix = np.append(np.cumprod(ratios[::-1])[::-1], 1.0)
        ev_dates = ev["date"].to_numpy()
        px_dates = prices.loc[rows, "date"].to_numpy()
        # first event strictly after each price date
        idx = np.searchsorted(ev_dates, px_dates, side="right")
        mult.loc[rows] = suffix[idx]
    return prices["close"] * mult


# -- build -----------------------------------------------------------------


def build(root: str | Path, since: str = "2010-01-01") -> dict[str, int]:
    root = Path(root)
    raw = RawStore(root)
    out = root / "clean"
    out.mkdir(parents=True, exist_ok=True)

    universe = build_universe(raw.read(STOCK_INFO), raw.read(DELISTING), since=since)
    tdays = raw.read(TRADING_DATE)
    tdays = (
        pd.DataFrame({"date": pd.to_datetime(tdays["date"]).sort_values().unique()})
        if not tdays.empty
        else pd.DataFrame(columns=["date"])
    )
    prices = clean_prices(raw.read_all(PRICE))
    revenue = clean_revenue(raw.read_all(MONTH_REVENUE))
    actions = clean_corporate_actions(
        raw.read_all(DIVIDEND_RESULT),
        raw.read_all(CAPITAL_REDUCTION),
        raw.read(SPLIT),
        raw.read(PAR_VALUE),
    )

    tables = {
        "universe": universe,
        "trading_days": tdays,
        "prices": prices,
        "revenue": revenue,
        "corporate_actions": actions,
    }
    for name, df in tables.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
    summary = {name: len(df) for name, df in tables.items()}
    log.info("clean layer built: %s", summary)
    return summary
