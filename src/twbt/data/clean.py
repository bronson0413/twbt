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
        # e.g. 興櫃 first-day open is 0 even when there is a close
        df.loc[df[c] <= 0, c] = np.nan
    return df.sort_values(["stock_id", "date"]).reset_index(drop=True)[cols]


def drop_emerging_period(prices: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Remove 興櫃-era rows of stocks that later listed (up to and including the switch date).

    Keeping them would let a backtest trade a stock before it was listed and,
    worse, only stocks that *did* list later — a survivorship bias.
    """
    if prices.empty or "emerging_until" not in universe:
        return prices
    cutoff = universe.set_index("stock_id")["emerging_until"].dropna()
    if cutoff.empty:
        return prices
    until = prices["stock_id"].map(cutoff)
    keep = until.isna() | (prices["date"] > until)
    dropped = int((~keep).sum())
    if dropped:
        log.info("dropped %d 興櫃-period price rows", dropped)
    return prices[keep].reset_index(drop=True)


# -- monthly revenue --------------------------------------------------------


def next_trading_day_after(
    dates: pd.Series, trading_days: pd.DatetimeIndex | None, inclusive: bool = False
) -> pd.Series:
    """First trading day after each date (or on/after it if ``inclusive``).

    Without a calendar, or past its end, falls back to the next calendar day.
    """
    fallback = dates if inclusive else dates + pd.Timedelta(days=1)
    if trading_days is None or len(trading_days) == 0:
        return fallback
    td = trading_days.sort_values().to_numpy()
    idx = np.searchsorted(td, dates.to_numpy(), side="left" if inclusive else "right")
    in_range = (idx < len(td)) & dates.notna().to_numpy()
    out = fallback.copy()
    out[in_range] = td[idx[in_range]]
    return out


def revenue_available_date(
    year: pd.Series, month: pd.Series, trading_days: pd.DatetimeIndex | None = None
) -> pd.Series:
    """Earliest date we allow a backtest to use revenue for (year, month).

    The legal deadline is the 10th of the following month, extended to the
    next business day when the 10th is a holiday (e.g. 10/10 National Day, or
    a weekend). The number is only known for sure from the trading day after
    that. Using an earlier date for companies that happened to publish early
    would make the backtest look better than a systematic monthly process can.
    """
    nxt = pd.to_datetime(
        {"year": year.astype(int), "month": month.astype(int), "day": 1}
    ) + pd.DateOffset(months=1)
    deadline = nxt + pd.Timedelta(days=REVENUE_DEADLINE_DAY - 1)
    deadline = next_trading_day_after(deadline, trading_days, inclusive=True)
    return next_trading_day_after(deadline, trading_days)


def clean_revenue(raw: pd.DataFrame, trading_days: pd.DatetimeIndex | None = None) -> pd.DataFrame:
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
    ).dt.strftime("%Y-%m")  # plain string: portable through parquet and DuckDB
    available = revenue_available_date(df["revenue_year"], df["revenue_month"], trading_days)

    # Late filers: if FinMind first saw the row after the deadline, the market
    # couldn't have used it earlier either. create_time is a date with no time,
    # so we assume it arrived after the close and allow it from the next
    # trading day.
    if "create_time" in raw.columns:
        ct = pd.to_datetime(raw["create_time"], errors="coerce")  # "" → NaT
        ct = ct.where(ct != CREATE_TIME_BACKFILL_DATE)
        late = next_trading_day_after(ct, trading_days)
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


# -- data fixes ----------------------------------------------------------------

LIMIT_CHANGE_DATE = pd.Timestamp("2015-06-01")
NEW_LISTING_GRACE_DAYS = 5


def unexplained_moves(
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    trading_days: pd.DatetimeIndex | None = None,
    tolerance: float = 0.015,
) -> pd.DataFrame:
    """Day-over-day moves beyond the price limit with no corporate action that day.

    Listed stocks can move at most ±10% a day (±7% before 2015-06-01), and new
    listings are free for their first five sessions. Anything else beyond the
    limit is a bad print or a missing event (capital reduction, par-value
    change, split before the event tables start). Only consecutive trading
    days are compared: a move across a suspension can legitimately compound
    several limit moves.

    Returns rows shaped like ``clean_corporate_actions`` output with
    ``kind="data_fix"`` and ``factor = prev_close / close``: applied like any
    other corporate action, they leave the holder's value unchanged, so the
    jump is neither a gain nor a loss.
    """
    cols = ["date", "stock_id", "kind", "detail", "before_price", "reference_price", "factor"]
    q = prices[prices["close"].notna()].sort_values(["stock_id", "date"])
    if q.empty:
        return pd.DataFrame(columns=cols)
    if trading_days is None or len(trading_days) == 0:
        trading_days = pd.DatetimeIndex(sorted(q["date"].unique()))
    pos = pd.Series(np.arange(len(trading_days)), index=pd.DatetimeIndex(trading_days))
    q = q.assign(pos=q["date"].map(pos))
    g = q.groupby("stock_id")
    q["prev_close"] = g["close"].shift(1)
    q["prev_pos"] = g["pos"].shift(1)
    q["nth"] = g.cumcount()
    ev = (actions.groupby(["stock_id", "date"])["factor"].prod()
          if not actions.empty else pd.Series(dtype=float))
    key = pd.MultiIndex.from_arrays([q["stock_id"], q["date"]])
    has_event = key.isin(ev.index) if len(ev) else np.zeros(len(q), dtype=bool)
    limit = np.where(q["date"] >= LIMIT_CHANGE_DATE, 0.10, 0.07) + tolerance
    move = q["close"] / q["prev_close"] - 1
    bad = ((move.abs() > limit) & (q["pos"] - q["prev_pos"] == 1)
           & (q["nth"] >= NEW_LISTING_GRACE_DAYS) & ~has_event)
    b = q[bad]
    return pd.DataFrame({
        "date": b["date"].to_numpy(),
        "stock_id": b["stock_id"].to_numpy(),
        "kind": "data_fix",
        "detail": [f"{m:+.1%}" for m in move[bad]],
        "before_price": b["prev_close"].to_numpy(),
        "reference_price": b["close"].to_numpy(),
        "factor": (b["prev_close"] / b["close"]).to_numpy(),
    })[cols]


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
    prices = drop_emerging_period(clean_prices(raw.read_all(PRICE)), universe)
    revenue = clean_revenue(raw.read_all(MONTH_REVENUE), pd.DatetimeIndex(tdays["date"]))
    actions = clean_corporate_actions(
        raw.read_all(DIVIDEND_RESULT),
        raw.read_all(CAPITAL_REDUCTION),
        raw.read(SPLIT),
        raw.read(PAR_VALUE),
    )

    fixes = unexplained_moves(prices, actions, pd.DatetimeIndex(tdays["date"]))
    if len(fixes):
        log.info("neutralized %d unexplained moves beyond the price limit", len(fixes))
        actions = (pd.concat([actions, fixes], ignore_index=True)
                   .sort_values(["stock_id", "date"]).reset_index(drop=True))

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
