"""Data audit: find price moves the market rules say cannot happen.

Taiwan listed stocks move at most ±10% a day (±7% before 2015-06-01). A
close-to-close move well beyond that, on a day with no recorded corporate
action, almost always means a missing event (capital reduction, split,
par-value change) or a bad print. Uncorrected, it shows up in a backtest as
free profit or loss — momentum strategies are especially drawn to them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine.market import MarketData

LIMIT_CHANGE_DATE = pd.Timestamp("2015-06-01")
NEW_LISTING_GRACE_DAYS = 5  # the first 5 trading days of a new listing have no price limit


def impossible_moves(market: MarketData, tolerance: float = 0.015) -> pd.DataFrame:
    """Close-to-close moves beyond the daily limit (+ tolerance) with no corporate action.

    Moves across a gap of missing quotes are compared over the gap, so only a
    gap of one session is checked (a multi-day suspension can legitimately
    compound several limit moves).
    """
    close = market.close
    prev = np.full_like(close, np.nan)
    prev[1:] = close[:-1]
    factor = np.ones_like(close)
    for t, events in market.actions_on.items():
        for col, *_rest, f in events:
            factor[t, col] *= f
    has_event = factor != 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        move = close * factor / prev - 1
    limit = np.where(market.dates >= LIMIT_CHANGE_DATE, 0.10, 0.07)[:, None] + tolerance
    t_idx = np.arange(len(market.dates))[:, None]
    seasoned = t_idx >= market.first_quote[None, :] + NEW_LISTING_GRACE_DAYS
    bad = (np.abs(move) > limit) & seasoned & ~has_event & np.isfinite(move)
    ti, ci = np.nonzero(bad)
    return pd.DataFrame({
        "date": market.dates[ti],
        "stock_id": [market.ids[c] for c in ci],
        "prev_close": prev[ti, ci],
        "close": close[ti, ci],
        "move": move[ti, ci],
    }).sort_values("move", key=np.abs, ascending=False).reset_index(drop=True)


def stock_summary(market: MarketData, stock_id: str, start, end) -> dict:
    """Raw prices, events and total return of one stock over a window (for spot checks)."""
    c = market.col[stock_id]
    t0, t1 = market.index_of(start), min(market.index_of(end), len(market.dates) - 1)
    events = [(market.dates[t].date().isoformat(), kind, detail, round(f, 6))
              for t in range(t0, t1 + 1)
              for col, kind, detail, _b, _r, f in market.actions_on.get(t, []) if col == c]
    tri = market.tri[:, c]
    return {
        "stock_id": stock_id,
        "start": market.dates[t0].date().isoformat(),
        "end": market.dates[t1].date().isoformat(),
        "close_start": float(market.close[t0, c]),
        "close_end": float(market.close[t1, c]),
        "events": events,
        "total_return": float(tri[t1] / tri[t0] - 1),
    }
