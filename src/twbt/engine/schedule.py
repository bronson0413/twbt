"""Rebalance calendars."""

from __future__ import annotations

import pandas as pd


def monthly_on_or_after(trading_days: pd.DatetimeIndex, day: int) -> list[pd.Timestamp]:
    """The first trading day on or after ``day`` of each month.

    Example: ``day=11`` signals right after monthly revenue becomes public in
    most months (the strategy itself only sees what is actually available).
    """
    days = pd.DatetimeIndex(trading_days).sort_values()
    out = []
    for _, month_days in days.to_series().groupby(days.to_period("M")):
        later = month_days[month_days.dt.day >= day]
        if not later.empty:
            out.append(later.iloc[0])
    return out
