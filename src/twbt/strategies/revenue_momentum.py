"""Monthly revenue momentum.

Score every eligible stock by revenue growth over its latest ``months``
published months versus the same months a year earlier, then hold the top
``top_n`` equally weighted.

Scoring (``method``):
* ``"sum"``: sum of the last N months / sum of the same N months last year - 1.
  Big months weigh more, so one month against a tiny base can't dominate.
* ``"mean"``: average of the N single-month YoY rates (the original version;
  kept for comparison — it is easily blown up by a near-zero base).

Filters (all applied on the signal day with data already public):
* liquidity: average daily traded value over ``value_days`` ≥ ``min_avg_value``
* freshness: the latest published month must be recent (stale filers skipped)
* tradable: has a close on the signal day
* industries: construction is excluded by default — builders book revenue on
  handover, so YoY jumps reflect project timing, not growth
* ``min_consecutive``: the last K single-month YoY rates must all be > 0
* ``min_base``: last year's N-month revenue sum must be at least this (NT$)
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from ..engine.context import Context

DEFAULT_EXCLUDED_INDUSTRIES = ("建材營造",)


def _monthly_yoy(ctx: Context, months: int) -> pd.DataFrame:
    """Latest ``months`` published months per stock with last year's value alongside."""
    rev = ctx.revenue(months=months + 12)
    cols = ["stock_id", "p", "revenue", "rev_ly"]
    if rev.empty:
        return pd.DataFrame(columns=cols)
    df = rev[["stock_id", "revenue_year", "revenue_month", "revenue"]].copy()
    df["p"] = df["revenue_year"] * 12 + df["revenue_month"] - 1  # month index
    ly = df[["stock_id", "p", "revenue"]].rename(columns={"revenue": "rev_ly"})
    ly["p"] += 12
    df = df.merge(ly, on=["stock_id", "p"], how="left")
    return df.sort_values("p").groupby("stock_id").tail(months)[cols]


def revenue_scores(
    ctx: Context,
    months: int = 3,
    max_staleness_months: int = 2,
    method: str = "sum",
    min_consecutive: int = 0,
    min_base: float = 0.0,
) -> pd.Series:
    """Revenue growth score per stock (higher = faster growth)."""
    if method not in ("sum", "mean"):
        raise ValueError(f"unknown method {method!r}")
    df = _monthly_yoy(ctx, months)
    if df.empty:
        return pd.Series(dtype=float)
    df = df[df["rev_ly"] > 0].copy()
    df["yoy"] = df["revenue"] / df["rev_ly"] - 1
    g = df.groupby("stock_id")
    stats = pd.DataFrame({
        "n": g.size(),
        "last_p": g["p"].max(),
        "rev": g["revenue"].sum(),
        "rev_ly": g["rev_ly"].sum(),
        "mean_yoy": g["yoy"].mean(),
    })
    if min_consecutive:
        k = min(min_consecutive, months)
        newest_first = df.groupby("stock_id")["p"].rank(ascending=False)
        recent = df[newest_first <= k]
        stats["recent_up"] = recent.groupby("stock_id")["yoy"].min() > 0
        stats["recent_up"] = stats["recent_up"].fillna(False).astype(bool)
    now_p = ctx.date.year * 12 + ctx.date.month - 1
    ok = (stats["n"] == months) & (stats["last_p"] >= now_p - max_staleness_months)
    ok &= stats["rev_ly"] >= max(min_base, 1)
    if min_consecutive:
        ok &= stats["recent_up"]
    score = stats["rev"] / stats["rev_ly"] - 1 if method == "sum" else stats["mean_yoy"]
    return score[ok]


def eligible(
    ctx: Context,
    min_avg_value: float = 10_000_000,
    value_days: int = 20,
    exclude_industries: Iterable[str] = DEFAULT_EXCLUDED_INDUSTRIES,
) -> list[str]:
    """Stocks a strategy may hold today: listed, liquid, quoted, allowed industry."""
    avg_value = ctx.value(value_days).mean()
    last_close = ctx.close(1).iloc[-1]
    industry = ctx.industry()
    banned = tuple(exclude_industries)
    out = []
    for s in ctx.universe():
        if avg_value.get(s, 0) < min_avg_value or pd.isna(last_close.get(s)):
            continue
        ind = industry.get(s, "") or ""
        if banned and any(b in ind for b in banned):
            continue
        out.append(s)
    return out


def revenue_momentum(
    top_n: int = 8,
    months: int = 3,
    min_avg_value: float = 10_000_000,
    value_days: int = 20,
    max_staleness_months: int = 2,
    method: str = "sum",
    min_consecutive: int = 0,
    min_base: float = 0.0,
    exclude_industries: Iterable[str] = DEFAULT_EXCLUDED_INDUSTRIES,
):
    exclude_industries = tuple(exclude_industries)

    def strategy(ctx: Context) -> dict[str, float]:
        scores = revenue_scores(ctx, months, max_staleness_months, method,
                                min_consecutive, min_base)
        ok = set(eligible(ctx, min_avg_value, value_days, exclude_industries))
        picks = scores[scores.index.isin(ok)].sort_values(ascending=False).head(top_n)
        # 1/top_n each: if fewer stocks qualify, the rest stays in cash
        return {s: 1.0 / top_n for s in picks.index}

    strategy.__name__ = f"revenue_momentum_top{top_n}_{months}m_{method}"
    return strategy
