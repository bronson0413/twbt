"""Monthly revenue momentum.

Rank stocks by the average year-over-year growth of their last ``months``
published monthly revenues, then hold the top ``top_n`` equally weighted.

Filters:
* liquidity: average daily traded value over ``value_days`` ≥ ``min_avg_value``
* freshness: the latest published month must be recent (stale filers skipped)
* tradable today: has a close on the signal day
"""

from __future__ import annotations

import pandas as pd

from ..engine.context import Context


def revenue_scores(
    ctx: Context, months: int = 3, max_staleness_months: int = 2
) -> pd.Series:
    """Average YoY revenue growth of the latest ``months`` months, per stock."""
    rev = ctx.revenue(months=months + 12)
    if rev.empty:
        return pd.Series(dtype=float)
    df = rev[["stock_id", "revenue_year", "revenue_month", "revenue"]].copy()
    df["p"] = df["revenue_year"] * 12 + df["revenue_month"] - 1  # month index
    ly = df[["stock_id", "p", "revenue"]].rename(columns={"revenue": "rev_ly"})
    ly["p"] += 12
    df = df.merge(ly, on=["stock_id", "p"], how="left")
    df = df[df["rev_ly"] > 0]
    df["yoy"] = df["revenue"] / df["rev_ly"] - 1

    latest = df.sort_values("p").groupby("stock_id").tail(months)
    stats = latest.groupby("stock_id").agg(yoy=("yoy", "mean"), n=("yoy", "size"),
                                           last_p=("p", "max"))
    now_p = ctx.date.year * 12 + ctx.date.month - 1
    ok = (stats["n"] == months) & (stats["last_p"] >= now_p - max_staleness_months)
    return stats.loc[ok, "yoy"]


def revenue_momentum(
    top_n: int = 8,
    months: int = 3,
    min_avg_value: float = 10_000_000,
    value_days: int = 20,
    max_staleness_months: int = 2,
):
    def strategy(ctx: Context) -> dict[str, float]:
        scores = revenue_scores(ctx, months, max_staleness_months)
        universe = set(ctx.universe())
        avg_value = ctx.value(value_days).mean()
        last_close = ctx.close(1).iloc[-1]
        eligible = [
            s for s in scores.index
            if s in universe and avg_value.get(s, 0) >= min_avg_value
            and pd.notna(last_close.get(s))
        ]
        picks = scores[eligible].sort_values(ascending=False).head(top_n)
        # 1/top_n each: if fewer stocks qualify, the rest stays in cash
        return {s: 1.0 / top_n for s in picks.index}

    strategy.__name__ = f"revenue_momentum_top{top_n}_{months}m"
    return strategy
