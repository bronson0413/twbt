"""FinLab's "monthly revenue momentum" strategy, rebuilt rule by rule.

Source: https://finlab.finance/strategies/monthly-revenue-momentum (2026).

    pct_change = (close / close.shift() - 1).rolling(5).mean()
    trend      = close > MA60 & close > MA20 & close > MA120
    rev_up     = rev.average(3) > rev.average(12)
    liquid     = volume.average(20) > 500 * 1000            # 20-stock version only
    position   = pct_change[trend & rev_up].is_largest(N)
    sim(position, resample="M", resample_offset="11D", stop_loss=0.1)

FinLab's ``close`` is the raw (unadjusted) close, so it is used here too.
Revenue is read point-in-time: a month counts once it is public.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine.context import Context
from .combo import rank_top


def revmom_scores(ctx: Context, liquid_shares: float | None = None) -> pd.Series:
    """5-day average daily return of stocks in an uptrend with rising revenue."""
    close = ctx.close(121)
    if len(close) < 121:
        return pd.Series(dtype=float)
    last = close.iloc[-1]
    pct = (close / close.shift(1) - 1).iloc[-5:].mean(skipna=False)
    trend = ((last > close.iloc[-20:].mean(skipna=False))
             & (last > close.iloc[-60:].mean(skipna=False))
             & (last > close.iloc[-120:].mean(skipna=False)))
    rev = ctx.revenue(months=12)
    if rev.empty:
        return pd.Series(dtype=float)
    rev = rev.sort_values(["stock_id", "revenue_year", "revenue_month"])
    g = rev.groupby("stock_id")["revenue"]
    n = g.size()
    avg3 = g.apply(lambda s: s.iloc[-3:].mean())
    avg12 = g.mean()
    rev_up = (avg3 > avg12) & (n >= 12)
    ok = trend & rev_up.reindex(trend.index).fillna(False).astype(bool)
    if liquid_shares is not None:
        ok &= ctx.volume(20).mean() > liquid_shares
    return pct[ok].replace([np.inf, -np.inf], np.nan).dropna()


def revmom(n: int = 10, liquid_shares: float | None = None, universe=None, name="revmom"):
    """Equal-weight the ``n`` stocks with the strongest 5-day momentum among those passing.

    ``universe``: optional callable(ctx) -> ids to restrict the candidates.
    """
    def strategy(ctx: Context) -> dict[str, float]:
        s = revmom_scores(ctx, liquid_shares)
        allowed = set(ctx.universe()) if universe is None else set(universe(ctx))
        s = s[s.index.isin(allowed)]
        return {sid: 1.0 / n for sid in rank_top(s, n)}

    strategy.__name__ = name
    return strategy
