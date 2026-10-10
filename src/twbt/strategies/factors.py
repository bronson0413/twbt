"""Classic cross-sectional factors, each defined once from the literature.

Every scorer returns a Series indexed by stock id where **higher is better**.
Definitions are fixed before any backtest; none of the parameters was tuned.

* VALUE   — average percentile rank of book-to-price and earnings-to-price
            (Fama & French 1992; Basu 1977). Loss makers get E/P = 0.
* QUALITY — return on equity, from the exchange's own daily ratios:
            P/B ÷ P/E = (P/B)·(E/P) = E/B. Loss makers get 0. Using the
            exchange ratios avoids guessing when each financial report became
            public (Novy-Marx 2013 and Asness et al. 2019 for quality).
* YIELD   — cash dividend yield as published by the exchange.
* LOWVOL  — minus the volatility of daily total returns over a year
            (Ang et al. 2006; Baker et al. 2011).
* INST    — foreign + investment-trust net buying over 20 days as a share of
            traded volume, a Taiwan-specific "smart money" flow signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine.context import Context


def _ratios(ctx: Context) -> pd.DataFrame:
    """P/E and P/B for stocks with a published P/B (P/E is missing for loss makers)."""
    pb = ctx.valuation("pb")
    pe = ctx.valuation("pe").reindex(pb.index)
    return pd.DataFrame({"pe": pe, "pb": pb})


def value_scores(ctx: Context) -> pd.Series:
    r = _ratios(ctx)
    if r.empty:
        return pd.Series(dtype=float)
    ep = (1 / r["pe"]).fillna(0.0)
    bp = 1 / r["pb"]
    return (ep.rank(pct=True) + bp.rank(pct=True)) / 2


def quality_scores(ctx: Context) -> pd.Series:
    r = _ratios(ctx)
    if r.empty:
        return pd.Series(dtype=float)
    return (r["pb"] / r["pe"]).fillna(0.0)


def yield_scores(ctx: Context) -> pd.Series:
    return ctx.valuation("dividend_yield")


def lowvol_scores(ctx: Context, days: int = 252, min_obs: int = 200) -> pd.Series:
    tr = ctx.total_return(days + 1)
    if len(tr) < days + 1:
        return pd.Series(dtype=float)
    r = tr.pct_change().iloc[1:]
    # a flat total-return index means no quote that day, not a zero return
    r = r.where(ctx.close(days).notna().to_numpy())
    ids = ctx.has_history(days)
    r = r[ids]
    ok = r.notna().sum() >= min_obs
    return -r.loc[:, ok].std()


def inst_scores(ctx: Context, days: int = 20) -> pd.Series:
    net = ctx.flows(days).sum()
    vol = ctx.volume(days).sum()
    ok = vol > 0
    out = (net[ok] / vol[ok]).replace([np.inf, -np.inf], np.nan).dropna()
    return out
