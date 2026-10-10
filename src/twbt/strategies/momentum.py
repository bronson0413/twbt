"""Price momentum (Jegadeesh & Titman, 1993).

Score = total return from ``lookback`` to ``skip`` trading days ago. Skipping
the most recent month avoids the well-documented one-month reversal.
"""

from __future__ import annotations

import pandas as pd

from ..engine.context import Context


def momentum_scores(ctx: Context, lookback: int = 252, skip: int = 21) -> pd.Series:
    if skip >= lookback:
        raise ValueError("skip must be shorter than lookback")
    tr = ctx.total_return(lookback + 1)
    if len(tr) < lookback + 1:
        return pd.Series(dtype=float)
    start, end = tr.iloc[0], tr.iloc[-1 - skip]
    ids = ctx.has_history(lookback)
    return (end[ids] / start[ids] - 1).dropna()
