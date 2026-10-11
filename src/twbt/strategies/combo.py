"""Generic top-N portfolio and rank combination of several scores."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import pandas as pd

from ..engine.context import Context
from .revenue_momentum import DEFAULT_EXCLUDED_INDUSTRIES, eligible

Scorer = Callable[[Context], pd.Series]


def rank_combo(scorers: Iterable[Scorer], weights: Iterable[float] | None = None) -> Scorer:
    """Weighted average of cross-sectional percentile ranks.

    Only stocks that have every score are ranked. Ranking first makes scores on
    very different scales (a growth rate, a price return) comparable and keeps
    outliers from dominating.
    """
    scorers = list(scorers)
    weights = list(weights) if weights is not None else [1.0] * len(scorers)
    if len(weights) != len(scorers):
        raise ValueError("one weight per scorer")

    def score(ctx: Context) -> pd.Series:
        raw = pd.concat([f(ctx) for f in scorers], axis=1).dropna()
        ranks = raw.rank(pct=True)  # ranked within the common set only
        return (ranks * weights).sum(axis=1) / sum(weights)

    return score


def rank_top(scores: pd.Series, n: int) -> list[str]:
    """The ``n`` highest scores; ties broken by stock id so results are reproducible.

    Rank-averaged scores tie often, and an unstable sort would otherwise let
    row order (e.g. from a database query) decide which stocks are held.
    """
    df = pd.DataFrame({"s": scores.to_numpy(), "id": scores.index.astype(str)})
    df = df.sort_values(["s", "id"], ascending=[False, True], kind="mergesort")
    return df["id"].head(n).tolist()


def top_n(
    scorer: Scorer,
    n: int = 30,
    min_avg_value: float = 10_000_000,
    exclude_industries: Iterable[str] = DEFAULT_EXCLUDED_INDUSTRIES,
    name: str = "top_n",
    top_value: int | None = None,
    keep_rank: int | None = None,
    held: Iterable[str] = (),
):
    """Equal-weight the ``n`` highest-scoring eligible stocks (rest in cash).

    ``keep_rank``: a buffer against needless trading. A stock already held
    stays while it still ranks within the top ``keep_rank``; only the freed
    slots are refilled with the best-ranked stocks. The strategy remembers
    what it picked last time (``held`` seeds that memory for live use).
    """
    banned = tuple(exclude_industries)
    state = {"held": list(held)}

    def strategy(ctx: Context) -> dict[str, float]:
        ok = set(eligible(ctx, min_avg_value, exclude_industries=banned, top_value=top_value))
        s = scorer(ctx)
        s = s[s.index.isin(ok)]
        if keep_rank and keep_rank > n:
            ranked = rank_top(s, len(s))
            band = set(ranked[:keep_rank])
            keep = [h for h in state["held"] if h in band][:n]
            picks = keep + [x for x in ranked if x not in keep][: n - len(keep)]
        else:
            picks = rank_top(s, n)
        state["held"] = picks
        return {sid: 1.0 / n for sid in picks}

    strategy.__name__ = name
    return strategy
