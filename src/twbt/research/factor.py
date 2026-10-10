"""Quintile and rank-IC analysis of a stock score.

On every signal date, all eligible stocks are ranked by the score and split
into ``groups`` equal-sized buckets. Each bucket's equal-weighted total return
until the next signal date is recorded. A useful signal shows returns rising
steadily from the lowest to the highest bucket; a useless one shows a flat or
random pattern.

This measures the signal itself, on the whole cross-section, so it is much
less noisy than a backtest that only ever holds the top few names. It ignores
costs and uses close-to-close returns, so it is optimistic by design: a signal
that fails here will not work after costs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..engine.context import Context
from ..engine.market import MarketData

Scorer = Callable[[Context], pd.Series]
Universe = Callable[[Context], list[str]]


def total_return_index(market: MarketData) -> np.ndarray:
    """See :attr:`MarketData.tri`."""
    return market.tri


@dataclass
class FactorReport:
    by_date: pd.DataFrame  # one row per signal date: bucket returns, universe, ic, n
    groups: int

    @property
    def bucket_cols(self) -> list[str]:
        return [f"Q{i}" for i in range(1, self.groups + 1)]

    def summary(self) -> dict[str, float]:
        d = self.by_date
        q = d[self.bucket_cols]
        top, bottom = f"Q{self.groups}", "Q1"
        spread = d[top] - d[bottom]
        ic = d["ic"].dropna()
        months_per_year = 12
        out = {f"{c}_monthly": float(q[c].mean()) for c in self.bucket_cols}
        out.update({
            "universe_monthly": float(d["universe"].mean()),
            "top_minus_bottom_monthly": float(spread.mean()),
            "top_minus_bottom_tstat": float(spread.mean() / spread.std() * np.sqrt(len(spread)))
            if spread.std() > 0 else np.nan,
            "top_beats_universe_rate": float((d[top] > d["universe"]).mean()),
            "ic_mean": float(ic.mean()),
            "ic_tstat": float(ic.mean() / ic.std() * np.sqrt(len(ic))) if ic.std() > 0 else np.nan,
            "ic_positive_rate": float((ic > 0).mean()),
            "top_annualized": float((1 + d[top]).prod() ** (months_per_year / len(d)) - 1),
            "universe_annualized": float((1 + d["universe"]).prod() ** (months_per_year / len(d))
                                         - 1),
            "periods": len(d),
            "avg_stocks": float(d["n"].mean()),
        })
        return out

    def yearly(self) -> pd.DataFrame:
        """Compounded return per calendar year for each bucket and the universe."""
        d = self.by_date
        cols = self.bucket_cols + ["universe"]
        return (1 + d[cols]).groupby(d.index.year).prod() - 1


def factor_analysis(
    market: MarketData,
    signal_dates: Iterable,
    scorer: Scorer,
    universe: Universe,
    groups: int = 5,
    min_stocks: int = 50,
) -> FactorReport:
    tri = market.tri
    n_days = len(market.dates)
    idx = sorted({t for t in (market.index_of(d) for d in signal_dates) if t < n_days})
    rows = []
    for t0, t1 in zip(idx[:-1], idx[1:], strict=True):
        ctx = Context(market, t0)
        ok = universe(ctx)
        scores = scorer(ctx)
        scores = scores[scores.index.isin(ok)].dropna()
        if len(scores) < min_stocks:
            continue
        cols = [market.col[s] for s in scores.index]
        fwd = pd.Series(tri[t1, cols] / tri[t0, cols] - 1, index=scores.index)
        uni_cols = [market.col[s] for s in ok]
        uni_ret = float(np.nanmean(tri[t1, uni_cols] / tri[t0, uni_cols] - 1))
        # rank first so ties don't break qcut
        bucket = pd.qcut(scores.rank(method="first"), groups, labels=False) + 1
        means = fwd.groupby(bucket).mean()
        row = {f"Q{int(k)}": float(v) for k, v in means.items()}
        row.update({
            "date": market.dates[t0],
            "universe": uni_ret,
            "ic": float(scores.rank().corr(fwd.rank())),
            "n": len(scores),
        })
        rows.append(row)
    if not rows:
        raise RuntimeError("沒有足夠的股票可以分組（資料太少或條件太嚴）")
    by_date = pd.DataFrame(rows).set_index("date")
    return FactorReport(by_date=by_date, groups=groups)


def format_report(name: str, rep: FactorReport) -> str:
    s = rep.summary()
    g = rep.groups
    buckets = "  ".join(f"Q{i} {s[f'Q{i}_monthly']:+.2%}" for i in range(1, g + 1))
    lines = [
        f"[{name}]  {s['periods']} 期，平均每期 {s['avg_stocks']:.0f} 檔",
        f"  各組平均月報酬（Q1 分數最低，Q{g} 最高）: {buckets}",
        f"  全體平均 {s['universe_monthly']:+.2%}；Q{g}-Q1 {s['top_minus_bottom_monthly']:+.2%}"
        f"（t={s['top_minus_bottom_tstat']:.2f}）",
        f"  Rank IC 平均 {s['ic_mean']:+.3f}（t={s['ic_tstat']:.2f}，為正的比例 "
        f"{s['ic_positive_rate']:.0%}）",
        f"  Q{g} 年化 {s['top_annualized']:+.1%} vs 全體 {s['universe_annualized']:+.1%}；"
        f"Q{g} 勝過全體的月份 {s['top_beats_universe_rate']:.0%}",
    ]
    return "\n".join(lines)
