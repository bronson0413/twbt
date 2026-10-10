"""Diagnostics for the valuation data and the stocks each signal actually picks.

Printed as plain text for the workflow report. Used to check that the
exchange ratios look sane per market and year, and to see what a signal
buys when its portfolio does unexpectedly badly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.query import Data
from ..engine import MarketData
from ..engine.context import Context
from ..engine.schedule import monthly_on_or_after
from ..strategies import eligible
from ..strategies.combo import rank_top
from .search import candidates


def ratio_summary(data: Data) -> str:
    q = """
        SELECT u.market, EXTRACT(year FROM v.date) AS year,
               COUNT(DISTINCT v.stock_id) AS stocks,
               quantile_cont(v.pe, 0.5) AS pe_med, quantile_cont(v.pb, 0.05) AS pb_p5,
               quantile_cont(v.pb, 0.5) AS pb_med, quantile_cont(v.pb, 0.95) AS pb_p95,
               quantile_cont(v.dividend_yield, 0.5) AS dy_med,
               quantile_cont(v.dividend_yield, 0.99) AS dy_p99,
               AVG(CASE WHEN v.pe IS NULL THEN 1 ELSE 0 END) AS pe_missing
        FROM valuation v JOIN universe u USING (stock_id)
        WHERE EXTRACT(month FROM v.date) = 6 AND EXTRACT(day FROM v.date) BETWEEN 10 AND 16
        GROUP BY 1, 2 ORDER BY 1, 2
    """
    df = data.sql(q)
    return "valuation by market and year (mid-June snapshot):\n" + df.to_string(
        index=False, float_format=lambda x: f"{x:.2f}")


def spot(data: Data, stocks: list[str]) -> str:
    q = """
        SELECT v.stock_id, v.date, v.pe, v.pb, v.dividend_yield, p.close
        FROM valuation v LEFT JOIN prices p USING (stock_id, date)
        WHERE v.stock_id IN (SELECT UNNEST(?)) AND EXTRACT(day FROM v.date) BETWEEN 10 AND 12
          AND EXTRACT(month FROM v.date) IN (3, 9) AND v.date >= '2019-01-01'
        ORDER BY 1, 2
    """
    return "spot check:\n" + data.sql(q, [stocks]).to_string(index=False)


def picks(data: Data, name: str, start: str, end: str, n: int = 30,
          show: tuple[str, ...] = ("2015-06-01", "2019-06-01")) -> str:
    """Who the signal holds: a few full lists, plus averages over every signal date."""
    market = MarketData.from_data(data, start, end, benchmarks=())
    days = market.dates[(market.dates >= pd.Timestamp(start)) & (market.dates <= pd.Timestamp(end))]
    sched = monthly_on_or_after(days, 11)
    scorer = candidates()[name]
    uni = data.sql("SELECT stock_id, stock_name, market, industry, delisted_date FROM universe")
    uni = uni.set_index("stock_id")
    tri = market.tri
    out = [f"=== {name} top-{n} ==="]
    agg = []
    for i, d in enumerate(sched[:-1]):
        t0, t1 = market.index_of(d), market.index_of(sched[i + 1])
        ctx = Context(market, t0)
        ok = set(eligible(ctx))
        s = scorer(ctx)
        top = rank_top(s[s.index.isin(ok)], n)
        if not top:
            continue
        cols = [market.col[x] for x in top]
        fwd = tri[t1, cols] / tri[t0, cols] - 1
        allc = [market.col[x] for x in ok]
        uni_fwd = np.nanmean(tri[t1, allc] / tri[t0, allc] - 1)
        pe, pb, dy = ctx.valuation("pe"), ctx.valuation("pb"), ctx.valuation("dividend_yield")
        val20 = ctx.value(20).mean()
        agg.append({"date": d, "fwd": float(np.nanmean(fwd)), "uni": float(uni_fwd),
                    "tpex": float(np.mean([uni["market"].get(x) == "tpex" for x in top])),
                    "pb_med": float(pb.reindex(top).median()),
                    "pe_med": float(pe.reindex(top).median()),
                    "dy_med": float(dy.reindex(top).median()),
                    "value_med_m": float(val20.reindex(top).median() / 1e6),
                    "delisted_later": float(np.mean([pd.notna(uni["delisted_date"].get(x))
                                                     for x in top]))})
        if any(abs((d - pd.Timestamp(x)).days) < 20 for x in show):
            rows = pd.DataFrame({
                "stock_id": top, "name": [uni["stock_name"].get(x, "") for x in top],
                "mkt": [uni["market"].get(x, "") for x in top],
                "industry": [str(uni["industry"].get(x, ""))[:6] for x in top],
                "pe": pe.reindex(top).to_numpy(), "pb": pb.reindex(top).to_numpy(),
                "yield": dy.reindex(top).to_numpy(), "score": s.reindex(top).to_numpy(),
                "value_M": (val20.reindex(top) / 1e6).to_numpy(), "fwd_1m": fwd})
            out.append(f"-- {d.date()} (universe next month {uni_fwd:+.2%}) --")
            out.append(rows.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    a = pd.DataFrame(agg).set_index("date")
    out.append("-- averages per year --")
    yearly = a.groupby(a.index.year).agg(
        picks_fwd=("fwd", "mean"), uni_fwd=("uni", "mean"), tpex=("tpex", "mean"),
        pb_med=("pb_med", "median"), pe_med=("pe_med", "median"), dy_med=("dy_med", "median"),
        value_M=("value_med_m", "median"), delisted=("delisted_later", "mean"))
    out.append(yearly.to_string(float_format=lambda x: f"{x:.3f}"))
    return "\n".join(out)


def inspect(data: Data, names: list[str], start: str = "2012-01-01",
            end: str = "2022-12-31") -> str:
    parts = [ratio_summary(data), spot(data, ["2330", "1101", "2412", "6488", "5483"])]
    parts += [picks(data, nm, start, end) for nm in names]
    return "\n\n".join(parts)
