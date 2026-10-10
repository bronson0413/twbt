import pandas as pd
import pytest
from conftest import make_client

from twbt.data.clean import build
from twbt.data.query import Data
from twbt.data.store import Manifest, RawStore
from twbt.data.update import Updater
from twbt.engine import BacktestConfig, Context, MarketData, run_backtest
from twbt.strategies import buy_and_hold, revenue_momentum, revenue_scores

DAY = pd.Timestamp("2024-04-12")


def revenue_rows(sid, growth, last_month=3, year=2024, base=100):
    """Three months of revenue in `year` growing `growth` vs the year before."""
    rows = []
    for m in range(last_month - 2, last_month + 1):
        for y, rev in ((year - 1, base), (year, base * (1 + growth))):
            rows.append({"stock_id": sid, "revenue_year": y, "revenue_month": m,
                         "revenue": rev, "period": f"{y}-{m:02d}",
                         "available_date": pd.Timestamp(y, m, 11) + pd.DateOffset(months=1)})
    return rows


def ctx_with(revenue, value=1e9, ids=("A", "B", "C", "D")):
    days = pd.bdate_range("2024-03-01", DAY)
    rows = [{"date": d, "stock_id": s, "open": 10, "high": 10, "low": 10, "close": 10,
             "value": value if s != "D" else 1.0} for d in days for s in ids]
    m = MarketData(pd.DataFrame(rows), pd.DataFrame({"stock_id": list(ids)}),
                   revenue=pd.DataFrame(revenue), trading_days=days)
    return Context(m, len(days) - 1)


def test_scores_are_mean_yoy():
    ctx = ctx_with(revenue_rows("A", 0.5) + revenue_rows("B", 0.1))
    s = revenue_scores(ctx)
    assert s["A"] == pytest.approx(0.5) and s["B"] == pytest.approx(0.1)


def test_stale_reporter_is_skipped():
    # latest month published is 2023-12: too old on 2024-04-12
    ctx = ctx_with(revenue_rows("A", 0.5, last_month=12, year=2023))
    assert revenue_scores(ctx).empty


def test_picks_top_growth_and_filters_illiquid():
    rows = (revenue_rows("A", 0.5) + revenue_rows("B", 0.1) + revenue_rows("C", 0.3)
            + revenue_rows("D", 9.0))  # D grows most but trades NT$1/day
    weights = revenue_momentum(top_n=2)(ctx_with(rows))
    assert weights == {"A": 0.5, "C": 0.5}


def test_fewer_qualifiers_leave_cash():
    weights = revenue_momentum(top_n=4)(ctx_with(revenue_rows("A", 0.5)))
    assert weights == {"A": 0.25}


def test_from_data_plumbing(server, tmp_path, fixed_now):
    Updater(make_client(server), RawStore(tmp_path), Manifest(tmp_path),
            end="2026-10-08", now=lambda: fixed_now["now"]).run()
    build(tmp_path)
    m = MarketData.from_data(Data(tmp_path), "2024-04-10", "2024-04-11")
    assert "0050" in m.col and "0050" not in m.universe_ids
    r = run_backtest(buy_and_hold("2330"), m, "2024-04-10", "2024-04-11",
                     [pd.Timestamp("2024-04-10")], BacktestConfig(capital=100_000))
    # signal on 04-10, but 2330 has no quote on 04-11: nothing can fill
    assert r.trades.empty and r.equity.iloc[-1] == 100_000


def test_equal_weight_skips_illiquid():
    from twbt.strategies import equal_weight
    w = equal_weight()(ctx_with([]))
    assert w == pytest.approx({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})


def _rows(sid, cur, last, year=2024, months=(1, 2, 3)):
    out = []
    for m, c, ly in zip(months, cur, last, strict=True):
        for y, v in ((year - 1, ly), (year, c)):
            out.append({"stock_id": sid, "revenue_year": y, "revenue_month": m, "revenue": v,
                        "period": f"{y}-{m:02d}",
                        "available_date": pd.Timestamp(y, m, 11) + pd.DateOffset(months=1)})
    return out


def test_sum_method_resists_tiny_base():
    # A: one month vs a near-zero base (+4900%), other months shrinking
    # B: steady +30%
    rows = _rows("A", [500, 50, 50], [10, 100, 100]) + _rows("B", [130, 130, 130],
                                                              [100, 100, 100])
    ctx = ctx_with(rows)
    mean = revenue_scores(ctx, method="mean")
    total = revenue_scores(ctx, method="sum")
    assert mean["A"] > mean["B"]                 # mean is fooled
    assert total["A"] == pytest.approx(600 / 210 - 1)
    assert total["B"] == pytest.approx(0.3)


def test_consecutive_growth_filter():
    rows = _rows("A", [200, 200, 90], [100, 100, 100]) + _rows("B", [110, 110, 110],
                                                              [100, 100, 100])
    s = revenue_scores(ctx_with(rows), min_consecutive=3)
    assert list(s.index) == ["B"]                # A's latest month shrank


def test_construction_excluded():
    from twbt.engine import Context, MarketData
    days = pd.bdate_range("2024-03-01", DAY)
    px = pd.DataFrame([{"date": d, "stock_id": s, "open": 10, "high": 10, "low": 10,
                        "close": 10, "value": 1e9} for d in days for s in ("A", "B")])
    uni = pd.DataFrame({"stock_id": ["A", "B"], "industry": ["建材營造業", "半導體業"]})
    rev = pd.DataFrame(revenue_rows("A", 5.0) + revenue_rows("B", 0.1))
    ctx = Context(MarketData(px, uni, revenue=rev, trading_days=days), len(days) - 1)
    assert revenue_momentum(top_n=2)(ctx) == {"B": 0.5}
    assert set(revenue_momentum(top_n=2, exclude_industries=())(ctx)) == {"A", "B"}
