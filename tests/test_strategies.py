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
