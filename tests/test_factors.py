import numpy as np
import pandas as pd
import pytest

from twbt.engine.context import Context
from twbt.engine.market import MarketData
from twbt.strategies import inst_scores, lowvol_scores, quality_scores, value_scores, yield_scores


def market(n_days=300, val=None, flows=None, closes=None):
    days = pd.bdate_range("2023-01-02", periods=n_days)
    ids = ["A", "B", "C"]
    closes = closes or {s: np.full(n_days, 10.0) for s in ids}
    prices = pd.DataFrame([{"date": d, "stock_id": s, "open": closes[s][i], "high": closes[s][i],
                            "low": closes[s][i], "close": closes[s][i], "volume": 1000,
                            "value": 10_000}
                           for i, d in enumerate(days) for s in ids])
    uni = pd.DataFrame({"stock_id": ids, "industry": "x", "delisted_date": pd.NaT})
    if val is not None:
        val = pd.DataFrame([{"date": days[-1], "stock_id": s, "pe": pe, "pb": pb,
                             "dividend_yield": y} for s, (pe, pb, y) in val.items()])
    if flows is not None:
        flows = pd.DataFrame([{"date": days[-1], "stock_id": s, "foreign_net": f,
                               "trust_net": t} for s, (f, t) in flows.items()])
    m = MarketData(prices, uni, valuation=val, flows=flows, trading_days=days)
    return Context(m, n_days - 1)


VAL = {"A": (10.0, 1.0, 5.0),        # cheap, ROE 10%
       "B": (40.0, 8.0, 1.0),        # expensive, ROE 20%
       "C": (np.nan, 0.5, 0.0)}      # loss maker, very low P/B


def test_value_prefers_cheap_and_counts_losses_as_zero_earnings_yield():
    s = value_scores(market(val=VAL))
    # E/P ranks A > B > C (C's loss counts as 0); B/P ranks C > A > B
    assert s["A"] == pytest.approx((1 + 2 / 3) / 2)
    assert s["B"] == pytest.approx((2 / 3 + 1 / 3) / 2)
    assert s["C"] == pytest.approx((1 / 3 + 1) / 2)


def test_quality_is_roe_from_ratios():
    s = quality_scores(market(val=VAL))
    assert s["A"] == pytest.approx(0.1) and s["B"] == pytest.approx(0.2) and s["C"] == 0


def test_yield_reads_published_yield():
    assert yield_scores(market(val=VAL)).to_dict() == {"A": 5.0, "B": 1.0, "C": 0.0}


def test_lowvol_ranks_calm_stock_higher():
    rng = np.random.default_rng(0)
    n = 300
    closes = {"A": 10 * np.cumprod(1 + rng.normal(0, 0.005, n)),
              "B": 10 * np.cumprod(1 + rng.normal(0, 0.03, n)),
              "C": np.full(n, 10.0)}
    s = lowvol_scores(market(closes=closes))
    assert s["A"] > s["B"]


def test_lowvol_needs_a_year_of_history():
    assert lowvol_scores(market(n_days=100)).empty


def test_inst_scales_net_buying_by_volume():
    s = inst_scores(market(flows={"A": (5000.0, 5000.0), "B": (-2000.0, 0.0)}))
    assert s["A"] == pytest.approx(10_000 / 20_000)
    assert s["B"] == pytest.approx(-2000 / 20_000)
    assert s["C"] == 0
