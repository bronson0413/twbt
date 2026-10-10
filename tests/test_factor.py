import numpy as np
import pandas as pd
import pytest

from twbt.engine import MarketData
from twbt.research.factor import factor_analysis, total_return_index


def test_total_return_index_handles_dividend_and_gap():
    days = pd.bdate_range("2024-01-01", periods=4)
    px = pd.DataFrame({"date": days, "stock_id": "A", "open": 1, "high": 1, "low": 1,
                       "close": [100.0, np.nan, 95.0, 95.0], "value": 1})
    ev = pd.DataFrame([{"date": days[1], "stock_id": "A", "kind": "dividend", "detail": "息",
                        "before_price": 100.0, "reference_price": 95.0,
                        "factor": 100 / 95}])
    m = MarketData(px, pd.DataFrame({"stock_id": ["A"]}), actions=ev, trading_days=days)
    tri = total_return_index(m)[:, 0]
    # dividend day had no quote: the return still adds up to zero over the gap
    assert tri[-1] == pytest.approx(1.0)


def test_quintiles_recover_a_true_signal():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2020-01-01", periods=400)
    n = 200
    ids = [f"{1000 + i}" for i in range(n)]
    skill = np.linspace(-0.01, 0.01, n)  # daily drift known in advance
    rets = skill + rng.normal(0, 0.01, (len(days), n))
    close = 100 * np.exp(np.cumsum(rets, axis=0))
    px = pd.DataFrame({"date": np.repeat(days, n), "stock_id": np.tile(ids, len(days)),
                       "close": close.ravel()})
    px["open"] = px["high"] = px["low"] = px["close"]
    px["value"] = 1e9
    m = MarketData(px, pd.DataFrame({"stock_id": ids}), trading_days=days)
    score = pd.Series(skill, index=ids)
    rep = factor_analysis(m, days[::20], lambda ctx: score, lambda ctx: ids)
    s = rep.summary()
    q = [s[f"Q{i}_monthly"] for i in range(1, 6)]
    assert q == sorted(q)                       # monotonic
    assert s["ic_mean"] > 0.5 and s["top_minus_bottom_tstat"] > 5


def test_random_signal_is_flat():
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2020-01-01", periods=400)
    n = 200
    ids = [f"{1000 + i}" for i in range(n)]
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(days), n)), axis=0))
    px = pd.DataFrame({"date": np.repeat(days, n), "stock_id": np.tile(ids, len(days)),
                       "close": close.ravel()})
    px["open"] = px["high"] = px["low"] = px["close"]
    px["value"] = 1e9
    m = MarketData(px, pd.DataFrame({"stock_id": ids}), trading_days=days)
    rep = factor_analysis(m, days[::20], lambda ctx: pd.Series(rng.random(n), index=ids),
                          lambda ctx: ids)
    assert abs(rep.summary()["ic_tstat"]) < 3
