import numpy as np
import pandas as pd
import pytest

from twbt.engine import Context, MarketData
from twbt.strategies import momentum_scores, rank_combo, top_n


def _market(closes: dict[str, list[float]], actions=None):
    n = len(next(iter(closes.values())))
    days = pd.bdate_range("2023-01-02", periods=n)
    rows = [{"date": d, "stock_id": s, "open": c, "high": c, "low": c, "close": c, "value": 1e9}
            for s, series in closes.items() for d, c in zip(days, series, strict=True)]
    return MarketData(pd.DataFrame(rows), pd.DataFrame({"stock_id": list(closes)}),
                      actions=actions, trading_days=days)


def test_momentum_skips_last_month():
    # A doubles over the year then crashes in the final month; B flat then jumps
    a = list(np.linspace(100, 200, 232)) + [100.0] * 21
    b = [100.0] * 232 + [300.0] * 21
    m = _market({"A": a, "B": b})
    s = momentum_scores(Context(m, 252), lookback=252, skip=21)
    assert s["A"] == pytest.approx(1.0, rel=1e-3)  # crash in skipped month ignored
    assert s["B"] == pytest.approx(0.0)


def test_momentum_uses_total_return():
    # price halves on a 2:1 split → no momentum lost
    closes = [100.0] * 100 + [50.0] * 153
    split = pd.DataFrame([{"date": pd.bdate_range("2023-01-02", periods=253)[100],
                           "stock_id": "A", "kind": "split", "detail": "",
                           "before_price": 100.0, "reference_price": 50.0, "factor": 2.0}])
    s = momentum_scores(Context(_market({"A": closes}, split), 252))
    assert s["A"] == pytest.approx(0.0)


def test_momentum_needs_full_history():
    m = _market({"A": [np.nan] * 100 + [10.0] * 153, "B": [10.0] * 253})
    assert list(momentum_scores(Context(m, 252)).index) == ["B"]


def test_rank_combo_averages_percentile_ranks():
    s1 = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0})
    s2 = pd.Series({"A": 300.0, "B": 200.0, "C": 100.0, "D": 999.0})
    combo = rank_combo([lambda c: s1, lambda c: s2])(None)
    assert set(combo.index) == {"A", "B", "C"}  # D lacks the first score
    assert combo["A"] == pytest.approx(combo["C"])  # opposite ranks cancel
    w = rank_combo([lambda c: s1, lambda c: s2], weights=[3, 1])(None)
    assert w.idxmax() == "C"


def test_top_n_equal_weights_best_scores():
    m = _market({s: [10.0] * 30 for s in "ABCD"})
    scores = pd.Series({"A": 1, "B": 4, "C": 3, "D": 2})
    w = top_n(lambda ctx: scores, n=2)(Context(m, 29))
    assert w == {"B": 0.5, "C": 0.5}


def test_rank_top_breaks_ties_by_id_regardless_of_order():
    from twbt.strategies.combo import rank_top
    s = pd.Series({"B": 0.5, "A": 0.5, "D": 0.9, "C": 0.5})
    assert rank_top(s, 3) == ["D", "A", "B"]
    assert rank_top(s.sample(frac=1, random_state=3), 3) == ["D", "A", "B"]
