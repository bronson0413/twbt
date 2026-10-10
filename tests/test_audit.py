import pandas as pd

from twbt.engine import MarketData
from twbt.research.audit import impossible_moves


def _m(closes, actions=None, start="2024-01-01"):
    days = pd.bdate_range(start, periods=len(closes))
    rows = [{"date": d, "stock_id": "A", "open": c, "high": c, "low": c, "close": c, "value": 1}
            for d, c in zip(days, closes, strict=True)]
    return MarketData(pd.DataFrame(rows), pd.DataFrame({"stock_id": ["A"]}), actions=actions,
                      trading_days=days), days


def test_flags_jump_without_event():
    m, _ = _m([10] * 8 + [20, 20])
    bad = impossible_moves(m)
    assert len(bad) == 1 and bad.loc[0, "move"] == 1.0


def test_capital_reduction_is_not_flagged():
    m, days = _m([10] * 8 + [20, 20])
    ev = pd.DataFrame([{"date": days[8], "stock_id": "A", "kind": "capital_reduction",
                        "detail": "", "before_price": 10.0, "reference_price": 20.0,
                        "factor": 0.5}])
    m2 = MarketData(pd.DataFrame([{"date": d, "stock_id": "A", "open": c, "high": c, "low": c,
                                   "close": c, "value": 1}
                                  for d, c in zip(days, [10] * 8 + [20, 20], strict=True)]),
                    pd.DataFrame({"stock_id": ["A"]}), actions=ev, trading_days=days)
    assert impossible_moves(m2).empty and len(impossible_moves(m)) == 1


def test_limit_moves_and_new_listings_allowed():
    assert impossible_moves(_m([10] * 6 + [11, 12.1])[0]).empty      # +10% days
    assert impossible_moves(_m([10, 20, 40] + [40] * 5)[0]).empty     # first 5 days free
    assert len(impossible_moves(_m([10] * 6 + [10.8], start="2014-01-01")[0])) == 1  # 7% era
