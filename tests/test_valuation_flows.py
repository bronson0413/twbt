import numpy as np
import pandas as pd

from twbt.data.clean import clean_institutional, clean_valuation
from twbt.engine.context import Context
from twbt.engine.market import MarketData


def test_valuation_zero_pe_means_loss_not_cheap():
    raw = pd.DataFrame({"date": ["2024-01-02", "2024-01-03"], "stock_id": ["1101", "1101"],
                        "PER": [12.5, 0.0], "PBR": [1.2, 0.0], "dividend_yield": [4.0, 0.0]})
    df = clean_valuation(raw)
    assert df["pe"].tolist()[0] == 12.5 and np.isnan(df["pe"].iloc[1])
    assert np.isnan(df["pb"].iloc[1])
    assert df["dividend_yield"].tolist() == [4.0, 0.0]  # no dividend is a real 0


def test_institutional_sums_foreign_types_and_drops_dealers():
    rows = [("Foreign_Investor", 1000, 200), ("Foreign_Dealer_Self", 50, 0),
            ("Investment_Trust", 0, 300), ("Dealer_self", 9999, 0), ("Dealer_Hedging", 0, 9999)]
    raw = pd.DataFrame([{"date": "2024-01-02", "stock_id": "2330", "name": n, "buy": b,
                         "sell": s} for n, b, s in rows])
    df = clean_institutional(raw)
    assert len(df) == 1
    assert df.loc[0, "foreign_net"] == 850 and df.loc[0, "trust_net"] == -300


def test_institutional_missing_trust_column_is_zero():
    raw = pd.DataFrame([{"date": "2012-01-02", "stock_id": "1101", "name": "Foreign_Investor",
                         "buy": 10, "sell": 0}])
    assert clean_institutional(raw).loc[0, "trust_net"] == 0


def _market():
    days = pd.bdate_range("2024-01-01", periods=15)
    prices = pd.DataFrame([{"date": d, "stock_id": s, "open": 10.0, "high": 10.0, "low": 10.0,
                            "close": 10.0, "volume": 1000, "value": 10000}
                           for d in days for s in ("A", "B")])
    uni = pd.DataFrame({"stock_id": ["A", "B"], "industry": ["x", "y"],
                        "delisted_date": [pd.NaT, pd.NaT]})
    val = pd.DataFrame([{"date": days[0], "stock_id": "A", "pe": 10.0, "pb": 1.0,
                         "dividend_yield": 3.0},
                        {"date": days[12], "stock_id": "B", "pe": 20.0, "pb": 2.0,
                         "dividend_yield": 1.0},
                        {"date": days[14], "stock_id": "B", "pe": 99.0, "pb": 9.0,
                         "dividend_yield": 0.0}])
    flows = pd.DataFrame([{"date": days[13], "stock_id": "A", "foreign_net": 100.0,
                           "trust_net": 50.0}])
    return days, MarketData(prices, uni, valuation=val, flows=flows, trading_days=days)


def test_valuation_is_point_in_time_and_drops_stale_values():
    days, m = _market()
    ctx = Context(m, 13)
    pe = ctx.valuation("pe")
    assert pe.to_dict() == {"B": 20.0}  # A's value is 13 days old; B's day-14 row is future


def test_flows_window_treats_missing_days_as_zero():
    _, m = _market()
    f = Context(m, 14).flows(5)
    assert f["A"].sum() == 150 and f["B"].sum() == 0


def test_loss_maker_pe_is_not_filled_from_an_earlier_profitable_day():
    days = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame([{"date": d, "stock_id": "A", "open": 10.0, "high": 10.0, "low": 10.0,
                            "close": 10.0, "volume": 1, "value": 10} for d in days])
    uni = pd.DataFrame({"stock_id": ["A"], "industry": ["x"], "delisted_date": [pd.NaT]})
    val = pd.DataFrame({"date": days[:2], "stock_id": "A", "pe": [15.0, np.nan],
                        "pb": [1.5, 1.4], "dividend_yield": [2.0, 2.0]})
    ctx = Context(MarketData(prices, uni, valuation=val, trading_days=days), 2)
    assert ctx.valuation("pe").empty
    assert ctx.valuation("pb").to_dict() == {"A": 1.4}


def test_live_context_sees_revenue_published_after_last_price():
    days = pd.bdate_range("2024-11-04", periods=4)  # last price 2024-11-07
    prices = pd.DataFrame([{"date": d, "stock_id": "A", "open": 10.0, "high": 10.0, "low": 10.0,
                            "close": 10.0, "volume": 1, "value": 10} for d in days])
    uni = pd.DataFrame({"stock_id": ["A"], "industry": ["x"], "delisted_date": [pd.NaT]})
    rev = pd.DataFrame({"stock_id": ["A"], "period": ["2024-10"], "revenue_year": [2024],
                        "revenue_month": [10], "revenue": [1],
                        "available_date": [pd.Timestamp("2024-11-11")]})
    ctx = Context(MarketData(prices, uni, revenue=rev, trading_days=days), 3)
    assert ctx.revenue().empty  # backtest: not public on the signal day
    ctx.revenue_asof = pd.Timestamp("2024-11-12")
    assert len(ctx.revenue()) == 1  # live run on the 12th: already public
