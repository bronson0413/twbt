import math

import numpy as np
import pandas as pd
import pytest

from twbt.engine import BacktestConfig, CostModel, MarketData, monthly_on_or_after, run_backtest
from twbt.engine.costs import ZERO_COST
from twbt.engine.metrics import max_drawdown

DAYS = pd.bdate_range("2024-01-01", periods=6)


def market(bars: dict[str, list], actions=None, universe=None, revenue=None, value=1e9):
    """bars: sid -> list of close, or (open, high, low, close), or None (no quote)."""
    rows = []
    for sid, series in bars.items():
        for d, b in zip(DAYS, series, strict=False):
            if b is None:
                o = h = lo = c = np.nan
            elif isinstance(b, tuple):
                o, h, lo, c = b
            else:
                o = h = lo = c = b
            rows.append({"date": d, "stock_id": sid, "open": o, "high": h, "low": lo,
                         "close": c, "value": value})
    uni = universe if universe is not None else pd.DataFrame(
        {"stock_id": list(bars), "delisted_date": pd.NaT})
    return MarketData(pd.DataFrame(rows), uni, revenue=revenue, actions=actions,
                      trading_days=DAYS)


def hold(weights):
    return lambda ctx: dict(weights)


def bt(m, strategy, rebalance=(DAYS[0],), **cfg):
    cfg.setdefault("costs", ZERO_COST)
    return run_backtest(strategy, m, DAYS[0], DAYS[-1], rebalance, BacktestConfig(**cfg))


def action(day, sid, kind, before, ref, detail=""):
    return {"date": DAYS[day], "stock_id": sid, "kind": kind, "detail": detail,
            "before_price": before, "reference_price": ref, "factor": before / ref}


# -- costs --------------------------------------------------------------------


def test_commission_floor_minimum_and_discount():
    c = CostModel()
    assert c.commission(100_000) == 142          # floor(142.5)
    assert c.commission(1_000) == 20             # minimum
    assert CostModel(commission_discount=0.28).commission(100_000) == 39  # floor(39.9)


def test_tax_stock_vs_etf():
    c = CostModel()
    assert c.tax(100_000, "2330") == 300
    assert c.tax(100_000, "0050") == 100


# -- execution timing ----------------------------------------------------------


def test_signal_fills_at_next_open_not_same_close():
    m = market({"A": [10, (15, 16, 14, 16), 16, 16, 16, 16]})
    r = bt(m, hold({"A": 1.0}), capital=1500)
    buy = r.trades.iloc[0]
    assert buy["date"] == DAYS[1] and buy["price"] == 15
    assert buy["shares"] == 100


def test_equity_tracks_price():
    m = market({"A": [10, 10, 12, 12, 9, 9]})
    r = bt(m, hold({"A": 1.0}), capital=1000)
    assert r.equity.tolist() == pytest.approx([1000, 1000, 1200, 1200, 900, 900])


def test_context_cannot_see_the_future():
    rev = pd.DataFrame({"stock_id": ["A", "A"], "revenue_year": [2023, 2023],
                        "revenue_month": [11, 12], "revenue": [1, 2],
                        "period": ["2023-11", "2023-12"],
                        "available_date": [DAYS[0], DAYS[3]]})
    m = market({"A": [10] * 6}, revenue=rev)
    seen = []

    def spy(ctx):
        months = ctx.revenue()["revenue_month"].tolist()
        seen.append((ctx.date, ctx.close(1000).index.max(), months))
        return {}

    bt(m, spy, rebalance=DAYS)
    for date, max_price_date, _ in seen:
        assert max_price_date == date
    assert seen[2][2] == [11]       # day 2: December not public yet
    assert seen[3][2] == [11, 12]   # day 3: now it is


# -- corporate actions -----------------------------------------------------------


def test_cash_dividend_keeps_equity():
    acts = pd.DataFrame([action(2, "A", "dividend", 100, 95, "息")])
    m = market({"A": [100, 100, 95, 95, 95, 95]}, actions=acts)
    r = bt(m, hold({"A": 1.0}), capital=10_000)
    assert r.equity.tolist() == pytest.approx([10_000] * 6)
    assert r.cash.iloc[2] == pytest.approx(500)  # 100 shares × 5


def test_split_multiplies_shares():
    acts = pd.DataFrame([action(2, "0050", "split", 188.65, 47.16)])
    m = market({"0050": [188.65, 188.65, 47.16, 47.16, 47.16, 47.16]}, actions=acts)
    r = bt(m, hold({"0050": 1.0}), capital=18_865)
    assert r.equity.iloc[2] == pytest.approx(18_865, rel=1e-3)
    row = r.trades[r.trades["side"] == "split"].iloc[0]
    assert row["shares"] == 300  # 100 → 400


def test_stock_dividend_pays_fraction_in_cash():
    # 105 shares × 1.1 = 115.5 → 115 shares + 0.5 share in cash
    acts = pd.DataFrame([action(2, "A", "dividend", 110, 100, "權")])
    m = market({"A": [110, 110, 100, 100, 100, 100]}, actions=acts)
    r = bt(m, hold({"A": 1.0}), capital=11_550)
    assert r.trades.iloc[0]["shares"] == 105
    assert r.cash.iloc[2] == pytest.approx(0.5 * 100)
    assert r.equity.iloc[2] == pytest.approx(11_550)


# -- market rules --------------------------------------------------------------


def test_board_lot_too_expensive_stays_in_cash():
    m = market({"A": [100] * 6})
    r = bt(m, hold({"A": 1.0}), capital=50_000, lot_size=1000)
    assert r.trades.empty and r.cash.iloc[-1] == 50_000


def test_odd_lots_allow_small_capital():
    m = market({"A": [100] * 6})
    r = bt(m, hold({"A": 1.0}), capital=50_000, lot_size=1)
    assert r.trades.iloc[0]["shares"] == 500


def test_cannot_buy_limit_up_locked_open():
    m = market({"A": [10, (11, 11, 11, 11), 11, 11, 11, 11]})
    r = bt(m, hold({"A": 1.0}), capital=1000)
    assert r.trades.empty


def test_old_7pct_limit_before_2015():
    days = pd.bdate_range("2014-01-01", periods=3)
    rows = [{"date": d, "stock_id": "A", "open": o, "high": o, "low": o, "close": o, "value": 1e9}
            for d, o in zip(days, [10.0, 10.7, 10.7], strict=True)]
    m = MarketData(pd.DataFrame(rows), pd.DataFrame({"stock_id": ["A"]}), trading_days=days)
    r = run_backtest(hold({"A": 1.0}), m, days[0], days[-1], [days[0]],
                     BacktestConfig(capital=1000, costs=ZERO_COST))
    assert r.trades.empty  # +7% locked was limit-up back then


def test_no_quote_no_trade():
    m = market({"A": [10, None, 10, 10, 10, 10]})
    r = bt(m, hold({"A": 1.0}), capital=1000)
    assert r.trades.empty


def test_delisting_converts_to_cash():
    uni = pd.DataFrame({"stock_id": ["A"], "delisted_date": [DAYS[3]]})
    m = market({"A": [10, 10, 12, None, None, None]}, universe=uni)
    r = bt(m, hold({"A": 1.0}), capital=1000)
    assert r.trades.iloc[-1]["side"] == "delist"
    assert r.equity.iloc[-1] == pytest.approx(1200)
    r0 = bt(m, hold({"A": 1.0}), capital=1000, delist_recovery=0.0)
    assert r0.equity.iloc[-1] == 0


def test_switching_sells_before_buying():
    m = market({"A": [10] * 6, "B": [20] * 6})
    r = bt(m, lambda ctx: {"A": 1.0} if ctx.date == DAYS[0] else {"B": 1.0},
           rebalance=DAYS[:2], capital=1000)
    assert r.trades["side"].tolist() == ["buy", "sell", "buy"]
    assert r.trades.iloc[-1]["shares"] == 50


def test_costs_never_overdraw_cash():
    m = market({"A": [33.3] * 6, "B": [77.7] * 6, "C": [12.1] * 6})
    r = bt(m, hold({"A": 0.34, "B": 0.33, "C": 0.33}), rebalance=DAYS,
           capital=10_000, costs=CostModel())
    assert (r.cash >= 0).all()
    s = r.summary()
    assert s["total_costs"] > 0 and r.equity.iloc[-1] < 10_000


def test_weight_validation():
    m = market({"A": [10] * 6})
    with pytest.raises(ValueError, match="leverage"):
        bt(m, hold({"A": 1.5}))
    with pytest.raises(ValueError, match="negative"):
        bt(m, hold({"A": -0.1}))
    with pytest.raises(ValueError, match="unknown"):
        bt(m, hold({"ZZZ": 0.1}))


# -- metrics & schedule ---------------------------------------------------------


def test_max_drawdown():
    assert max_drawdown(pd.Series([100, 120, 90, 130])) == pytest.approx(-0.25)


def test_summary_cagr():
    idx = pd.to_datetime(["2020-01-01", "2022-01-01"])
    from twbt.engine.metrics import summary
    out = summary(pd.Series([100.0, 121.0], index=idx), pd.DataFrame())
    assert out["cagr"] == pytest.approx(0.1, rel=1e-3)
    assert math.isclose(out["total_return"], 0.21)


def test_monthly_schedule():
    days = pd.bdate_range("2026-09-01", "2026-11-30")
    got = monthly_on_or_after(days, 11)
    assert [d.date().isoformat() for d in got] == ["2026-09-11", "2026-10-12", "2026-11-11"]


def test_zero_open_is_not_tradable():
    m = market({"A": [(0.0, 10, 10, 10), (0.0, 10, 10, 10), 10, 10, 10, 10]})
    r = bt(m, hold({"A": 1.0}), rebalance=DAYS[:2], capital=1000)  # day-1 fill fails
    assert r.trades.iloc[0]["date"] == DAYS[2]


def test_attribution_finds_the_mover():
    m = market({"A": [10, 10, 10, 15, 15, 15], "B": [10] * 6})
    r = bt(m, hold({"A": 0.5, "B": 0.5}), capital=2000)
    att = r.attribution(DAYS[3])
    assert att.loc[0, "stock_id"] == "A" and att.loc[0, "pnl"] == pytest.approx(100 * 5)
    assert att.loc[0, "pct_of_equity"] == pytest.approx(0.25)


def _one_stock_market(path):
    import numpy as np
    import pandas as pd

    from twbt.engine import MarketData
    days = pd.bdate_range("2024-01-01", periods=len(path))
    prices = pd.DataFrame({"date": days, "stock_id": "A", "open": path, "high": path,
                           "low": path, "close": path, "volume": 1000, "value": 1e8})
    uni = pd.DataFrame({"stock_id": ["A"], "industry": ["x"], "delisted_date": [pd.NaT]})
    return days, MarketData(prices, uni, trading_days=days), np


def test_stop_loss_sells_next_open_and_stays_out_until_rebalance():
    from twbt.engine import BacktestConfig, run_backtest
    from twbt.engine.costs import ZERO_COST

    path = [100, 100, 95, 89, 85, 80, 120, 130]
    days, m, _ = _one_stock_market(path)
    cfg = BacktestConfig(capital=10_000, costs=ZERO_COST, stop_loss=0.10)
    r = run_backtest(lambda ctx: {"A": 1.0}, m, days[0], days[-1], [days[0]], cfg)
    stops = r.trades[r.trades["side"] == "stop"]
    # bought at day-1 open 100; day-3 close 89 is -11% → sold at day-4 open 85
    assert len(stops) == 1 and stops.iloc[0]["date"] == days[4]
    assert r.equity.iloc[-1] == r.equity.iloc[4]  # in cash for the rally afterwards


def test_same_close_fill_trades_on_the_signal_bar():
    from twbt.engine import BacktestConfig, run_backtest
    from twbt.engine.costs import ZERO_COST

    path = [100, 200, 200]
    days, m, _ = _one_stock_market(path)
    common = dict(capital=10_000, costs=ZERO_COST)
    nxt = run_backtest(lambda ctx: {"A": 1.0}, m, days[0], days[-1], [days[0]],
                       BacktestConfig(**common))
    same = run_backtest(lambda ctx: {"A": 1.0}, m, days[0], days[-1], [days[0]],
                        BacktestConfig(**common, fill="same_close"))
    assert same.equity.iloc[-1] == 20_000   # bought at 100 on the signal close
    assert nxt.equity.iloc[-1] == 10_000    # realistic: bought at the next open, 200
