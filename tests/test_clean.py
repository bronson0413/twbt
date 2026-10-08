import numpy as np
import pandas as pd
import pytest

from twbt.data.clean import adjusted_close, clean_corporate_actions, clean_prices, clean_revenue


def rev(year, month, create_time=""):
    return {"stock_id": "2330", "revenue_year": year, "revenue_month": month,
            "revenue": 1, "date": "x", "country": "Taiwan", "create_time": create_time}


def test_revenue_available_from_11th_of_next_month():
    df = clean_revenue(pd.DataFrame([rev(2024, 3), rev(2024, 12)]))
    assert df["available_date"].tolist() == [pd.Timestamp("2024-04-11"),
                                             pd.Timestamp("2025-01-11")]


def test_late_filer_uses_create_time_plus_one_day():
    df = clean_revenue(pd.DataFrame([rev(2026, 6, create_time="2026-07-20")]))
    assert df.loc[0, "available_date"] == pd.Timestamp("2026-07-21")


def test_early_filer_still_waits_for_deadline():
    df = clean_revenue(pd.DataFrame([rev(2026, 6, create_time="2026-07-03")]))
    assert df.loc[0, "available_date"] == pd.Timestamp("2026-07-11")


def test_create_time_backfill_placeholder_is_ignored():
    # every row FinMind wrote on 2026-04-21 carries that date, meaningless
    df = clean_revenue(pd.DataFrame([rev(2025, 1, create_time="2026-04-21")]))
    assert df.loc[0, "available_date"] == pd.Timestamp("2025-02-11")


def test_deadline_on_holiday_is_extended():
    # 2026-10-10 is National Day (Saturday); companies may publish on Mon 10-12
    tdays = pd.DatetimeIndex(["2026-10-08", "2026-10-09", "2026-10-12", "2026-10-13"])
    df = clean_revenue(pd.DataFrame([rev(2026, 9)]), tdays)
    assert df.loc[0, "available_date"] == pd.Timestamp("2026-10-13")


def test_available_date_lands_on_trading_day():
    # 2024-03-10 is a Sunday → deadline Mon 03-11 → usable Tue 03-12
    tdays = pd.DatetimeIndex(["2024-03-08", "2024-03-11", "2024-03-12"])
    df = clean_revenue(pd.DataFrame([rev(2024, 2)]), tdays)
    assert df.loc[0, "available_date"] == pd.Timestamp("2024-03-12")


def test_real_tsmc_rows_from_finmind():
    # Actual API rows (2026-10-08 fetch). 05-10 is a Sunday → deadline Mon 05-11.
    # June create_time is Mon 07-13 although
    # the deadline was Fri 07-10, so we wait for Tue 07-14 (conservative).
    tdays = pd.bdate_range("2026-05-01", "2026-07-31")
    df = clean_revenue(pd.DataFrame([rev(2026, 4, "2026-05-08"), rev(2026, 6, "2026-07-13")]),
                       tdays)
    assert df["available_date"].tolist() == [pd.Timestamp("2026-05-12"),
                                             pd.Timestamp("2026-07-14")]


def test_zero_price_days_become_nan():
    raw = pd.DataFrame([
        {"date": "2025-07-29", "stock_id": "2317", "Trading_Volume": 10, "Trading_money": 1715,
         "open": 171.5, "max": 171.5, "min": 171.5, "close": 171.5},
        {"date": "2025-07-30", "stock_id": "2317", "Trading_Volume": 0, "Trading_money": 0,
         "open": 0, "max": 0, "min": 0, "close": 0},
    ])
    px = clean_prices(raw)
    assert px["has_quote"].tolist() == [True, False]
    assert np.isnan(px.loc[1, "close"])


def _div(date, before, ref, sid="2330"):
    return {"date": date, "stock_id": sid, "before_price": before, "reference_price": ref,
            "stock_or_cache_dividend": "息"}


def test_corporate_action_factors():
    ev = clean_corporate_actions(
        pd.DataFrame([_div("2024-04-11", 800, 796)]),
        pd.DataFrame([{"date": "2024-05-01", "stock_id": "1111",
                       "ClosingPriceonTheLastTradingDay": 20.0,
                       "PostReductionReferencePrice": 40.0,
                       "ReasonforCapitalReduction": "彌補虧損"}]),
        pd.DataFrame(),
        pd.DataFrame([{"date": "2024-06-01", "stock_id": "2222", "stock_name": "x",
                       "before_close": 750.0, "after_ref_close": 375.0}]),
    )
    f = ev.set_index("stock_id")["factor"]
    assert f["2330"] == pytest.approx(800 / 796)
    assert f["1111"] == pytest.approx(0.5)   # reduction: price goes up, shares go down
    assert f["2222"] == pytest.approx(2.0)   # par value 10 → 5: price halves


def test_invalid_event_prices_dropped():
    ev = clean_corporate_actions(
        pd.DataFrame([_div("2024-04-11", 0, 796)]), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    )
    assert ev.empty


def test_adjusted_close_removes_ex_date_drop():
    prices = pd.DataFrame({
        "stock_id": ["A"] * 4,
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
        "close": [100.0, 100.0, 95.0, 95.0],  # 5-dollar cash dividend on 01-04
    })
    ev = clean_corporate_actions(
        pd.DataFrame([_div("2024-01-04", 100, 95, sid="A")]),
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
    )
    adj = adjusted_close(prices, ev)
    assert adj.tolist() == pytest.approx([95.0, 95.0, 95.0, 95.0])  # no fake -5% return


def test_adjusted_close_compounds_multiple_events():
    prices = pd.DataFrame({
        "stock_id": ["A"] * 3,
        "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        "close": [100.0, 50.0, 25.0],
    })
    ev = clean_corporate_actions(
        pd.DataFrame([_div("2024-02-01", 100, 50, "A"), _div("2024-03-01", 50, 25, "A")]),
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
    )
    assert adjusted_close(prices, ev).tolist() == pytest.approx([25.0, 25.0, 25.0])
