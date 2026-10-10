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


def test_zero_open_with_valid_close_becomes_nan():
    raw = pd.DataFrame([{"date": "2023-10-17", "stock_id": "8272", "Trading_Volume": 10,
                         "Trading_money": 1300, "open": 0, "max": 136, "min": 112.5,
                         "close": 130}])
    px = clean_prices(raw)
    assert np.isnan(px.loc[0, "open"]) and px.loc[0, "close"] == 130


def test_emerging_period_prices_dropped():
    from twbt.data.clean import drop_emerging_period
    px = pd.DataFrame({"stock_id": ["1294"] * 3 + ["2330"],
                       "date": pd.to_datetime(["2024-09-24", "2024-09-25", "2024-09-26",
                                               "2024-09-24"])})
    uni = pd.DataFrame({"stock_id": ["1294", "2330"],
                        "emerging_until": [pd.Timestamp("2024-09-25"), pd.NaT]})
    out = drop_emerging_period(px, uni)
    assert out["date"].dt.strftime("%m-%d").tolist() == ["09-26", "09-24"]


def _px(closes, sid="A", start="2024-01-01"):
    days = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"date": days, "stock_id": sid, "close": closes}), days


def test_unexplained_moves_become_value_neutral_fixes():
    from twbt.data.clean import unexplained_moves
    px, days = _px([10.0] * 6 + [241, 3.43, 238.5, 238.5])  # glitch day in the middle
    px.loc[:5, "close"] = 241.0
    fixes = unexplained_moves(px, pd.DataFrame(columns=["stock_id", "date", "factor"]), days)
    assert fixes["date"].tolist() == [days[7], days[8]]
    # applying both factors restores the original value: 241 → 241 × (241/3.43) × (3.43/238.5)
    assert 3.43 * fixes["factor"].iloc[0] == pytest.approx(241)
    assert (fixes["factor"].prod() * 238.5) == pytest.approx(241)


def test_unexplained_moves_ignore_recorded_events_gaps_and_new_listings():
    from twbt.data.clean import unexplained_moves
    px, days = _px([10, 20, 40, 40, 40, 40, 40, 80.0, 80.0])  # first 5 days free
    ev = pd.DataFrame({"stock_id": ["A"], "date": [days[7]], "factor": [0.5]})
    assert unexplained_moves(px, ev, days).empty            # day 7 has an event
    gap = px.drop(index=7).copy()
    gap.loc[8, "close"] = 100.0
    assert unexplained_moves(gap, ev.iloc[:0], days).empty  # across a missing day


def test_build_includes_data_fixes(server, tmp_path, fixed_now):
    from conftest import make_client, price_row

    from twbt.data.clean import build
    from twbt.data.store import Manifest, RawStore
    from twbt.data.update import Updater
    days = pd.bdate_range("2024-01-01", periods=8)
    server.tables["TaiwanStockPrice"] += [
        price_row("2330", d.date().isoformat(), c)
        for d, c in zip(days, [50, 50, 50, 50, 50, 50, 100, 100], strict=True)]
    server.tables["TaiwanStockTradingDate"] += [{"date": d.date().isoformat()} for d in days]
    Updater(make_client(server), RawStore(tmp_path), Manifest(tmp_path), end="2026-10-08",
            now=lambda: fixed_now["now"]).run()
    build(tmp_path)
    ev = pd.read_parquet(tmp_path / "clean" / "corporate_actions.parquet")
    fix = ev[ev["kind"] == "data_fix"]
    jan = fix[fix["date"] == days[6]]
    assert len(jan) == 1 and jan["factor"].iloc[0] == pytest.approx(0.5)


def test_same_adjustment_in_two_tables_counts_once():
    # real case: 5314 on 2025-03-31, par value 10 → 0.5, in both split and par-value tables
    split = pd.DataFrame([{"date": "2025-03-31", "stock_id": "5314", "type": "分割",
                           "before_price": 1390.0, "after_price": 69.5}])
    par = pd.DataFrame([{"date": "2025-03-31", "stock_id": "5314", "stock_name": "世紀",
                         "before_close": 1390.0, "after_ref_close": 69.5}])
    ev = clean_corporate_actions(pd.DataFrame(), pd.DataFrame(), split, par)
    assert len(ev) == 1 and ev["factor"].iloc[0] == pytest.approx(20)


def test_sequential_events_on_one_day_are_kept():
    # a dividend and a reduction measured from different bases are two real steps
    div = pd.DataFrame([{"date": "2024-05-01", "stock_id": "1", "before_price": 100.0,
                         "reference_price": 95.0, "stock_or_cache_dividend": "息"}])
    red = pd.DataFrame([{"date": "2024-05-01", "stock_id": "1",
                         "ClosingPriceonTheLastTradingDay": 95.0,
                         "PostReductionReferencePrice": 190.0}])
    ev = clean_corporate_actions(div, red, pd.DataFrame(), pd.DataFrame())
    assert len(ev) == 2
