"""A fake FinMind server so tests never touch the network."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from twbt.data.client import FinMindClient, RateLimiter, RetryPolicy


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


class FakeFinMind:
    """Serves ``tables[dataset]`` (list of row dicts) like the real API."""

    def __init__(self, tables: dict[str, list[dict]]) -> None:
        self.tables = tables
        self.requests: list[dict] = []
        self.fail_on: set[tuple[str, str | None]] = set()
        self.script: list[tuple[int, dict]] = []  # responses to return first

    def __call__(self, params: dict) -> tuple[int, dict]:
        self.requests.append(dict(params))
        if self.script:
            return self.script.pop(0)
        ds = params["dataset"]
        if (ds, params.get("data_id")) in self.fail_on:
            return 400, {"status": 400, "msg": "boom"}
        rows = self.tables.get(ds, [])
        if "data_id" in params:
            rows = [r for r in rows if r.get("stock_id") == params["data_id"]]
        if "start_date" in params:
            rows = [r for r in rows if "date" not in r or r["date"] >= params["start_date"]]
        if "end_date" in params:
            rows = [r for r in rows if "date" not in r or r["date"] <= params["end_date"]]
        return 200, {"status": 200, "msg": "success", "data": rows}


def make_client(server: FakeFinMind, clock: FakeClock | None = None, max_calls: int = 10_000):
    clock = clock or FakeClock()
    limiter = RateLimiter(max_calls, clock=clock, sleep=clock.sleep)
    return FinMindClient(server, limiter, RetryPolicy(base_delay=1.0), sleep=clock.sleep)


def price_row(sid: str, d: str, close: float, vol: int = 1000) -> dict:
    return {
        "date": d, "stock_id": sid, "Trading_Volume": vol, "Trading_money": int(vol * close),
        "open": close, "max": close, "min": close, "close": close, "spread": 0.0,
        "Trading_turnover": 1,
    }


@pytest.fixture
def server() -> FakeFinMind:
    tables = {
        "TaiwanStockInfo": [
            {"industry_category": "ETF", "stock_id": "0050", "stock_name": "元大台灣50",
             "type": "twse", "date": "2026-10-07"},
            {"industry_category": "半導體業", "stock_id": "2330", "stock_name": "台積電",
             "type": "twse", "date": "2026-10-07"},
            {"industry_category": "金融保險業", "stock_id": "2881A", "stock_name": "富邦特",
             "type": "twse", "date": "2026-10-07"},
            # moved from 興櫃 to 上櫃: two rows, latest wins
            {"industry_category": "生技醫療業", "stock_id": "1294", "stock_name": "漢田生技",
             "type": "emerging", "date": "2024-09-25"},
            {"industry_category": "生技醫療業", "stock_id": "1294", "stock_name": "漢田生技",
             "type": "tpex", "date": "2026-10-07"},
            # still 興櫃: excluded
            {"industry_category": "食品工業", "stock_id": "1260", "stock_name": "富味鄉",
             "type": "emerging", "date": "2026-10-07"},
        ],
        "TaiwanStockDelisting": [
            {"date": "2015-06-01", "stock_id": "9999", "stock_name": "倒閉公司"},
            {"date": "2005-01-01", "stock_id": "8888", "stock_name": "太早下市"},
        ],
        "TaiwanStockTradingDate": [
            {"date": d} for d in ["2015-05-29", "2015-06-01", "2024-04-10", "2024-04-11"]
        ],
        "TaiwanStockSplitPrice": [],
        "TaiwanStockParValueChange": [],
        "TaiwanStockPrice": [
            price_row("2330", "2024-04-10", 800.0),
            price_row("2330", "2024-04-11", 0.0, vol=0),  # no published price
            price_row("1294", "2024-04-10", 50.0),
            price_row("9999", "2015-05-29", 10.0),
        ],
        "TaiwanStockMonthRevenue": [
            {"date": "2024-04-01", "stock_id": "2330", "country": "Taiwan",
             "revenue": 195_000_000_000, "revenue_month": 3, "revenue_year": 2024,
             "create_time": ""},
        ],
        "TaiwanStockDividendResult": [
            {"date": "2024-04-11", "stock_id": "2330", "before_price": 800.0,
             "after_price": 796.0, "stock_and_cache_dividend": 4.0,
             "stock_or_cache_dividend": "息", "max_price": 875.0, "min_price": 717.0,
             "open_price": 796.0, "reference_price": 796.0},
        ],
        "TaiwanStockCapitalReductionReferencePrice": [],
    }
    return FakeFinMind(tables)


@pytest.fixture
def fixed_now():
    t = {"now": datetime(2026, 10, 8, tzinfo=timezone.utc)}
    return t
