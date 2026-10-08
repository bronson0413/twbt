from datetime import datetime, timedelta, timezone

import pytest
from conftest import make_client

from twbt.data.client import FinMindError
from twbt.data.datasets import MONTH_REVENUE, PRICE
from twbt.data.store import Manifest, RawStore
from twbt.data.update import Updater


def make_updater(server, tmp_path, now, **kw):
    return Updater(
        make_client(server), RawStore(tmp_path), Manifest(tmp_path),
        start="2010-01-01", end="2026-10-08", now=lambda: now["now"], **kw,
    )


def per_stock_requests(server, dataset):
    return [r for r in server.requests if r["dataset"] == dataset and "data_id" in r]


def test_full_run_fetches_every_stock_once(server, tmp_path, fixed_now):
    up = make_updater(server, tmp_path, fixed_now)
    report = up.run()
    assert not report.failed
    # universe: 2330, 1294, 9999 → one call per stock per dataset
    assert sorted(r["data_id"] for r in per_stock_requests(server, PRICE.name)) == [
        "1294", "2330", "9999"
    ]
    assert RawStore(tmp_path).read(PRICE, "2330")["close"].tolist() == [800.0, 0.0]


def test_rerun_soon_after_skips_everything(server, tmp_path, fixed_now):
    make_updater(server, tmp_path, fixed_now).run()
    n = len(server.requests)
    make_updater(server, tmp_path, fixed_now).run()
    assert len(server.requests) == n  # resume: nothing left to do


def test_resume_after_crash_only_fetches_remaining(server, tmp_path, fixed_now):
    server.fail_on = {(PRICE.name, s) for s in ("1294", "2330", "9999")}
    up = make_updater(server, tmp_path, fixed_now, max_consecutive_failures=1)
    with pytest.raises(FinMindError):
        up.run(datasets=[MONTH_REVENUE, PRICE])  # dies on first price call
    done_rev = Manifest(tmp_path).keys(MONTH_REVENUE.name)
    assert done_rev == {"1294"}  # stock-major order: 1294 revenue done before its price failed

    server.fail_on = set()
    server.requests.clear()
    make_updater(server, tmp_path, fixed_now).run(datasets=[MONTH_REVENUE, PRICE])
    rev_ids = sorted(r["data_id"] for r in per_stock_requests(server, MONTH_REVENUE.name))
    assert rev_ids == ["2330", "9999"]  # 1294 revenue not fetched again


def test_incremental_start_dates(server, tmp_path, fixed_now):
    make_updater(server, tmp_path, fixed_now).run()
    server.requests.clear()
    fixed_now["now"] += timedelta(days=30)
    make_updater(server, tmp_path, fixed_now).run()

    price_2330 = [r for r in per_stock_requests(server, PRICE.name) if r["data_id"] == "2330"]
    assert price_2330[0]["start_date"] == "2024-04-12"  # day after last stored date

    rev_2330 = [r for r in per_stock_requests(server, MONTH_REVENUE.name)
                if r["data_id"] == "2330"]
    # revenue re-pulls 120 days back to catch revisions
    assert rev_2330[0]["start_date"] == "2023-12-03"


def test_delisted_stock_fetched_only_once(server, tmp_path, fixed_now):
    make_updater(server, tmp_path, fixed_now).run()
    server.requests.clear()
    fixed_now["now"] = datetime(2027, 1, 1, tzinfo=timezone.utc)
    make_updater(server, tmp_path, fixed_now).run()
    assert "9999" not in {r.get("data_id") for r in server.requests}


def test_single_failure_is_recorded_and_run_continues(server, tmp_path, fixed_now):
    server.fail_on = {(PRICE.name, "1294")}
    report = make_updater(server, tmp_path, fixed_now).run(datasets=[PRICE])
    assert [(d, k) for d, k, _ in report.failed] == [(PRICE.name, "1294")]
    assert Manifest(tmp_path).keys(PRICE.name) == {"2330", "9999"}  # retried next run
