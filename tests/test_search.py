"""Strategy search on synthetic data where cheap stocks (low P/B) earn more."""

import json

import numpy as np
import pandas as pd
import pytest

from twbt.data.query import Data
from twbt.research.search import SearchConfig, run_search, save

N = 120


@pytest.fixture(scope="module")
def search_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("search")
    rng = np.random.default_rng(11)
    days = pd.bdate_range("2017-01-02", "2023-12-29")
    ids = [str(1101 + i) for i in range(N)]
    pb = np.exp(rng.normal(0.3, 0.6, N))          # persistent P/B
    pe = pb / rng.uniform(0.05, 0.2, N)           # ROE unrelated to returns
    drift = 0.0006 - 0.0006 * np.log(pb)          # cheaper → higher return
    close = 50 * np.exp(np.cumsum(drift + rng.normal(0, 0.012, (len(days), N)), axis=0))
    bench = 50 * np.exp(np.cumsum(0.0003 + rng.normal(0, 0.01, len(days))))
    frames, val, flows = [], [], []
    for j, sid in enumerate(ids + ["0050"]):
        c = close[:, j] if sid != "0050" else bench
        frames.append(pd.DataFrame({"date": days, "stock_id": sid, "open": c, "high": c,
                                    "low": c, "close": c, "volume": 100_000, "value": 5e7,
                                    "has_quote": True}))
        if sid == "0050":
            continue
        val.append(pd.DataFrame({"date": days, "stock_id": sid, "pe": pe[j], "pb": pb[j],
                                 "dividend_yield": 3.0}))
        flows.append(pd.DataFrame({"date": days, "stock_id": sid,
                                   "foreign_net": rng.normal(0, 1000, len(days)),
                                   "trust_net": 0.0}))
    months = pd.period_range("2015-01", "2023-11", freq="M")
    rev = [{"stock_id": sid, "period": str(p), "revenue_year": p.year, "revenue_month": p.month,
            "revenue": int(1e8 * (1 + 0.1 * rng.normal())),
            "available_date": p.to_timestamp() + pd.DateOffset(months=1, days=10)}
           for sid in ids for p in months]
    clean = root / "clean"
    clean.mkdir()
    pd.DataFrame({"stock_id": ids, "stock_name": ids, "industry": "電子工業", "market": "twse",
                  "delisted_date": pd.NaT}).to_parquet(clean / "universe.parquet")
    pd.DataFrame({"date": days}).to_parquet(clean / "trading_days.parquet")
    pd.concat(frames, ignore_index=True).to_parquet(clean / "prices.parquet")
    pd.DataFrame(rev).to_parquet(clean / "revenue.parquet")
    pd.DataFrame({"date": pd.to_datetime([]), "stock_id": [], "kind": [], "detail": [],
                  "before_price": [], "reference_price": [], "factor": []}).astype(
        {"stock_id": str, "kind": str, "detail": str, "before_price": float,
         "reference_price": float, "factor": float}).to_parquet(
        clean / "corporate_actions.parquet")
    pd.concat(val, ignore_index=True).to_parquet(clean / "valuation.parquet")
    pd.concat(flows, ignore_index=True).to_parquet(clean / "flows.parquet")
    return root


@pytest.fixture(scope="module")
def search(search_dir):
    cfg = SearchConfig(start="2019-01-01", split="2022-01-01", end="2023-12-29", n=15,
                       min_avg_value=0)
    res = run_search(Data(search_dir), cfg, log=lambda m: None)
    return res, save(res, search_dir / "out"), search_dir / "out"


def test_value_signal_is_found_and_selected(search):
    _, n, _ = search
    v = n["factor"]["VALUE"]
    assert v["is"]["ic_mean"] > 0.05 and v["oos"]["ic_mean"] > 0.05
    assert n["selected"] in {"VALUE", "QV"}
    assert abs(n["factor"]["INST"]["full"]["ic_mean"]) < 0.05  # noise flows: no signal


def test_report_written_without_missing_numbers(search):
    _, n, out = search
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert n["selected"] in text and "nan" not in text.lower()
    assert (out / "equity.png").stat().st_size > 0
    assert json.loads((out / "results.json").read_text())["selected"] == n["selected"]
    assert n["coverage"]["valuation"] == pytest.approx(1.0)


def test_search_needs_new_tables(tmp_path, search_dir):
    import shutil

    shutil.copytree(search_dir / "clean", tmp_path / "clean")
    (tmp_path / "clean" / "flows.parquet").unlink()
    with pytest.raises(RuntimeError):
        run_search(Data(tmp_path), SearchConfig(), log=lambda m: None)


def test_live_orders_and_monthly_diff(search, search_dir, tmp_path):
    from twbt.live import diff, run

    _, n, out = search
    first = run(search_dir, out / "results.json", tmp_path / "live")
    t = first["target"]
    assert t["strategy"] == n["selected"] and len(t["holdings"]) == n["n"]
    assert all(h["shares"] > 0 for h in t["holdings"])
    assert "第一次建立部位" in first["markdown"]
    # pretend last month held something else: one stock sold, all of this month's bought
    prev = dict(t, date="2023-11-13", holdings=[dict(t["holdings"][0]),
                                                 dict(t["holdings"][0], stock_id="9999")])
    p = tmp_path / "prev.json"
    p.write_text(__import__("json").dumps(prev), encoding="utf-8")
    second = run(search_dir, out / "results.json", tmp_path / "live2", previous=p)
    d = diff(second["target"], prev)
    assert [h["stock_id"] for h in d["sell"]] == ["9999"]
    assert len(d["keep"]) == 1 and len(d["buy"]) == n["n"] - 1
    assert "賣出" in second["markdown"] and "續抱" in second["markdown"]
    # rerun in the same month still diffs against last month, not against itself
    third = run(search_dir, out / "results.json", tmp_path / "live2",
                previous=tmp_path / "live2" / "latest.json")
    assert "9999" in third["markdown"]


def test_track_reports_holdings_since_signal(search, search_dir, tmp_path):
    import json as _json

    from twbt.live import run, track

    _, n, out = search
    run(search_dir, out / "results.json", tmp_path / "live")
    latest = _json.loads((tmp_path / "live" / "latest.json").read_text(encoding="utf-8"))
    latest["date"] = "2023-10-02"  # pretend the list was made two months before the data end
    (tmp_path / "old.json").write_text(_json.dumps(latest), encoding="utf-8")
    text = track(search_dir, tmp_path / "old.json", tmp_path / "live")
    assert "持股等權重報酬" in text and "nan" not in text.lower()
    assert all(h["stock_id"] in text for h in latest["holdings"])


def test_cli_default_holdings_come_from_config(monkeypatch):
    """The CLI once defaulted to 30 holdings and silently overrode the config's 10."""
    import twbt.research.search as search_mod
    from twbt import cli

    seen = {}

    def fake_run(data, cfg, log=None, only=None):
        seen["n"] = cfg.n
        raise RuntimeError("stop")

    monkeypatch.setattr(search_mod, "run_search", fake_run)
    monkeypatch.setattr(cli, "Data", lambda root: None)
    assert cli.main(["search"]) == 1
    assert seen["n"] == search_mod.SearchConfig().n == 10


def test_inspect_runs_on_synthetic_data(search_dir):
    from twbt.research.inspect import inspect

    text = inspect(Data(search_dir), ["VALUE"], start="2019-01-01", end="2020-12-31")
    assert "VALUE top-30" in text and "averages per year" in text


def test_large_cap_universe_and_cagr_selection(search_dir):
    from twbt.engine import MarketData
    from twbt.engine.context import Context
    from twbt.strategies import eligible

    data = Data(search_dir)
    m = MarketData.from_data(data, "2021-01-04", "2021-06-30")
    ctx = Context(m, len(m.dates) - 1)
    big = eligible(ctx, 0, top_value=20)
    assert len(big) == 20 and set(big) <= set(eligible(ctx, 0))

    cfg = SearchConfig(start="2019-01-01", split="2022-01-01", end="2023-12-29", n=5,
                       min_avg_value=0, top_value=60, select="cagr", grid_n=(5,))
    res = run_search(data, cfg, log=lambda m: None, only=["VALUE", "LOWVOL", "INST"])
    from twbt.research.study import curve_stats, split_curve
    cagr = {k: curve_stats(split_curve(res.portfolios[k].equity, pd.Timestamp("2022-01-01"))[0])
            ["cagr"] for k in ("VALUE", "LOWVOL", "INST")}
    assert res.selected == max(cagr, key=cagr.get)
    n = save(res, search_dir / "out_large")
    assert n["top_value"] == 60 and n["select"] == "cagr"
    text = (search_dir / "out_large" / "REPORT.md").read_text(encoding="utf-8")
    assert "前 60 大" in text and "年化報酬最高" in text


def test_plan_check_and_live_with_buffer(search_dir, tmp_path):
    import json as _json

    from twbt.live import run
    from twbt.research.plan import Plan, check, save

    plan = Plan(start="2019-01-01", split="2022-01-01", top_value=60, n=5, keep_rank=10,
                min_avg_value=0)
    out = check(Data(search_dir), plan, log=lambda m: None)
    assert out["runs"]["PLAN"]["turnover"] <= out["runs"]["NO_BUFFER"]["turnover"] + 1e-9
    save(out, tmp_path / "plan")
    text = (tmp_path / "plan" / "REPORT.md").read_text(encoding="utf-8")
    assert "實盤設計" in text and "nan" not in text.lower()
    cfg = tmp_path / "plan" / "config.json"
    first = run(search_dir, cfg, tmp_path / "live")
    assert len(first["target"]["holdings"]) == 5
    # last month held one of today's top-10 but not top-5: the buffer keeps it
    from twbt.engine import MarketData
    from twbt.engine.context import Context
    from twbt.research.search import candidates
    from twbt.strategies import eligible
    data = Data(search_dir)
    m = MarketData.from_data(data, "2023-12-29", "2023-12-29", benchmarks=())
    ctx = Context(m, len(m.dates) - 1)
    s = candidates()["COMBO"](ctx)
    s = s[s.index.isin(eligible(ctx, 0, top_value=60))].sort_values(ascending=False)
    sixth = s.index[6]
    prev = dict(first["target"], date="2023-11-13",
                holdings=[{"stock_id": sixth, "name": "", "industry": "", "close": 1.0,
                           "shares": 1, "weight": 0.2, "score": 0}])
    p = tmp_path / "prev.json"
    p.write_text(_json.dumps(prev), encoding="utf-8")
    second = run(search_dir, cfg, tmp_path / "live2", previous=p)
    assert sixth in [h["stock_id"] for h in second["target"]["holdings"]]


def test_list_from_another_strategy_is_not_treated_as_holdings(search, search_dir, tmp_path):
    import json as _json

    from twbt.live import run

    _, n, out = search
    other = {"date": "2023-11-13", "strategy": "LOWVOL", "n": n["n"], "capital": 1,
             "holdings": [{"stock_id": "9999", "name": "", "industry": "", "close": 1.0,
                           "shares": 1, "weight": 1.0, "score": 0}]}
    p = tmp_path / "other.json"
    p.write_text(_json.dumps(other), encoding="utf-8")
    res = _json.loads((out / "results.json").read_text(encoding="utf-8"))
    if res["selected"] == "LOWVOL":
        res["selected"] = "VALUE"
    cfg = tmp_path / "cfg.json"
    cfg.write_text(_json.dumps(res), encoding="utf-8")
    r = run(search_dir, cfg, tmp_path / "live", previous=p)
    assert "第一次建立部位" in r["markdown"] and "9999" not in r["markdown"]
