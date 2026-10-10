"""End-to-end research study on synthetic data where the true answer is known."""

import json

import numpy as np
import pandas as pd
import pytest

from twbt.data.query import Data
from twbt.research.study import StudyConfig, active_stats, curve_stats, run_study, save

N = 150


@pytest.fixture(scope="module")
def study_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("study")
    rng = np.random.default_rng(7)
    days = pd.bdate_range("2017-01-02", "2023-12-29")
    ids = [str(1101 + i) for i in range(N)]
    growth = rng.normal(0.05, 0.25, N)            # persistent annual revenue growth
    drift = 0.0002 + growth * 0.002               # faster growers earn more
    rets = drift + rng.normal(0, 0.015, (len(days), N))
    close = 50 * np.exp(np.cumsum(rets, axis=0))
    bench = 50 * np.exp(np.cumsum(0.0004 + rng.normal(0, 0.01, len(days))))
    frames = []
    for j, sid in enumerate(ids + ["0050"]):
        c = close[:, j] if sid != "0050" else bench
        frames.append(pd.DataFrame({"date": days, "stock_id": sid, "open": c, "high": c,
                                    "low": c, "close": c, "volume": 1, "value": 5e7,
                                    "has_quote": True}))
    prices = pd.concat(frames, ignore_index=True)
    months = pd.period_range("2015-01", "2023-11", freq="M")
    rev = []
    for j, sid in enumerate(ids):
        lvl = 1e8 * np.exp(np.cumsum(np.full(len(months), np.log1p(growth[j]) / 12)
                                     + rng.normal(0, 0.05, len(months))))
        for p, v in zip(months, lvl, strict=True):
            rev.append({"stock_id": sid, "period": str(p), "revenue_year": p.year,
                        "revenue_month": p.month, "revenue": int(max(v, 1)),
                        "available_date": p.to_timestamp() + pd.DateOffset(months=1, days=10)})
    clean = root / "clean"
    clean.mkdir()
    pd.DataFrame({"stock_id": ids, "stock_name": ids, "industry": "電子工業", "market": "twse",
                  "delisted_date": pd.NaT}).to_parquet(clean / "universe.parquet")
    pd.DataFrame({"date": days}).to_parquet(clean / "trading_days.parquet")
    prices.to_parquet(clean / "prices.parquet")
    pd.DataFrame(rev).to_parquet(clean / "revenue.parquet")
    pd.DataFrame({"date": pd.to_datetime([]), "stock_id": [], "kind": [], "detail": [],
                  "before_price": [], "reference_price": [], "factor": []}).astype(
        {"stock_id": str, "kind": str, "detail": str, "before_price": float,
         "reference_price": float, "factor": float}).to_parquet(
        clean / "corporate_actions.parquet")
    return root


@pytest.fixture(scope="module")
def study(study_dir):
    cfg = StudyConfig(start="2019-01-01", split="2022-01-01", end="2023-12-29",
                      main_n=15, grid_n=(15,), min_avg_value=0)
    res = run_study(Data(study_dir), cfg, log=lambda m: None)
    numbers = save(res, study_dir / "out")
    return res, numbers, study_dir / "out"


def test_true_signal_is_found(study):
    _, n, _ = study
    rev = n["factor"]["REV"]
    assert rev["is"]["ic_mean"] > 0.05 and rev["oos"]["ic_mean"] > 0.05
    assert rev["full"]["top_minus_bottom_tstat"] > 3
    assert n["selected"] in {"REV", "MOM", "COMBO"}


def test_outputs_written(study):
    _, n, out = study
    for f in ("REPORT.md", "results.json", "quintiles.png", "equity.png", "spread.png"):
        assert (out / f).stat().st_size > 0
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "樣本外" in text and "nan" not in text.lower()
    assert json.loads((out / "results.json").read_text())["selected"] == n["selected"]


def test_selection_uses_in_sample_only(study):
    res, n, _ = study
    split = pd.Timestamp(res.cfg.split)
    is_sharpe = {k: curve_stats(res.portfolios[k].equity[res.portfolios[k].equity.index < split]
                                )["sharpe"] for k in ("REV", "MOM", "COMBO")}
    assert n["selected"] == max(is_sharpe, key=is_sharpe.get)


def test_active_stats():
    idx = pd.bdate_range("2020-01-01", periods=500)
    base = pd.Series(np.exp(np.cumsum(np.full(500, 0.0004))), index=idx)
    same = active_stats(base, base)
    assert same["active_return"] == pytest.approx(0) and np.isnan(same["ir"])


def test_industry_neutral_ranks_within_industry():
    from twbt.research.study import industry_neutral

    class Ctx:
        def industry(self):
            return pd.Series({"A": "x", "B": "x", "C": "y", "D": "y", "E": "z"})

    raw = pd.Series({"A": 1.0, "B": 2.0, "C": 100.0, "D": 200.0, "E": 5.0})
    s = industry_neutral(lambda ctx: raw, min_group=2)(Ctx())
    assert s["B"] == s["D"] == 1.0 and s["A"] == s["C"] == 0.5  # scale-free within industry
    assert "E" not in s.index                                  # industry too small


def test_report_has_robustness_sections(study):
    _, n, out = study
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    for heading in ("產業內部重新排名", "多頭月與空頭月", "## 7. 結論"):
        assert heading in text
    assert n["regime"]["up"]["months"] + n["regime"]["down"]["months"] == \
        n["factor"][n["selected"]]["full"]["periods"]
