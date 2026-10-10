"""CLI backtest / signal on a small synthetic clean layer."""

import numpy as np
import pandas as pd
import pytest

from twbt.cli import main

IDS = ["1101", "2330", "2454", "3008", "6505"]
GROWTH = {"1101": -0.1, "2330": 0.4, "2454": 0.2, "3008": 0.0, "6505": 0.05}


@pytest.fixture
def clean_dir(tmp_path):
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2021-01-01", "2024-12-31")
    rows = []
    for sid in IDS + ["0050"]:
        drift = 0.0008 if sid == "2330" else 0.0002
        px = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, len(days))))
        rows.append(pd.DataFrame({"date": days, "stock_id": sid, "open": px, "high": px * 1.01,
                                  "low": px * 0.99, "close": px, "volume": 1_000_000,
                                  "value": 5e8, "has_quote": True}))
    prices = pd.concat(rows, ignore_index=True)
    months = pd.period_range("2020-01", "2024-11", freq="M")
    rev = []
    for sid in IDS:
        for i, p in enumerate(months):
            rev.append({"stock_id": sid, "period": str(p), "revenue_year": p.year,
                        "revenue_month": p.month,
                        "revenue": int(1e8 * (1 + GROWTH[sid]) ** (i / 12)),
                        "available_date": p.to_timestamp() + pd.DateOffset(months=1, days=10)})
    clean = tmp_path / "clean"
    clean.mkdir()
    pd.DataFrame({"stock_id": IDS, "stock_name": IDS, "industry": "x", "market": "twse",
                  "delisted_date": pd.NaT}).to_parquet(clean / "universe.parquet")
    pd.DataFrame({"date": days}).to_parquet(clean / "trading_days.parquet")
    prices.to_parquet(clean / "prices.parquet")
    pd.DataFrame(rev).to_parquet(clean / "revenue.parquet")
    pd.DataFrame(columns=["date", "stock_id", "kind", "detail", "before_price",
                          "reference_price", "factor"]).astype(
        {"date": "datetime64[ns]", "stock_id": str, "kind": str, "detail": str,
         "before_price": float, "reference_price": float, "factor": float}
    ).to_parquet(clean / "corporate_actions.parquet")
    return tmp_path


def test_backtest_command(clean_dir, capsys):
    assert main(["--data-dir", str(clean_dir), "backtest", "--from", "2022-06-01",
                 "--end", "2024-12-31", "--top-n", "2", "--discount", "0.28"]) == 0
    out = capsys.readouterr().out
    assert "年化報酬" in out and "0050" in out
    eq = pd.read_csv(clean_dir / "results" / "equity.csv", index_col=0)
    assert list(eq.columns) == ["strategy", "benchmark"]
    trades = pd.read_csv(clean_dir / "results" / "trades.csv", dtype={"stock_id": str})
    # fastest revenue growers are the only stocks ever bought
    assert set(trades.loc[trades["side"] == "buy", "stock_id"]) == {"2330", "2454"}


def test_signal_command(clean_dir, capsys):
    assert main(["--data-dir", str(clean_dir), "signal", "--end", "2024-12-31",
                 "--top-n", "2", "--capital", "250000"]) == 0
    out = capsys.readouterr().out
    assert "2330" in out and "2454" in out and "1101" not in out


def test_backtest_without_benchmark_explains(clean_dir, capsys):
    px = pd.read_parquet(clean_dir / "clean" / "prices.parquet")
    px[px["stock_id"] != "0050"].to_parquet(clean_dir / "clean" / "prices.parquet")
    assert main(["--data-dir", str(clean_dir), "backtest", "--from", "2022-06-01",
                 "--end", "2024-12-31"]) == 1
    assert "0050" in capsys.readouterr().err


def test_factor_command(clean_dir, capsys):
    assert main(["--data-dir", str(clean_dir), "factor", "--from", "2022-01-01",
                 "--end", "2024-12-31", "--groups", "2", "--min-avg-value", "0"]) in (0, 1)
    out = capsys.readouterr()
    # 5 synthetic stocks are too few to split: must explain, not crash
    assert "原版" in out.out or "沒有足夠" in out.err
