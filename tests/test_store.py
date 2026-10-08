import pandas as pd

from twbt.data.datasets import MONTH_REVENUE, PRICE
from twbt.data.store import Manifest, RawStore


def test_upsert_appends_and_dedupes_keeping_latest(tmp_path):
    store = RawStore(tmp_path)
    store.upsert(
        MONTH_REVENUE,
        pd.DataFrame({"stock_id": ["2330"] * 2, "revenue_year": [2024, 2024],
                      "revenue_month": [1, 2], "revenue": [100, 200]}),
        "2330",
    )
    # month 2 revised, month 3 new
    store.upsert(
        MONTH_REVENUE,
        pd.DataFrame({"stock_id": ["2330"] * 2, "revenue_year": [2024, 2024],
                      "revenue_month": [2, 3], "revenue": [210, 300]}),
        "2330",
    )
    df = store.read(MONTH_REVENUE, "2330").sort_values("revenue_month")
    assert df["revenue"].tolist() == [100, 210, 300]


def test_upsert_leaves_no_temp_files(tmp_path):
    store = RawStore(tmp_path)
    store.upsert(PRICE, pd.DataFrame({"stock_id": ["1"], "date": ["2024-01-02"]}), "1")
    files = list((tmp_path / "raw" / PRICE.name).iterdir())
    assert [f.name for f in files] == ["1.parquet"]


def test_failed_write_keeps_old_file(tmp_path, monkeypatch):
    store = RawStore(tmp_path)
    store.upsert(PRICE, pd.DataFrame({"stock_id": ["1"], "date": ["2024-01-02"]}), "1")

    def boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    try:
        store.upsert(PRICE, pd.DataFrame({"stock_id": ["1"], "date": ["2024-01-03"]}), "1")
    except KeyboardInterrupt:
        pass
    monkeypatch.undo()
    assert store.read(PRICE, "1")["date"].tolist() == ["2024-01-02"]
    assert [f.name for f in (tmp_path / "raw" / PRICE.name).iterdir()] == ["1.parquet"]


def test_manifest_persists(tmp_path):
    m = Manifest(tmp_path)
    m.mark("TaiwanStockPrice", "2330", "2024-04-11")
    again = Manifest(tmp_path)
    assert again.get("TaiwanStockPrice", "2330")["last_date"] == "2024-04-11"
