import pandas as pd

from twbt.data.universe import build_universe


def test_universe_filters(server):
    info = pd.DataFrame(server.tables["TaiwanStockInfo"])
    delisting = pd.DataFrame(server.tables["TaiwanStockDelisting"])
    uni = build_universe(info, delisting, since="2010-01-01")
    ids = uni["stock_id"].tolist()

    assert "2330" in ids
    assert "0050" not in ids        # ETF
    assert "2881A" not in ids       # preferred share
    assert "1260" not in ids        # still 興櫃
    assert "1294" in ids            # moved to 上櫃: latest row wins
    assert uni.set_index("stock_id").loc["1294", "market"] == "tpex"
    assert "9999" in ids            # delisted after `since`: kept (survivorship)
    assert "8888" not in ids        # delisted before `since`: not needed


def test_reused_code_counts_as_current():
    info = pd.DataFrame([{"industry_category": "x", "stock_id": "1234", "stock_name": "新",
                          "type": "twse", "date": "2026-01-01"}])
    delisting = pd.DataFrame([{"date": "2015-01-01", "stock_id": "1234", "stock_name": "舊"}])
    uni = build_universe(info, delisting, since="2010-01-01")
    assert len(uni) == 1
    assert pd.isna(uni.loc[0, "delisted_date"])


def test_emerging_until_and_innovation_board(server):
    info = pd.DataFrame(server.tables["TaiwanStockInfo"] + [
        {"industry_category": "電子工業", "stock_id": "7740", "stock_name": "熙特爾-創",
         "type": "twse", "date": "2026-10-07"},
    ])
    uni = build_universe(info, pd.DataFrame(), since="2010-01-01").set_index("stock_id")
    assert uni.loc["1294", "emerging_until"] == pd.Timestamp("2024-09-25")
    assert pd.isna(uni.loc["2330", "emerging_until"])
    assert "7740" not in uni.index  # 創新板: qualified investors only
