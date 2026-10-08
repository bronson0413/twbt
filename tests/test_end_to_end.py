"""update → build → point-in-time queries, all against the fake server."""

import pandas as pd
import pytest
from conftest import make_client

from twbt.data.clean import build
from twbt.data.query import Data
from twbt.data.store import Manifest, RawStore
from twbt.data.update import Updater


@pytest.fixture
def data(server, tmp_path, fixed_now):
    Updater(make_client(server), RawStore(tmp_path), Manifest(tmp_path),
            start="2010-01-01", end="2026-10-08", now=lambda: fixed_now["now"]).run()
    build(tmp_path)
    return Data(tmp_path)


def test_march_revenue_hidden_on_april_10(data):
    assert data.revenue_asof("2024-04-10").empty


def test_march_revenue_visible_on_april_11(data):
    df = data.revenue_asof("2024-04-11")
    assert df["stock_id"].tolist() == ["2330"]
    assert df.loc[0, "revenue_month"] == 3


def test_universe_on_includes_delisted_before_delisting(data):
    assert "9999" in data.universe_on("2015-05-29")["stock_id"].tolist()
    assert "9999" not in data.universe_on("2015-06-01")["stock_id"].tolist()


def test_universe_on_excludes_not_yet_trading(data):
    # 2330 has no price before 2024-04-10 in the fixture
    assert "2330" not in data.universe_on("2024-01-02")["stock_id"].tolist()


def test_prices_and_actions_roundtrip(data):
    px = data.prices("2024-04-01", "2024-04-30", stocks=["2330"])
    assert len(px) == 2 and pd.isna(px.loc[1, "close"])
    ev = data.corporate_actions("2024-01-01", "2024-12-31")
    assert ev.loc[0, "factor"] == pytest.approx(800 / 796)


def test_trading_days(data):
    days = data.trading_days("2024-01-01", "2024-12-31")
    assert list(days.strftime("%Y-%m-%d")) == ["2024-04-10", "2024-04-11"]


def test_missing_clean_layer_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="twbt build"):
        Data(tmp_path)
