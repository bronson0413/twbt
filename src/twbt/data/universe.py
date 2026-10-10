"""Stock universe: listed + OTC common stocks, including delisted ones.

Survivorship bias comes from testing only on companies that still exist
today. ``TaiwanStockInfo`` lists current securities; ``TaiwanStockDelisting``
lists the ones that are gone. We take the union and keep common stocks only.
"""

from __future__ import annotations

import re

import pandas as pd

# Common stocks are four digits not starting with 0. This drops ETFs (00xx),
# preferred shares (e.g. 2881A), warrants (6 digits), ETNs and bonds.
COMMON_STOCK_ID = re.compile(r"^[1-9]\d{3}$")

EXCLUDED_INDUSTRIES = {
    "ETF",
    "ETN",
    "存託憑證",
    "受益證券",
    "大盤",
    "Index",
    "所有證券",
}
LISTED_MARKETS = {"twse", "tpex"}


def build_universe(info: pd.DataFrame, delisting: pd.DataFrame, since: str) -> pd.DataFrame:
    """One row per stock: stock_id, stock_name, industry, market, delisted_date, emerging_until.

    ``since``: delisted stocks that left before this date can't appear in the
    backtest, so we don't spend API calls on them.

    ``emerging_until``: for stocks that moved from 興櫃 to 上市/上櫃, the date
    their 興櫃 row stops. Prices up to then are 興櫃 quotes (negotiated, often
    not tradable for retail) and must not be used as if the stock were listed.
    """
    cols = ["stock_id", "stock_name", "industry", "market", "delisted_date", "emerging_until"]

    current = pd.DataFrame(columns=cols)
    if not info.empty:
        df = info.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["stock_id"] = df["stock_id"].astype(str)
        emerging_until = df[df["type"] == "emerging"].groupby("stock_id")["date"].max()
        # A stock that moved from 興櫃 to 上市/上櫃 keeps both rows; the row with
        # the latest date is its current market (per FinMind docs).
        df = df.sort_values("date").drop_duplicates("stock_id", keep="last")
        df = df[df["type"].isin(LISTED_MARKETS)]
        df = df[~df["industry_category"].isin(EXCLUDED_INDUSTRIES)]
        df = df[df["stock_id"].str.match(COMMON_STOCK_ID)]
        # 創新板 (Taiwan Innovation Board) is limited to qualified investors.
        df = df[~df["stock_name"].astype(str).str.endswith("-創")]
        current = pd.DataFrame(
            {
                "stock_id": df["stock_id"],
                "stock_name": df["stock_name"],
                "industry": df["industry_category"],
                "market": df["type"],
                "delisted_date": pd.NaT,
                "emerging_until": pd.to_datetime(df["stock_id"].map(emerging_until.to_dict())),
            }
        )

    gone = pd.DataFrame(columns=cols)
    if not delisting.empty:
        d = delisting.copy()
        d["stock_id"] = d["stock_id"].astype(str)
        d["date"] = pd.to_datetime(d["date"])
        d = d[d["stock_id"].str.match(COMMON_STOCK_ID)]
        d = d[d["date"] >= pd.Timestamp(since)]
        # If a code shows up delisted more than once, the latest delisting wins.
        d = d.sort_values("date").drop_duplicates("stock_id", keep="last")
        gone = pd.DataFrame(
            {
                "stock_id": d["stock_id"],
                "stock_name": d["stock_name"],
                "industry": pd.NA,
                "market": "unknown",
                "delisted_date": d["date"],
                "emerging_until": pd.NaT,
            }
        )
        # A code that is delisted but also currently listed was reused or
        # relisted. Treat it as current; the delisting record is history.
        gone = gone[~gone["stock_id"].isin(current["stock_id"])]

    frames = [f for f in (current, gone) if not f.empty]
    if not frames:
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("stock_id").reset_index(drop=True)
