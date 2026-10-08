"""Which FinMind datasets we download and how.

Every dataset here is available on the FinMind free tier. Free accounts must
query per stock (``data_id``) for per-stock datasets; "global" datasets come
back in a single call.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    per_stock: bool
    # Columns that uniquely identify a row. Used to dedupe on re-download.
    key: tuple[str, ...]
    # On incremental updates, re-fetch this many days before the last stored
    # date. Monthly revenue is sometimes revised after publication, so we
    # re-pull recent months to pick up corrections.
    overlap_days: int = 0
    # Free-text note for the README / status output.
    note: str = ""


STOCK_INFO = DatasetSpec("TaiwanStockInfo", per_stock=False, key=("stock_id", "type"))
DELISTING = DatasetSpec("TaiwanStockDelisting", per_stock=False, key=("stock_id", "date"))
TRADING_DATE = DatasetSpec("TaiwanStockTradingDate", per_stock=False, key=("date",))
SPLIT = DatasetSpec("TaiwanStockSplitPrice", per_stock=False, key=("stock_id", "date"))
PAR_VALUE = DatasetSpec("TaiwanStockParValueChange", per_stock=False, key=("stock_id", "date"))

PRICE = DatasetSpec("TaiwanStockPrice", per_stock=True, key=("stock_id", "date"))
MONTH_REVENUE = DatasetSpec(
    "TaiwanStockMonthRevenue",
    per_stock=True,
    key=("stock_id", "revenue_year", "revenue_month"),
    overlap_days=120,
    note="revisions overwrite older values; only the latest version is kept",
)
DIVIDEND_RESULT = DatasetSpec("TaiwanStockDividendResult", per_stock=True, key=("stock_id", "date"))
CAPITAL_REDUCTION = DatasetSpec(
    "TaiwanStockCapitalReductionReferencePrice", per_stock=True, key=("stock_id", "date")
)

GLOBAL_DATASETS: tuple[DatasetSpec, ...] = (STOCK_INFO, DELISTING, TRADING_DATE, SPLIT, PAR_VALUE)
PER_STOCK_DATASETS: tuple[DatasetSpec, ...] = (
    PRICE,
    MONTH_REVENUE,
    DIVIDEND_RESULT,
    CAPITAL_REDUCTION,
)

BY_NAME: dict[str, DatasetSpec] = {d.name: d for d in GLOBAL_DATASETS + PER_STOCK_DATASETS}
