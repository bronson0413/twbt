"""Read-only, point-in-time access to the clean layer via DuckDB.

Strategies get data through this class only. Every accessor that could leak
future information takes an ``asof`` date and filters on it, so a strategy
cannot accidentally read revenue that wasn't published yet.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

TABLES = ("universe", "trading_days", "prices", "revenue", "corporate_actions")


class Data:
    def __init__(self, root: str | Path) -> None:
        clean = Path(root) / "clean"
        missing = [t for t in TABLES if not (clean / f"{t}.parquet").exists()]
        if missing:
            raise FileNotFoundError(f"clean layer missing {missing}; run `twbt build` first")
        self.con = duckdb.connect()
        for t in TABLES:
            path = (clean / f"{t}.parquet").as_posix().replace("'", "''")
            self.con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{path}')")

    def sql(self, query: str, params: list | None = None) -> pd.DataFrame:
        return self.con.execute(query, params or []).df()

    # -- point-in-time accessors ----------------------------------------------

    def revenue_asof(self, asof: str | pd.Timestamp, months: int | None = None) -> pd.DataFrame:
        """Revenue rows already public on ``asof`` (available_date <= asof).

        ``months``: keep only the latest N published months per stock.
        """
        asof = pd.Timestamp(asof)
        q = """
            SELECT stock_id, period, revenue_year, revenue_month, revenue, available_date
            FROM revenue
            WHERE available_date <= ?
        """
        df = self.sql(q, [asof])
        if months is not None and not df.empty:
            df = (
                df.sort_values(["stock_id", "revenue_year", "revenue_month"])
                .groupby("stock_id")
                .tail(months)
            )
        return df.reset_index(drop=True)

    def universe_on(self, asof: str | pd.Timestamp) -> pd.DataFrame:
        """Stocks tradable on ``asof``: already trading and not yet delisted."""
        asof = pd.Timestamp(asof)
        q = """
            WITH first_trade AS (
                SELECT stock_id, MIN(date) AS first_date
                FROM prices WHERE has_quote GROUP BY stock_id
            )
            SELECT u.*, f.first_date
            FROM universe u JOIN first_trade f USING (stock_id)
            WHERE f.first_date <= ?
              AND (u.delisted_date IS NULL OR u.delisted_date > ?)
            ORDER BY stock_id
        """
        return self.sql(q, [asof, asof])

    def prices(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        stocks: list[str] | None = None,
    ) -> pd.DataFrame:
        q = "SELECT * FROM prices WHERE date BETWEEN ? AND ?"
        params: list = [pd.Timestamp(start), pd.Timestamp(end)]
        if stocks:
            q += " AND stock_id IN (SELECT UNNEST(?))"
            params.append(list(stocks))
        return self.sql(q + " ORDER BY stock_id, date", params)

    def trading_days(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
        df = self.sql(
            "SELECT date FROM trading_days WHERE date BETWEEN ? AND ? ORDER BY date",
            [pd.Timestamp(start), pd.Timestamp(end)],
        )
        return pd.DatetimeIndex(df["date"])

    def corporate_actions(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
        return self.sql(
            "SELECT * FROM corporate_actions WHERE date BETWEEN ? AND ? "
            "ORDER BY stock_id, date, kind",
            [pd.Timestamp(start), pd.Timestamp(end)],
        )
