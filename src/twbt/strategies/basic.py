from __future__ import annotations

from ..engine.context import Context


def buy_and_hold(stock_id: str):
    """Hold one security at 100%. On a rebalance day, idle cash (e.g. dividends) is reinvested."""

    def strategy(ctx: Context) -> dict[str, float]:
        return {stock_id: 1.0}

    strategy.__name__ = f"buy_and_hold_{stock_id}"
    return strategy
