from __future__ import annotations

from ..engine.context import Context


def buy_and_hold(stock_id: str):
    """Hold one security at 100%. On a rebalance day, idle cash (e.g. dividends) is reinvested."""

    def strategy(ctx: Context) -> dict[str, float]:
        return {stock_id: 1.0}

    strategy.__name__ = f"buy_and_hold_{stock_id}"
    return strategy


def equal_weight(min_avg_value: float = 10_000_000, value_days: int = 20):
    """Every liquid stock in the universe, equally weighted: a sanity baseline.

    If this roughly tracks the market, the engine and data are behaving; a
    stock-picking strategy should then be judged against it as well as 0050.
    """

    def strategy(ctx: Context) -> dict[str, float]:
        avg_value = ctx.value(value_days).mean()
        last = ctx.close(1).iloc[-1]
        ids = [s for s in ctx.universe()
               if avg_value.get(s, 0) >= min_avg_value and last.get(s, 0) > 0]
        return {s: 1.0 / len(ids) for s in ids} if ids else {}

    strategy.__name__ = "equal_weight"
    return strategy
