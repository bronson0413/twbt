"""Daily simulation loop.

Timeline of one trading day ``t``:

1. Corporate actions dated ``t`` hit holdings before the open
   (cash dividends paid in cash; stock dividends, splits, reductions change shares).
2. Holdings of stocks delisted on ``t`` are converted to cash.
3. If a signal was produced yesterday, orders fill at today's **open**.
4. Holdings are marked at today's close.
5. If ``t`` is a rebalance day, the strategy runs on data up to ``t``'s close.
   Its target weights are filled at the next day's open (step 3).

Signals never trade on the same bar they were computed from.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import metrics
from .context import Context
from .costs import CostModel
from .market import MarketData

Strategy = Callable[[Context], dict[str, float]]

# TWSE/TPEx daily price limit: ±7% before 2015-06-01, ±10% since.
LIMIT_CHANGE_DATE = pd.Timestamp("2015-06-01")
CASH_DIVIDEND_LABELS = {"息", "除息"}


@dataclass
class BacktestConfig:
    capital: float = 250_000.0
    costs: CostModel = field(default_factory=CostModel)
    lot_size: int = 1  # 1 = odd lots allowed; 1000 = board lots only
    # Fraction of the last close recovered when a held stock is delisted.
    # 1.0 is optimistic for bankruptcies; check the delisting events in the
    # result and rerun with 0.0 to see how much they matter.
    delist_recovery: float = 1.0


@dataclass
class Result:
    equity: pd.Series
    cash: pd.Series
    trades: pd.DataFrame
    signals: list[tuple[pd.Timestamp, dict[str, float]]]
    config: BacktestConfig

    def summary(self, rf: float = 0.0) -> dict[str, float]:
        return metrics.summary(self.equity, self.trades, rf)

    def monthly_returns(self) -> pd.DataFrame:
        return metrics.monthly_returns(self.equity)

    def compare(self, benchmark: Result) -> dict[str, float]:
        return metrics.compare(self.equity, benchmark.equity)


TRADE_COLUMNS = ["date", "stock_id", "side", "shares", "price", "value", "commission", "tax"]


class _Book:
    """Cash, holdings and the trade log."""

    def __init__(self, market: MarketData, cfg: BacktestConfig) -> None:
        self.m = market
        self.cfg = cfg
        self.cash = float(cfg.capital)
        self.shares = np.zeros(len(market.ids))
        self.last_close = np.full(len(market.ids), np.nan)
        self.log: list[tuple] = []

    def record(self, date, col, side, shares, price, value, commission=0.0, tax=0.0):
        self.log.append((date, self.m.ids[col], side, float(shares), float(price), float(value),
                         float(commission), float(tax)))

    def equity(self, prices: np.ndarray) -> float:
        held = self.shares != 0
        return self.cash + float(np.nansum(self.shares[held] * prices[held]))

    # -- step 1 -------------------------------------------------------------
    def corporate_actions(self, t: int) -> None:
        date = self.m.dates[t]
        for col, kind, detail, before, ref, factor in self.m.actions_on.get(t, []):
            self.last_close[col] = ref  # valuation basis if no quote today
            held = self.shares[col]
            if held == 0:
                continue
            if kind == "dividend" and detail in CASH_DIVIDEND_LABELS:
                amount = held * (before - ref)
                self.cash += amount
                self.record(date, col, "dividend", held, before - ref, amount)
            else:
                new = held * factor
                whole = math.floor(new + 1e-9)
                # fractions of a share are paid out in cash
                cash_in_lieu = (new - whole) * ref
                self.cash += cash_in_lieu
                self.shares[col] = whole
                self.record(date, col, kind, whole - held, ref, cash_in_lieu)

    # -- step 2 -------------------------------------------------------------
    def delistings(self, t: int) -> None:
        for col in np.flatnonzero((self.m.delist_idx == t) & (self.shares != 0)):
            px = self.last_close[col] * self.cfg.delist_recovery
            px = 0.0 if np.isnan(px) else px
            value = self.shares[col] * px
            self.cash += value
            self.record(self.m.dates[t], col, "delist", -self.shares[col], px, value)
            self.shares[col] = 0.0

    # -- step 3 -------------------------------------------------------------
    def rebalance(self, t: int, weights: dict[str, float]) -> None:
        m, c, date = self.m, self.cfg.costs, self.m.dates[t]
        o, hi, lo = m.open[t], m.high[t], m.low[t]
        prev = self.last_close
        ref = np.where(np.isnan(o), prev, o)
        equity = self.equity(ref)

        target_w = np.zeros(len(m.ids))
        for sid, w in weights.items():
            target_w[m.col[sid]] = w

        tradable = ~np.isnan(o) & (np.nan_to_num(o) > 0)
        thr = 0.095 if date >= LIMIT_CHANGE_DATE else 0.065
        with np.errstate(invalid="ignore"):
            locked = tradable & (o == hi) & (o == lo)
            locked_up = locked & (o >= prev * (1 + thr))
            locked_down = locked & (o <= prev * (1 - thr))

        lot = self.cfg.lot_size
        target = self.shares.copy()
        for col in np.flatnonzero(tradable):
            want = target_w[col] * equity / c.buy_price(o[col])
            if not np.isfinite(want):
                continue
            target[col] = math.floor(want / lot) * lot

        # sells first, to raise cash
        for col in np.flatnonzero(tradable & ~locked_down & (target < self.shares)):
            qty = self.shares[col] - target[col]
            px = c.sell_price(o[col])
            value = qty * px
            fee, tax = c.commission(value), c.tax(value, m.ids[col])
            self.cash += value - fee - tax
            self.shares[col] = target[col]
            self.record(date, col, "sell", -qty, px, value, fee, tax)

        buys = np.flatnonzero(tradable & ~locked_up & (target > self.shares))
        if len(buys) == 0:
            return
        qty = target[buys] - self.shares[buys]
        need = sum(c.buy_cost(q, o[col]) for q, col in zip(qty, buys, strict=True))
        if need > self.cash:  # not enough cash: scale all buys down evenly
            scale = self.cash / need
            qty = np.floor(qty * scale / lot) * lot
        # biggest targets first; trim any order that still doesn't fit
        for i in np.argsort(-target_w[buys]):
            col, q = buys[i], qty[i]
            while q > 0 and c.buy_cost(q, o[col]) > self.cash:
                q -= lot
            if q <= 0:
                continue
            px = c.buy_price(o[col])
            value = q * px
            fee = c.commission(value)
            self.cash -= value + fee
            self.shares[col] += q
            self.record(date, col, "buy", q, px, value, fee)

    # -- step 4 -------------------------------------------------------------
    def mark(self, t: int) -> float:
        today = self.m.close[t]
        self.last_close = np.where(np.isnan(today), self.last_close, today)
        return self.equity(self.last_close)


def _validate(weights: dict[str, float], market: MarketData, date) -> dict[str, float]:
    unknown = [s for s in weights if s not in market.col]
    if unknown:
        raise ValueError(f"{date.date()}: unknown stocks {unknown[:5]}")
    if any(w < 0 for w in weights.values()):
        raise ValueError(f"{date.date()}: negative weights (no shorting)")
    total = sum(weights.values())
    if total > 1 + 1e-9:
        raise ValueError(f"{date.date()}: weights sum to {total:.4f} > 1 (no leverage)")
    return {s: w for s, w in weights.items() if w > 0}


def run_backtest(
    strategy: Strategy,
    market: MarketData,
    start,
    end,
    rebalance_dates: Iterable,
    config: BacktestConfig | None = None,
) -> Result:
    cfg = config or BacktestConfig()
    book = _Book(market, cfg)
    t0, t1 = market.index_of(start), market.index_of(pd.Timestamp(end) + pd.Timedelta(days=1))
    if t0 >= t1:
        raise ValueError("no trading days between start and end")
    rebal = {pd.Timestamp(d) for d in rebalance_dates}

    # closes before the start are the valuation basis on day one
    for t in range(t0):
        today = market.close[t]
        book.last_close = np.where(np.isnan(today), book.last_close, today)

    pending: dict[str, float] | None = None
    signals: list[tuple[pd.Timestamp, dict[str, float]]] = []
    equity, cash = [], []
    for t in range(t0, t1):
        book.corporate_actions(t)
        book.delistings(t)
        if pending is not None:
            book.rebalance(t, pending)
            pending = None
        equity.append(book.mark(t))
        cash.append(book.cash)
        date = market.dates[t]
        if date in rebal:
            pending = _validate(strategy(Context(market, t)), market, date)
            signals.append((date, pending))

    idx = market.dates[t0:t1]
    return Result(
        equity=pd.Series(equity, index=idx, name="equity"),
        cash=pd.Series(cash, index=idx, name="cash"),
        trades=pd.DataFrame(book.log, columns=TRADE_COLUMNS),
        signals=signals,
        config=cfg,
    )
