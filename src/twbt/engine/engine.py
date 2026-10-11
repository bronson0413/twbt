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
    # Sell a holding once its total return since purchase falls below -stop_loss
    # (checked at each close, sold at the next fill); it stays in cash until the
    # next rebalance. None = no stop.
    stop_loss: float | None = None
    # "next_open": orders fill at the next day's open (realistic).
    # "same_close": orders fill at the close of the signal day itself — what
    # some libraries assume by default; it cannot be traded, since the close is
    # needed to compute the signal. For comparison only.
    fill: str = "next_open"


@dataclass
class Result:
    equity: pd.Series
    cash: pd.Series
    trades: pd.DataFrame
    signals: list[tuple[pd.Timestamp, dict[str, float]]]
    config: BacktestConfig
    # shares held at each day's close: {stock_id: shares}, one dict per day
    positions: list[dict[str, float]] = field(default_factory=list)
    market: MarketData | None = field(default=None, repr=False)

    def attribution(self, date) -> pd.DataFrame:
        """Each holding's contribution to the move from the previous close to ``date``'s close.

        Contribution = shares held at the previous close × (close × event factor − previous
        close). Trades made at the open are ignored, so it explains drift, not fills.
        """
        m = self.market
        t = m.index_of(date) - m.index_of(self.equity.index[0])
        if t <= 0:
            raise ValueError("no previous day in the backtest")
        g = m.index_of(date)
        prev_pos = self.positions[t - 1]
        factor = {m.ids[col]: f for col, *_r, f in m.actions_on.get(g, [])}
        rows = []
        for sid, sh in prev_pos.items():
            c = m.col[sid]
            p0 = self._last(c, g - 1)
            p1 = m.close[g, c] if not np.isnan(m.close[g, c]) else p0
            pnl = sh * (p1 * factor.get(sid, 1.0) - p0)
            rows.append({"stock_id": sid, "shares": sh, "prev_close": p0, "close": p1,
                         "event_factor": factor.get(sid, 1.0), "pnl": pnl})
        df = pd.DataFrame(rows)
        base = self.equity.iloc[t - 1]
        if not df.empty:
            df["pct_of_equity"] = df["pnl"] / base
            df = df.sort_values("pnl", key=np.abs, ascending=False).reset_index(drop=True)
        return df

    def _last(self, col: int, t: int) -> float:
        c = self.market.close[: t + 1, col]
        valid = c[~np.isnan(c)]
        return float(valid[-1]) if len(valid) else np.nan

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
        # col -> (total return index on the buy day, fill price / that day's close)
        self.entry_tri: dict[int, tuple[float, float]] = {}

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
    def rebalance(self, t: int, weights: dict[str, float], at_close: bool = False) -> None:
        m, c, date = self.m, self.cfg.costs, self.m.dates[t]
        o, hi, lo = (m.close[t] if at_close else m.open[t]), m.high[t], m.low[t]
        prev = self.last_close if not at_close else self._prev_close(t)
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
            if target[col] == 0:
                self.entry_tri.pop(col, None)

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
            ratio = c.buy_price(o[col]) / m.close[t, col] if m.close[t, col] > 0 else 1.0
            self.entry_tri.setdefault(col, (self.m.tri[t, col], ratio))

    def sell_all(self, t: int, cols: set[int], at_close: bool = False) -> set[int]:
        """Sell every share of ``cols``; returns those that could not trade today."""
        m, c, date = self.m, self.cfg.costs, self.m.dates[t]
        px_row = m.close[t] if at_close else m.open[t]
        prev = self._prev_close(t) if at_close else self.last_close
        thr = 0.095 if date >= LIMIT_CHANGE_DATE else 0.065
        left = set()
        for col in cols:
            if self.shares[col] == 0:
                continue
            p = px_row[col]
            locked = (not np.isnan(p) and p == m.high[t, col] == m.low[t, col]
                      and p <= prev[col] * (1 - thr))
            if np.isnan(p) or p <= 0 or locked:
                left.add(col)
                continue
            qty = self.shares[col]
            sp = c.sell_price(p)
            value = qty * sp
            fee, tax = c.commission(value), c.tax(value, m.ids[col])
            self.cash += value - fee - tax
            self.shares[col] = 0.0
            self.entry_tri.pop(col, None)
            self.record(date, col, "stop", -qty, sp, value, fee, tax)
        return left

    def _prev_close(self, t: int) -> np.ndarray:
        if t == 0:
            return self.last_close
        prev = self.m.close[t - 1]
        return np.where(np.isnan(prev), self.last_close, prev)

    def stopped(self, t: int, stop: float) -> list[int]:
        """Holdings whose total return since purchase (at the fill price) is below -stop."""
        out = []
        for col, (tri0, fill_vs_close) in list(self.entry_tri.items()):
            if self.shares[col] == 0:
                continue
            r = self.m.tri[t, col] / (tri0 * fill_vs_close) - 1
            if np.isfinite(r) and r < -stop:
                out.append(col)
        return out

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

    if cfg.fill not in ("next_open", "same_close"):
        raise ValueError(f"unknown fill {cfg.fill!r}")
    same_close = cfg.fill == "same_close"
    pending: dict[str, float] | None = None
    exits: set[int] = set()
    signals: list[tuple[pd.Timestamp, dict[str, float]]] = []
    positions: list[dict[str, float]] = []
    equity, cash = [], []
    for t in range(t0, t1):
        book.corporate_actions(t)
        book.delistings(t)
        if pending is not None:
            book.rebalance(t, pending, at_close=same_close)
            pending = None
            exits.clear()
        if exits:
            exits = book.sell_all(t, exits, at_close=same_close)
        date = market.dates[t]
        if same_close and date in rebal:
            target = _validate(strategy(Context(market, t)), market, date)
            signals.append((date, target))
            book.rebalance(t, target, at_close=True)
            exits.clear()
        elif date in rebal:
            pending = _validate(strategy(Context(market, t)), market, date)
            signals.append((date, pending))
        if cfg.stop_loss and pending is None:
            exits |= set(book.stopped(t, cfg.stop_loss))
        equity.append(book.mark(t))
        held = np.flatnonzero(book.shares)
        positions.append({market.ids[c]: float(book.shares[c]) for c in held})
        cash.append(book.cash)

    idx = market.dates[t0:t1]
    return Result(
        equity=pd.Series(equity, index=idx, name="equity"),
        cash=pd.Series(cash, index=idx, name="cash"),
        trades=pd.DataFrame(book.log, columns=TRADE_COLUMNS),
        signals=signals,
        config=cfg,
        positions=positions,
        market=market,
    )
