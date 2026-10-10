"""High-level runs used by the CLI: backtest vs benchmark, and this month's signal."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from .data.query import Data
from .engine import BacktestConfig, Context, CostModel, MarketData, Result, run_backtest
from .engine.schedule import monthly_on_or_after
from .research.factor import factor_analysis, format_report
from .strategies import buy_and_hold, equal_weight, revenue_momentum
from .strategies.revenue_momentum import DEFAULT_EXCLUDED_INDUSTRIES, eligible, revenue_scores

BENCHMARK = "0050"


@dataclass
class Settings:
    start: str = "2013-01-01"
    end: str = date.today().isoformat()
    capital: float = 250_000
    strategy_name: str = "revenue"
    top_n: int = 8
    months: int = 3
    method: str = "sum"
    min_consecutive: int = 3
    exclude_construction: bool = True
    min_avg_value: float = 10_000_000
    rebalance_day: int = 11
    commission_discount: float = 1.0
    min_commission: float = 20.0
    slippage: float = 0.001
    lot_size: int = 1
    delist_recovery: float = 1.0

    def config(self) -> BacktestConfig:
        costs = CostModel(commission_discount=self.commission_discount,
                          min_commission=self.min_commission, slippage=self.slippage)
        return BacktestConfig(capital=self.capital, costs=costs, lot_size=self.lot_size,
                              delist_recovery=self.delist_recovery)

    @property
    def excluded(self) -> tuple[str, ...]:
        return DEFAULT_EXCLUDED_INDUSTRIES if self.exclude_construction else ()

    def strategy(self):
        if self.strategy_name == "equal":
            return equal_weight(min_avg_value=self.min_avg_value)
        if self.strategy_name in ("combo", "momentum"):
            from .research.study import StudyConfig, signals
            from .strategies import top_n
            f = signals(StudyConfig())["COMBO" if self.strategy_name == "combo" else "MOM"]
            return top_n(f, self.top_n, self.min_avg_value, self.excluded,
                         name=self.strategy_name)
        return revenue_momentum(top_n=self.top_n, months=self.months,
                                min_avg_value=self.min_avg_value, method=self.method,
                                min_consecutive=self.min_consecutive,
                                exclude_industries=self.excluded)


def backtest(data_dir: str | Path, s: Settings) -> tuple[Result, Result]:
    data = Data(data_dir)
    market = MarketData.from_data(data, s.start, s.end, benchmarks=(BENCHMARK,))
    if BENCHMARK not in market.col:
        raise RuntimeError(f"基準 {BENCHMARK} 還沒有價格資料（下載尚未完成），請等下載完再回測")
    days = market.dates[(market.dates >= pd.Timestamp(s.start))
                        & (market.dates <= pd.Timestamp(s.end))]
    schedule = monthly_on_or_after(days, s.rebalance_day)
    strat = run_backtest(s.strategy(), market, s.start, s.end, schedule, s.config())
    bench = run_backtest(buy_and_hold(BENCHMARK), market, s.start, s.end, schedule, s.config())
    return strat, bench


def signal(data_dir: str | Path, s: Settings) -> pd.DataFrame:
    """Target portfolio from the latest data, with share counts at the last close."""
    data = Data(data_dir)
    market = MarketData.from_data(data, s.end, s.end, benchmarks=())
    t = len(market.dates) - 1
    ctx = Context(market, t)
    weights = s.strategy()(ctx)
    close = ctx.close(1).iloc[-1]
    names = data.sql("SELECT stock_id, stock_name FROM universe").set_index("stock_id")
    rows = []
    for sid, w in sorted(weights.items(), key=lambda kv: -kv[1]):
        px = close[sid]
        shares = int(w * s.capital / px // s.lot_size * s.lot_size)
        rows.append({"stock_id": sid, "name": names["stock_name"].get(sid, ""), "weight": w,
                     "close": px, "shares": shares, "value": shares * px})
    out = pd.DataFrame(rows)
    out.attrs["date"] = ctx.date.date().isoformat()
    return out


def _width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int, right: bool = False) -> str:
    gap = " " * max(width - _width(text), 0)
    return gap + text if right else text + gap


def format_summary(strat: Result, bench: Result) -> str:
    a, b = strat.summary(), bench.summary()
    rows = [
        ("期間", f"{a['start']} ~ {a['end']}", ""),
        ("期末資產", f"{a['final_equity']:,.0f}", f"{b['final_equity']:,.0f}"),
        ("年化報酬", f"{a['cagr']:.1%}", f"{b['cagr']:.1%}"),
        ("年化波動", f"{a['ann_vol']:.1%}", f"{b['ann_vol']:.1%}"),
        ("Sharpe", f"{a['sharpe']:.2f}", f"{b['sharpe']:.2f}"),
        ("最大回撤", f"{a['max_drawdown']:.1%}", f"{b['max_drawdown']:.1%}"),
        ("年換手率", f"{a['turnover_per_year']:.1f}x", f"{b['turnover_per_year']:.1f}x"),
        ("交易成本合計", f"{a['total_costs']:,.0f}", f"{b['total_costs']:,.0f}"),
        ("交易筆數", f"{a['n_trades']}", f"{b['n_trades']}"),
    ]
    c = strat.compare(bench)
    lines = [_pad("", 14) + _pad("策略", 16, True) + _pad(BENCHMARK, 16, True)]
    lines += [_pad(k, 14) + _pad(x, 16, True) + _pad(y, 16, True) for k, x, y in rows[1:]]
    lines.insert(0, f"期間 {rows[0][1]}")
    lines.append(f"\n超額年化報酬 {c['excess_cagr']:+.1%}，月勝率 {c['monthly_win_rate']:.0%}，"
                 f"追蹤誤差 {c['tracking_error']:.1%}")
    delist = strat.trades[strat.trades["side"] == "delist"]
    if len(delist):
        lines.append(f"注意：持有期間遇到 {len(delist)} 次下市，以最後收盤價結算"
                     f"（--delist-recovery 0 可看最壞情況）")
    return "\n".join(lines)


# Signal variants compared by `twbt factor`: (label, method, min_consecutive, exclude_construction)
FACTOR_VARIANTS = [
    ("原版：單月年增率平均", "mean", 0, False),
    ("改用 3 個月合計", "sum", 0, False),
    ("合計 + 排除營建", "sum", 0, True),
    ("合計 + 排除營建 + 連續 3 個月成長", "sum", 3, True),
]


def factor(data_dir: str | Path, s: Settings, groups: int = 5) -> str:
    data = Data(data_dir)
    market = MarketData.from_data(data, s.start, s.end, benchmarks=())
    days = market.dates[(market.dates >= pd.Timestamp(s.start))
                        & (market.dates <= pd.Timestamp(s.end))]
    schedule = monthly_on_or_after(days, s.rebalance_day)
    out = [f"期間 {s.start} ~ {s.end}，每月 {s.rebalance_day} 日後第一個交易日分組，"
           f"持有到下一次；不計成本、收盤價對收盤價"]
    for label, method, consec, excl in FACTOR_VARIANTS:
        banned = DEFAULT_EXCLUDED_INDUSTRIES if excl else ()

        def scorer(ctx, method=method, consec=consec):
            return revenue_scores(ctx, s.months, method=method, min_consecutive=consec)

        def universe(ctx, banned=banned):
            return eligible(ctx, s.min_avg_value, exclude_industries=banned)

        # The consecutive-growth filter removes stocks rather than ranking them,
        # so too few may be left to split into groups: use fewer groups.
        rep = factor_analysis(market, schedule, scorer, universe,
                              groups=groups, min_stocks=groups * 10)
        out.append("")
        out.append(format_report(label, rep))
    return "\n".join(out)
