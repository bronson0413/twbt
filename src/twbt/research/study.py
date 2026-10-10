"""The research study behind docs/research: one command, every number from data.

Protocol (fixed before looking at out-of-sample results):

* Universe: listed common stocks, 20-day average traded value ≥ NT$10M,
  construction (建材營造) excluded, data point-in-time.
* Three candidate signals, chosen from the literature rather than by search:
  REV (3-month revenue vs same months last year), MOM (12-1 month price
  momentum), COMBO (average percentile rank of the two).
* In-sample: signal dates before ``split``. Out-of-sample: from ``split`` on.
* Selection rule: the candidate whose top-30 portfolio has the highest
  in-sample Sharpe ratio. All candidates are reported out-of-sample anyway.
* Costs: commission 0.1425% × 2.8折 with NT$20 minimum, 0.3% sell tax,
  0.1% slippage per side, fills at the next day's open.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.query import Data
from ..engine import BacktestConfig, CostModel, MarketData, Result, run_backtest
from ..engine.context import Context
from ..engine.costs import ZERO_COST
from ..engine.metrics import max_drawdown
from ..engine.schedule import monthly_on_or_after
from ..strategies import (
    buy_and_hold,
    eligible,
    equal_weight,
    momentum_scores,
    rank_combo,
    revenue_scores,
    top_n,
)
from .factor import FactorReport, factor_analysis

BENCHMARK = "0050"
TRADING_DAYS = 245


@dataclass
class StudyConfig:
    start: str = "2012-01-01"
    end: str | None = None
    split: str = "2023-01-01"
    capital: float = 250_000
    main_n: int = 30
    grid_n: tuple[int, ...] = (10, 30, 50)
    min_avg_value: float = 10_000_000
    rebalance_day: int = 11
    groups: int = 5
    costs: CostModel = field(
        default_factory=lambda: CostModel(commission_discount=0.28, min_commission=20,
                                          slippage=0.001))
    list_price_costs: CostModel = field(
        default_factory=lambda: CostModel(commission_discount=1.0, min_commission=20,
                                          slippage=0.001))


def signals(cfg: StudyConfig) -> dict[str, Callable[[Context], pd.Series]]:
    def rev(ctx):
        return revenue_scores(ctx, months=3, method="sum")

    def mom(ctx):
        return momentum_scores(ctx, lookback=252, skip=21)

    return {"REV": rev, "MOM": mom, "COMBO": rank_combo([rev, mom])}


# -- statistics on a slice of an equity curve -----------------------------------


def curve_stats(equity: pd.Series) -> dict[str, float]:
    eq = equity.dropna()
    r = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    sd = r.std()
    return {
        "cagr": float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) if years > 0 else np.nan,
        "vol": float(sd * np.sqrt(TRADING_DAYS)),
        "sharpe": float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan,
        "max_dd": max_drawdown(eq),
    }


def active_stats(strategy: pd.Series, base: pd.Series) -> dict[str, float]:
    """Excess return over a baseline: annualized active return, tracking error, IR, t."""
    s, b = strategy.align(base, join="inner")
    active = (s.pct_change() - b.pct_change()).dropna()
    te = active.std() * np.sqrt(TRADING_DAYS)
    ann = active.mean() * TRADING_DAYS
    n_years = len(active) / TRADING_DAYS
    return {
        "active_return": float(ann),
        "tracking_error": float(te),
        "ir": float(ann / te) if te > 0 else np.nan,
        # t-stat of mean active return ≈ IR × sqrt(years)
        "t": float(ann / te * np.sqrt(n_years)) if te > 0 else np.nan,
    }


def split_curve(eq: pd.Series, split: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
    ins = eq[eq.index < split]
    # OOS starts from the last in-sample value so the first OOS day's return counts
    oos = eq[eq.index >= ins.index[-1]] if len(ins) else eq
    return ins, oos


def split_factor(rep: FactorReport, split: pd.Timestamp) -> tuple[FactorReport, FactorReport]:
    d = rep.by_date
    return (FactorReport(d[d.index < split], rep.groups),
            FactorReport(d[d.index >= split], rep.groups))


# -- the study ---------------------------------------------------------------


@dataclass
class StudyResult:
    cfg: StudyConfig
    factor: dict[str, FactorReport]
    liquidity: dict[str, dict[str, FactorReport]]
    portfolios: dict[str, Result]
    grid: pd.DataFrame
    selected: str
    last_date: pd.Timestamp
    data_fixes: int = 0
    stock_days: int = 0


def run_study(data: Data, cfg: StudyConfig, log: Callable[[str], None] = print) -> StudyResult:
    end = cfg.end or str(data.sql("SELECT MAX(date) AS d FROM prices")["d"].iloc[0].date())
    split = pd.Timestamp(cfg.split)
    log(f"loading market data {cfg.start} ~ {end}")
    market = MarketData.from_data(data, cfg.start, end, benchmarks=(BENCHMARK,))
    if BENCHMARK not in market.col:
        raise RuntimeError(f"基準 {BENCHMARK} 沒有價格資料")
    days = market.dates[(market.dates >= pd.Timestamp(cfg.start))
                        & (market.dates <= pd.Timestamp(end))]
    schedule = monthly_on_or_after(days, cfg.rebalance_day)
    sigs = signals(cfg)

    def universe(ctx):
        return eligible(ctx, cfg.min_avg_value)

    factor = {}
    for name, f in sigs.items():
        log(f"factor analysis {name}")
        factor[name] = factor_analysis(market, schedule, f, universe, groups=cfg.groups)

    liquidity: dict[str, dict[str, FactorReport]] = {}
    for name, f in sigs.items():
        liquidity[name] = {}
        for k, label in enumerate(("low", "mid", "high")):
            def uni_k(ctx, k=k):
                ok = universe(ctx)
                v = ctx.value(20).mean()[ok].rank(method="first")
                bucket = pd.qcut(v, 3, labels=False)
                return list(bucket[bucket == k].index)
            log(f"liquidity tercile {name}/{label}")
            liquidity[name][label] = factor_analysis(market, schedule, f, uni_k,
                                                     groups=cfg.groups, min_stocks=cfg.groups * 10)

    def bt(strategy, costs):
        return run_backtest(strategy, market, cfg.start, end, schedule,
                            BacktestConfig(capital=cfg.capital, costs=costs))

    portfolios: dict[str, Result] = {}
    for name, f in sigs.items():
        log(f"backtest {name} top{cfg.main_n}")
        portfolios[name] = bt(top_n(f, cfg.main_n, cfg.min_avg_value, name=name), cfg.costs)
    log("backtest baselines")
    # The equal-weight universe is an index, not a portfolio a NT$250k account
    # could hold: ~800 names at the NT$20 minimum fee would cost ~7% per trade.
    # It is therefore computed without costs, which makes it a harder baseline.
    portfolios["EQUAL"] = bt(equal_weight(cfg.min_avg_value), ZERO_COST)
    portfolios[BENCHMARK] = bt(buy_and_hold(BENCHMARK), cfg.costs)

    ins_sharpe = {n: curve_stats(split_curve(portfolios[n].equity, split)[0])["sharpe"]
                  for n in sigs}
    selected = max(ins_sharpe, key=lambda n: ins_sharpe[n])
    log(f"selected by in-sample Sharpe: {selected} {ins_sharpe}")

    rows = []
    for n in cfg.grid_n:
        for cost_label, costs in (("2.8折", cfg.costs), ("原價", cfg.list_price_costs)):
            if n == cfg.main_n and cost_label == "2.8折":
                res = portfolios[selected]
            else:
                log(f"grid {selected} top{n} {cost_label}")
                res = bt(top_n(sigs[selected], n, cfg.min_avg_value, name=selected), costs)
            ins, oos = split_curve(res.equity, split)
            rows.append({"n": n, "costs": cost_label,
                         "cagr_full": curve_stats(res.equity)["cagr"],
                         "sharpe_full": curve_stats(res.equity)["sharpe"],
                         "cagr_is": curve_stats(ins)["cagr"],
                         "cagr_oos": curve_stats(oos)["cagr"],
                         "costs_total": res.summary()["total_costs"],
                         "turnover": res.summary()["turnover_per_year"]})
    fixes = int(data.sql("SELECT COUNT(*) AS n FROM corporate_actions "
                         "WHERE kind = 'data_fix' AND date >= ?", [pd.Timestamp(cfg.start)])
                .iloc[0, 0])
    stock_days = int(np.isfinite(market.close[market.index_of(cfg.start):]).sum())
    return StudyResult(cfg, factor, liquidity, portfolios, pd.DataFrame(rows), selected,
                       market.dates[-1], fixes, stock_days)


# -- numbers for the report -----------------------------------------------------


def collect(res: StudyResult) -> dict:
    split = pd.Timestamp(res.cfg.split)
    out: dict = {"selected": res.selected, "split": res.cfg.split,
                 "start": res.cfg.start, "last_date": str(res.last_date.date()),
                 "main_n": res.cfg.main_n, "factor": {}, "liquidity": {}, "portfolio": {},
                 "data_fixes": res.data_fixes, "stock_days": res.stock_days}
    for name, rep in res.factor.items():
        ins, oos = split_factor(rep, split)
        out["factor"][name] = {"full": rep.summary(), "is": ins.summary(), "oos": oos.summary()}
    for name, by in res.liquidity.items():
        out["liquidity"][name] = {k: v.summary() for k, v in by.items()}
    eq_equal = res.portfolios["EQUAL"].equity
    for name, r in res.portfolios.items():
        ins, oos = split_curve(r.equity, split)
        e_ins, e_oos = split_curve(eq_equal, split)
        out["portfolio"][name] = {
            "full": curve_stats(r.equity), "is": curve_stats(ins), "oos": curve_stats(oos),
            "vs_equal_full": active_stats(r.equity, eq_equal),
            "vs_equal_is": active_stats(ins, e_ins),
            "vs_equal_oos": active_stats(oos, e_oos),
            "turnover": r.summary()["turnover_per_year"],
            "costs_total": r.summary()["total_costs"],
            "final": float(r.equity.iloc[-1]),
        }
    out["grid"] = res.grid.to_dict(orient="records")
    sel = res.factor[res.selected].by_date
    out["yearly_spread"] = {
        int(y): float(v) for y, v in
        ((1 + sel[f"Q{res.cfg.groups}"]).groupby(sel.index.year).prod()
         - (1 + sel["Q1"]).groupby(sel.index.year).prod()).items()
    }
    return out


def save(res: StudyResult, out_dir: str | Path) -> dict:
    from .report import write_report  # matplotlib only needed here

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    numbers = collect(res)
    (out_dir / "results.json").write_text(json.dumps(numbers, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    write_report(res, numbers, out_dir)
    return numbers
