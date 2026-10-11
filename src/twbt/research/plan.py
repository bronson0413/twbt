"""The live trading plan: one fixed design, checked against history.

The design was fixed on 2026-10-11 from the owner's constraints and the
first study, before this check was run:

* signal COMBO (monthly revenue growth + 12-1 price momentum) — the
  Taiwan-specific signal the first study selected by its pre-set rule
* only large stocks: the 150 highest by 120-day average traded value
  (owner: no small, news-driven stocks)
* 10 holdings, equal weight, NT$300,000, odd lots
* buffer: a holding stays while it ranks in the top 20, to cut turnover
* the owner's broker costs: 0.1425% × 6.5折, NT$1 minimum for odd lots,
  0.3% sell tax, 0.2% slippage per side (odd-lot spreads are wider)

This check only shows what the design costs and how it behaved; nothing is
chosen from it. The period from 2023 had already been looked at in the
earlier studies, so none of it is a clean out-of-sample test. The clean
test is live trading from 2026-10 on.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.query import Data
from ..engine import BacktestConfig, CostModel, MarketData, run_backtest
from ..engine.context import Context
from ..engine.costs import ZERO_COST
from ..engine.schedule import monthly_on_or_after
from ..strategies import buy_and_hold, eligible, top_n
from .search import candidates
from .study import active_stats, curve_stats, split_curve

BENCHMARK = "0050"


@dataclass
class Plan:
    selected: str = "COMBO"
    n: int = 10
    capital: float = 300_000
    top_value: int = 150
    keep_rank: int = 20
    commission_discount: float = 0.65
    min_commission: float = 1.0
    slippage: float = 0.002
    min_avg_value: float = 10_000_000
    rebalance_day: int = 11
    fixed_on: str = "2026-10-11"
    start: str = "2012-01-01"
    split: str = "2023-01-01"

    def costs(self) -> CostModel:
        return CostModel(commission_discount=self.commission_discount,
                         min_commission=self.min_commission, slippage=self.slippage)


def large_equal(plan: Plan):
    """Every stock in the plan's large-stock universe, equal weight (a baseline)."""
    def strategy(ctx: Context) -> dict[str, float]:
        ids = eligible(ctx, plan.min_avg_value, top_value=plan.top_value)
        return {s: 1.0 / len(ids) for s in ids} if ids else {}

    strategy.__name__ = "large_equal"
    return strategy


def check(data: Data, plan: Plan, log=print) -> dict:
    end = str(data.sql("SELECT MAX(date) AS d FROM prices")["d"].iloc[0].date())
    split = pd.Timestamp(plan.split)
    market = MarketData.from_data(data, plan.start, end, benchmarks=(BENCHMARK,))
    days = market.dates[(market.dates >= pd.Timestamp(plan.start))
                        & (market.dates <= pd.Timestamp(end))]
    schedule = monthly_on_or_after(days, plan.rebalance_day)
    f = candidates()[plan.selected]

    def bt(strategy, costs):
        return run_backtest(strategy, market, plan.start, end, schedule,
                            BacktestConfig(capital=plan.capital, costs=costs))

    runs = {}
    log("plan")
    runs["PLAN"] = bt(top_n(f, plan.n, plan.min_avg_value, name="plan",
                            top_value=plan.top_value, keep_rank=plan.keep_rank), plan.costs())
    log("plan without buffer")
    runs["NO_BUFFER"] = bt(top_n(f, plan.n, plan.min_avg_value, name="no_buffer",
                                 top_value=plan.top_value), plan.costs())
    log("baselines")
    runs["LARGE_EQUAL"] = bt(large_equal(plan), ZERO_COST)
    runs[BENCHMARK] = bt(buy_and_hold(BENCHMARK), plan.costs())

    base = runs["LARGE_EQUAL"].equity
    b_ins, b_oos = split_curve(base, split)
    out = {"plan": asdict(plan), "last_date": end, "runs": {}}
    for name, r in runs.items():
        eq = r.equity
        ins, oos = split_curve(eq, split)
        years = (eq.index[-1] - eq.index[0]).days / 365.25
        s = r.summary()
        out["runs"][name] = {
            "full": curve_stats(eq), "is": curve_stats(ins), "oos": curve_stats(oos),
            "vs_large_is": active_stats(ins, b_ins), "vs_large_oos": active_stats(oos, b_oos),
            "turnover": s["turnover_per_year"],
            "fees_tax_per_year": float(s["total_costs"] / eq.mean() / years),
            "worst_12m": float((eq / eq.shift(245) - 1).min()),
        }
    plan_eq, bench_eq = runs["PLAN"].equity, runs[BENCHMARK].equity
    rel = (plan_eq / plan_eq.shift(245)) / (bench_eq / bench_eq.shift(245)) - 1
    out["worst_12m_vs_0050"] = float(rel.min())
    out["best_12m_vs_0050"] = float(rel.max())
    out["yearly"] = {}
    for name in ("PLAN", "LARGE_EQUAL", BENCHMARK):
        eq = runs[name].equity
        last = eq.groupby(eq.index.year).last()
        prev = last.shift(1).fillna(eq.iloc[0])
        out["yearly"][name] = {int(y): float(v) for y, v in (last / prev - 1).items()}
    out["months_beating_0050"] = float(
        (plan_eq.resample("ME").last().pct_change() > bench_eq.resample("ME").last().pct_change())
        .iloc[1:].mean())
    return out


def _p(x, signed=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:+.1%}" if signed else f"{x:.1%}"


def render(o: dict) -> str:
    p, R = o["plan"], o["runs"]
    names = {"PLAN": "實盤設計（含緩衝）", "NO_BUFFER": "同設計、不加緩衝",
             "LARGE_EQUAL": "大型股池等權重（不計成本）", BENCHMARK: "0050"}
    L = [
        "# 實盤設計與歷史檢查",
        "",
        f"> 由 `twbt plan` 自動產生。設計於 {p['fixed_on']} 固定，之後才跑這份檢查；"
        f"資料 {p['start']} ~ {o['last_date']}。",
        "",
        "## 設計（固定，不依結果調整）",
        "",
        f"- 訊號：{p['selected']}（近 3 個月營收年增 ＋ 12-1 個月價格動能，百分位排名平均）",
        f"- 股票池：120 日平均成交值前 {p['top_value']} 大，20 日均成交值 ≥ 1,000 萬，排除建材營造",
        f"- 持股：{p['n']} 檔等權重，資金 {p['capital']:,.0f} 元，零股",
        f"- 緩衝：持股仍在前 {p['keep_rank']} 名就續抱，只補空出的名額",
        f"- 成本：手續費 0.1425% × {p['commission_discount']*10:g} 折、零股每筆最低 "
        f"{p['min_commission']:g} 元；證交稅 0.3%；每邊滑價 {p['slippage']:.1%}",
        f"- 時間：每月 {p['rebalance_day']} 日後第一個交易日收盤產生名單，隔天開盤成交",
        "",
        "## 結果",
        "",
        "| 組合 | 2012–2022 年化 | 最大回撤 | 2023 起年化 | 最大回撤 | 超額 vs 大型股等權重（2012–2022） | t | 年換手 | 手續費＋稅/年 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in ("PLAN", "NO_BUFFER", "LARGE_EQUAL", BENCHMARK):
        r = R[k]
        rel = ("–", "–") if k == "LARGE_EQUAL" else (
            _p(r["vs_large_is"]["active_return"], True), f"{r['vs_large_is']['t']:.2f}")
        L.append(f"| {names[k]} | {_p(r['is']['cagr'])} | {_p(r['is']['max_dd'])} | "
                 f"{_p(r['oos']['cagr'])} | {_p(r['oos']['max_dd'])} | {rel[0]} | {rel[1]} | "
                 f"{r['turnover']:.1f}x | {_p(r['fees_tax_per_year'])} |")
    L += [
        "",
        f"- 最差的 12 個月：實盤設計 {_p(R['PLAN']['worst_12m'], True)}，0050 "
        f"{_p(R[BENCHMARK]['worst_12m'], True)}。",
        f"- 相對 0050 最差的 12 個月 {_p(o['worst_12m_vs_0050'], True)}，最好的 12 個月 "
        f"{_p(o['best_12m_vs_0050'], True)}；單月勝過 0050 的比例 {o['months_beating_0050']:.0%}。",
        "",
        "## 逐年報酬",
        "",
        "| 年度 | 實盤設計 | 大型股等權重 | 0050 |",
        "|---|---:|---:|---:|",
    ]
    y = o["yearly"]
    for year in sorted(y["PLAN"]):
        L.append(f"| {year} | {_p(y['PLAN'][year], True)} | {_p(y['LARGE_EQUAL'].get(year), True)} | "
                 f"{_p(y[BENCHMARK].get(year), True)} |")
    L += [
        "",
        "## 怎麼讀這份檢查",
        "",
        "- 這份檢查只用來確認成本、換手和回撤的量級，沒有用它挑選任何參數。",
        "- 2023 年以後的資料在先前的研究已經看過，所以這裡沒有乾淨的樣本外；"
        f"真正的樣本外是 {p['fixed_on'][:7]} 起的實盤。",
        "- 滑價是估計值；實際成交價會在實盤每月記錄，用來修正這個假設。",
    ]
    return "\n".join(L) + "\n"


def save(o: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "check.json").write_text(json.dumps(o, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "REPORT.md").write_text(render(o), encoding="utf-8")
    (out / "config.json").write_text(json.dumps(o["plan"], ensure_ascii=False, indent=1),
                                     encoding="utf-8")
